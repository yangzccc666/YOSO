const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const context = { exports: {} }
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname, '../src/frameStepHold.ts'), 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
}).outputText, context)
const { createFrameStepHold } = context.exports

const flush = async () => { for (let i = 0; i < 6; i++) await Promise.resolve() }
function fixture(latency = 20, result = true) {
  let now = 0, sequence = 0, active = 0, peak = 0
  const timers = new Map(), calls = [], changes = []
  const state = { supported: true, busy: false, index: 50, count: 100 }
  const clock = {
    now: () => now,
    set: (callback, delay) => { const id = ++sequence; timers.set(id, { at: now + delay, callback }); return id },
    clear: id => timers.delete(id),
  }
  const controller = createFrameStepHold(() => state, direction => {
    calls.push({ at: now, direction }); active++; peak = Math.max(peak, active); state.busy = true
    return new Promise(resolve => clock.set(() => {
      active--; state.busy = false
      if (result !== false) state.index += direction
      resolve(result)
    }, latency))
  }, direction => changes.push(direction), clock)
  const advance = async amount => {
    const end = now + amount
    for (let limit = 0; limit < 10000; limit++) {
      const next = [...timers].filter(([, task]) => task.at <= end).sort((a, b) => a[1].at - b[1].at)[0]
      if (!next) break
      now = next[1].at; timers.delete(next[0]); next[1].callback(); await flush()
    }
    now = end; await flush()
  }
  return { controller, state, calls, changes, advance, peak: () => peak }
}

test('short pointer hold produces one frame and release cancels repetition', async () => {
  const f = fixture()
  assert.equal(f.controller.start(1), true)
  await f.advance(50); f.controller.stop(); await f.advance(1000)
  assert.equal(f.calls.length, 1)
  assert.equal(f.state.index, 51)
  assert.deepEqual(f.changes, [1, null])
})

test('long hold waits 350ms then repeats sequentially in the selected direction', async () => {
  const f = fixture()
  f.controller.start(-1)
  await f.advance(349); assert.equal(f.calls.length, 1)
  await f.advance(1); assert.equal(f.calls.length, 2)
  await f.advance(450)
  assert.ok(f.calls.length >= 6)
  assert.ok(f.calls.every(call => call.direction === -1))
  assert.equal(f.peak(), 1)
  f.controller.stop()
})

test('slow inference never queues overlapping requests or resumes after release', async () => {
  const f = fixture(600)
  f.controller.start(1)
  await f.advance(599); assert.equal(f.calls.length, 1)
  await f.advance(31); assert.equal(f.calls.length, 2)
  f.controller.stop(); await f.advance(2000)
  assert.equal(f.calls.length, 2)
  assert.equal(f.peak(), 1)
})

test('pending frame acknowledgement blocks repetition until state is ready', async () => {
  const f = fixture()
  f.controller.start(1); await f.advance(25)
  f.state.busy = true; await f.advance(900)
  assert.equal(f.calls.length, 1)
  f.state.busy = false; await f.advance(65)
  assert.equal(f.calls.length, 2)
  f.controller.stop()
})

test('first and last frames stop a held direction without extra requests', async () => {
  for (const direction of [-1, 1]) {
    const f = fixture()
    f.state.index = direction < 0 ? 1 : 98
    f.controller.start(direction); await f.advance(1000)
    assert.equal(f.calls.length, 1)
    assert.equal(f.changes.at(-1), null)
    assert.equal(f.controller.start(direction), false)
  }
})

test('request failures stop repetition instead of continuously retrying', async () => {
  const f = fixture(20, false)
  f.controller.start(1); await f.advance(1000)
  assert.equal(f.calls.length, 1)
  assert.equal(f.changes.at(-1), null)
})

test('unsupported sessions and external busy state cannot start a hold', async () => {
  const f = fixture()
  f.state.busy = true; assert.equal(f.controller.start(1), false)
  f.state.busy = false; f.state.supported = false; assert.equal(f.controller.start(-1), false)
  f.state.supported = true; f.controller.start(1); await f.advance(25)
  f.state.supported = false; f.controller.reconcile(); await f.advance(1000)
  assert.equal(f.calls.length, 1)
  assert.equal(f.changes.at(-1), null)
})

test('stop/restart cannot share an old request or its timer', async () => {
  const f = fixture(500)
  f.controller.start(1); f.controller.stop()
  assert.equal(f.controller.start(-1), false)
  await f.advance(501)
  assert.equal(f.controller.start(-1), true)
  await f.advance(100); f.controller.stop(); await f.advance(1500)
  assert.deepEqual(f.calls.map(call => call.direction), [1, -1])
  assert.equal(f.peak(), 1)
})
