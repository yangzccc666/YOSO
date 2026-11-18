const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

function setup(accepted) {
  const prompts = []
  const context = { exports: {}, window: { confirm: message => { prompts.push(message); return accepted } } }
  const source = fs.readFileSync(path.join(__dirname, '../src/confirmActions.ts'), 'utf8')
  vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context)
  return { ...context.exports, prompts }
}

for (const accepted of [false, true]) {
  test(`stop and close return explicit confirmation result: ${accepted}`, () => {
    const guard = setup(accepted)
    assert.equal(guard.confirmStop('抽帧任务 A'), accepted)
    assert.equal(guard.confirmClose('本机设置', '未保存的修改不会保留'), accepted)
    assert.equal(guard.prompts.length, 2)
    assert.match(guard.prompts[0], /抽帧任务 A/)
    assert.match(guard.prompts[0], /取消.*保持当前任务运行/)
    assert.match(guard.prompts[1], /未保存的修改不会保留/)
  })
}

test('operation is not executed when confirmation is cancelled', () => {
  let requests = 0
  const cancelled = setup(false)
  if (cancelled.confirmStop('训练')) requests++
  if (cancelled.confirmClose('设置')) requests++
  assert.equal(requests, 0)
  const confirmed = setup(true)
  if (confirmed.confirmStop('训练')) requests++
  assert.equal(requests, 1)
})
