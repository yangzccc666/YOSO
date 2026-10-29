const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')

// Use the project's compiler and Node's test runner: no new test dependency.
const source = fs.readFileSync(path.join(__dirname, '../src/videoPlaybackShortcut.ts'), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
const context = { exports: {} }
vm.runInNewContext(compiled, context)
const { shouldToggleVideoOnSpace } = context.exports
const { getVideoPlaybackAction } = context.exports

function setup() {
  const body = { nodeType: 1, closest: () => null }
  const document = { hidden: false, activeElement: body, querySelector: () => null }
  const preview = { isConnected: true, ownerDocument: document, contains: target => Boolean(target.inside) }
  const event = { key: ' ', target: body }
  return { document, preview, event, accepted: (overrides = {}) => shouldToggleVideoOnSpace({ ...event, ...overrides }, preview) }
}

test('Space works on the page and focused player', () => {
  const { accepted } = setup()
  assert.equal(accepted(), true)
  assert.equal(accepted({ target: { nodeType: 1, inside: true, closest: () => null } }), true)
  assert.equal(accepted({ key: 'Spacebar' }), true)
})

test('non-Space keys and modified shortcuts remain untouched', () => {
  const { accepted } = setup()
  for (const key of ['Enter', 'Escape', 'Tab', 'ArrowRight', 'Process', 'Unidentified']) assert.equal(accepted({ key }), false)
  for (const flag of ['ctrlKey', 'altKey', 'metaKey', 'shiftKey', 'isComposing', 'defaultPrevented']) assert.equal(accepted({ [flag]: true }), false)
  assert.equal(accepted({ keyCode: 229 }), false)
})

test('no interception while editing text, passwords, numbers or checkboxes', () => {
  const { accepted } = setup()
  for (const type of ['text', 'password', 'number', 'checkbox', 'file', 'radio', 'search']) {
    const input = { nodeType: 1, type, closest: selector => selector === 'input' ? input : null }
    assert.equal(accepted({ target: input }), false, type)
  }
  for (const kind of ['textarea', 'select', '[role="textbox"]', '[role="combobox"]', '[role="searchbox"]', '[role="spinbutton"]']) {
    const field = { nodeType: 1, closest: selector => selector.includes(kind) ? field : null }
    assert.equal(accepted({ target: field }), false, kind)
  }
  assert.equal(accepted({ target: { nodeType: 1, isContentEditable: true, closest: () => null } }), false)
})

test('slider and playback buttons work only inside the preview', () => {
  const { accepted } = setup()
  const slider = { nodeType: 1, type: 'range', inside: true, closest: selector => selector === 'input' ? slider : null }
  assert.equal(accepted({ target: slider }), true)
  slider.inside = false
  assert.equal(accepted({ target: slider }), false)
  const button = { nodeType: 1, inside: true, closest: selector => selector.startsWith('button,') ? button : null }
  assert.equal(accepted({ target: button }), true)
  button.inside = false
  assert.equal(accepted({ target: button }), false)
})

test('open modals and hidden or unmounted players cannot control a video', () => {
  const { document, preview, event, accepted } = setup()
  document.querySelector = () => ({})
  assert.equal(accepted(), false)
  document.querySelector = () => null
  document.hidden = true
  assert.equal(accepted(), false)
  document.hidden = false
  preview.isConnected = false
  assert.equal(accepted(), false)
  assert.equal(shouldToggleVideoOnSpace(event, null), false)
})

test('falls back to the focused element and consumes repeats without treating them as text', () => {
  const { document, accepted } = setup()
  assert.equal(accepted({ target: null, repeat: true }), true)
  document.activeElement = { nodeType: 1, isContentEditable: true, closest: () => null }
  assert.equal(accepted({ target: null }), false)
})

test('comma steps backward and period steps forward in English and Chinese layouts', () => {
  const { event, preview } = setup()
  for (const key of [',', '，']) assert.equal(getVideoPlaybackAction({ ...event, key }, preview), 'backward')
  for (const key of ['.', '。', '．']) assert.equal(getVideoPlaybackAction({ ...event, key }, preview), 'forward')
  assert.equal(getVideoPlaybackAction(event, preview), 'toggle')
  for (const key of [',', '.', '，', '。']) {
    assert.equal(getVideoPlaybackAction({ ...event, key, isComposing: true }, preview), null)
    assert.equal(getVideoPlaybackAction({ ...event, key, ctrlKey: true }, preview), null)
    assert.equal(getVideoPlaybackAction({ ...event, key, target: {nodeType: 1, isContentEditable: true, closest: () => null} }, preview), null)
  }
})

test('stale polling cannot rewind an acknowledged frame step', () => {
  const input = fs.readFileSync(path.join(__dirname, '../src/playbackStatus.ts'), 'utf8')
  const code = ts.transpileModule(input, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText
  const scope = {exports: {}}
  vm.runInNewContext(code, scope)
  const { retainNewestPlaybackStatus } = scope.exports
  const current = {id: 'one', controlGeneration: 3, frameVersion: 20, frameCount: 20, frameStepPending: false}
  assert.equal(retainNewestPlaybackStatus(current, {...current, controlGeneration: 2}), current)
  assert.equal(retainNewestPlaybackStatus(current, {...current, frameVersion: 19}), current)
  assert.equal(retainNewestPlaybackStatus(current, {...current, frameStepPending: true}), current)
  const back = {...current, controlGeneration: 4, sourceFrameIndex: 5}
  assert.equal(retainNewestPlaybackStatus(current, back), back)
})
