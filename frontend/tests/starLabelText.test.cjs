const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const source = fs.readFileSync(path.join(__dirname, '../src/starLabelText.ts'), 'utf8')
const context = { exports: {} }
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS } }).outputText, context)
const { formatStarLabelText } = context.exports

test('the provided eight-class example becomes one name per line', () => {
  const names = ['MARK', 'start_okp', 'hold_inspect', 'hold_mark', 'OKP', 'PICK', 'NGO', 'start_NGO']
  assert.equal(formatStarLabelText(names.map((name, i) => `${i}: ${name}`).join('\n')), names.join('\n'))
})

test('accepts Windows newlines, blank lines, indentation and Chinese colons', () => {
  assert.equal(formatStarLabelText('\uFEFF 0 ： 中文类别\r\n\r\n  1:  start_NGO  \r\n'), '中文类别\nstart_NGO')
})

test('supports existing comma-separated input and plain names without altering names', () => {
  assert.equal(formatStarLabelText('0: MARK, 1：start_okp，2:hold_inspect'), 'MARK\nstart_okp\nhold_inspect')
  assert.equal(formatStarLabelText('MARK\nstart_okp\nhold_inspect'), 'MARK\nstart_okp\nhold_inspect')
})

test('preserves pasted order and duplicates instead of sorting or silently changing IDs', () => {
  assert.equal(formatStarLabelText('5: PICK\n0: MARK\n10: PICK'), 'PICK\nMARK\nPICK')
})

test('only removes a leading integer index and colon, not numbers within names', () => {
  assert.equal(formatStarLabelText('0: stage_2:OK\n1: 123\nPart: NG\n2_start\n3. model'), 'stage_2:OK\n123\nPart: NG\n2_start\n3. model')
})

test('formatting twice is stable for ordinary class names', () => {
  const once = formatStarLabelText('0: MARK\n1: start_NGO')
  assert.equal(formatStarLabelText(once), once)
})

test('empty names after an index are rejected rather than dropping a class', () => {
  assert.throws(() => formatStarLabelText('0: MARK\n1:\n2: NGO'), /第 2 项只有序号/)
  assert.throws(() => formatStarLabelText('0：  '), /没有类别名称/)
})

test('empty input and NUL are rejected', () => {
  assert.throws(() => formatStarLabelText(' \r\n ， ,'), /粘贴类别名称/)
  assert.throws(() => formatStarLabelText('0: bad\0name'), /空字符/)
})
