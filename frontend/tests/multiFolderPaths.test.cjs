const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const source = fs.readFileSync(path.join(__dirname, '../src/multiFolderPaths.ts'), 'utf8')
const context = { exports: {} }
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText, context)
const { appendFolderPaths } = context.exports

test('folder picker appends instead of replacing pasted paths', () => {
  assert.equal(appendFolderPaths('/data/视频 one\n/data/two', ['/data/three']), '/data/视频 one\n/data/two\n/data/three')
})
test('repeated folder selections are deduplicated and cancellation keeps input', () => {
  assert.equal(appendFolderPaths('/a\n/a\n\n/b', ['/a', '/c']), '/a\n/b\n/c')
  assert.equal(appendFolderPaths('/a', []), '/a')
})
test('quotes and Windows newlines are normalized without splitting spaces or drive letters', () => {
  assert.equal(appendFolderPaths('"C:\\Data One"\r\n\'D:\\Data Two\'\r\n', ['C:\\Data One', '/中文路径']), 'C:\\Data One\nD:\\Data Two\n/中文路径')
})
