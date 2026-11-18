const {test} = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const source = fs.readFileSync(path.join(__dirname, '../src/jetsonModelLabels.ts'), 'utf8')
const context = {exports:{}}
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const parse = text => Array.from(context.exports.parseExportModelPaths(text))
test('PT models preserve input order, spaces, Unicode and remove repeated paths',()=>{
 assert.deepEqual(parse(' /a/中文 a.pt\r\n\n/b/b.pt\r\n/a/中文 a.pt '),['/a/中文 a.pt','/b/b.pt'])
})
test('the existing JSON-array batch format is supported',()=>{
 assert.deepEqual(parse('["/a.pt", " /b.pt ", "/a.pt"]'),['/a.pt','/b.pt'])
 assert.deepEqual(parse(''),[])
})
test('malformed JSON and non-path items are explained',()=>{
 assert.throws(()=>parse('[bad'),/格式不正确/)
 assert.throws(()=>parse('[1,"/a.pt"]'),/必须是文本/)
})
