const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{},Date}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(require('node:path').join(__dirname,'../src/packageNaming.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
const {buildPackageTitle,parsePackageTitle,bumpPackageVersion,comparePackageVersions,packageTitleWarning,localDate}=context.exports
test('screenshot naming format, date and explicit title fields round trip',()=>{
 assert.equal(buildPackageTitle('ct2','2',[1,1,0],'2026-11-17'),'CT2-2#1.1.0.1117')
 const parsed=parsePackageTitle('CT2-2#1.1.0.1117');assert.equal(parsed.line,'CT2');assert.equal(parsed.station,2)
 assert.equal(localDate(new Date(2026,0,2)),'2026-01-02')
})
test('major minor patch reset lower positions without changing base',()=>{
 const original=[1,2,3]
 assert.equal(bumpPackageVersion(original,'major').join('.'),'2.0.0')
 assert.equal(bumpPackageVersion(original,'minor').join('.'),'1.3.0')
 assert.equal(bumpPackageVersion(original,'patch').join('.'),'1.2.4')
 assert.equal(original.join('.'),'1.2.3')
 assert(comparePackageVersions([1,10,0],[1,9,9])>0)
 assert.throws(()=>bumpPackageVersion([9999,0,0],'major'))
})
test('invalid names and dates do not generate misleading titles',()=>{
 for(const args of [['中','2',[1,0,0],'2026-11-17'],['CT2','0',[1,0,0],'2026-11-17'],['CT2','2',[0,0,0],'2026-11-17'],['CT2','2',[1,0,0],'2026-02-29'],['CT2','2',[NaN,0,0],'2026-11-17']])assert.throws(()=>buildPackageTitle(...args))
 assert.equal(parsePackageTitle('CT2-2#1.0.0.0230'),null)
 assert.equal(parsePackageTitle('manual-title'),null)
 assert.equal(parsePackageTitle('CT2-2#1.0.0.1117\n'),null)
})
test('duplicate and label-order reminders are scoped to line and workstation',()=>{
 const history=[{line:'CT2',station:2,version:[1,1,0],latest:{labels:['MARK','PICK']}}]
 assert(packageTitleWarning('CT2-2#1.1.0.1117',['MARK','PICK'],history).includes('未递增'))
 assert(packageTitleWarning('CT2-2#1.1.1.1117',['PICK','MARK'],history).includes('主版号'))
 assert.equal(packageTitleWarning('CT2-2#2.0.0.1117',['PICK','MARK'],history),'')
 assert.equal(packageTitleWarning('CT2-3#1.0.0.1117',['NEW'],history),'')
 assert.equal(packageTitleWarning('manual title',['NEW'],history),'')
})
