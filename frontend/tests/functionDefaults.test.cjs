const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
const source=fs.readFileSync(path.join(__dirname,'../src/functionDefaults.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
const {workingValuesFor,defaultValuesFor,withoutSecrets,trainingParametersSnapshot,parseTrainingParameters}=context.exports
const item={id:'frames',pathFields:[{id:'folder'}],parameters:[
 {id:'fps',type:'number',default:5},{id:'flag',type:'boolean',default:true},
 {id:'new_field',type:'text',default:'new default'},{id:'credential',type:'password',default:''}
]}
test('saved defaults take precedence over stale browser cache and include new fields',()=>{
 const saved={...item,savedDefaults:{values:{paths:{folder:'/saved'},parameters:{fps:20,flag:false,limit:''}}}}
 const result=workingValuesFor(saved,JSON.stringify({paths:{folder:'/stale'},parameters:{fps:3}}))
 assert.equal(result.paths.folder,'/saved');assert.equal(result.parameters.fps,20)
 assert.equal(result.parameters.flag,false);assert.equal(result.parameters.limit,'');assert.equal(result.parameters.new_field,'new default')
 assert.equal(workingValuesFor(saved).parameters.fps,20,'fresh browser profile does not require localStorage')
})
test('legacy last-used values merge current schema and tolerate broken browser storage',()=>{
 assert.equal(workingValuesFor(item,JSON.stringify({paths:{folder:'/legacy'},parameters:{fps:2}})).parameters.new_field,'new default')
 assert.equal(workingValuesFor(item,'broken-json').parameters.fps,5)
 assert.equal(workingValuesFor(item,'{"paths":"bad","parameters":[]}').paths.folder,'')
 assert.equal(defaultValuesFor(item).parameters.fps,5)
})
test('plain passwords and connection tokens are stripped without disabling remember-password',()=>{
 const safe=withoutSecrets(item,{paths:{},parameters:{credential:'secret',remote_password:'secret',remote_connection_token:'token',remember_password:true,fps:0}})
 assert.equal(safe.parameters.credential,'');assert.equal(safe.parameters.remote_password,'');assert.equal(safe.parameters.remote_connection_token,'')
 assert.equal(safe.parameters.remember_password,true);assert.equal(safe.parameters.fps,0)
})
test('training snapshots retain current settings and deliberately leave the device blank',()=>{
 const result=parseTrainingParameters(trainingParametersSnapshot({epochs:300,batch:8,device:'2',data:'/data.yaml',remote_password:'secret'}))
 assert.equal(result.epochs,300);assert.equal(result.batch,8);assert.equal(result.device,'');assert.equal(result.data,'/data.yaml');assert.equal(result.remote_password,undefined)
 assert.equal(Object.keys(parseTrainingParameters('bad-json')).length,0)
})
