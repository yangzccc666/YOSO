const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{},require:()=>({REMEMBERED_PASSWORD_VALUE:'__YOLO_REMEMBERED_SSH_PASSWORD__'})}
const source=fs.readFileSync(path.join(__dirname,'../src/trainingSessions.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {trainingSessionConnection,trainingSessionFailedWithOOM}=context.exports
test('training reconnect and stop do not send a different device password',()=>{
 const target={host:'server-a',port:22,username:'dell'}
 const other=trainingSessionConnection({remote_host:'server-b',remote_username:'dell',remote_password:'wrong-host-secret'},target)
 assert.equal(other.remote_host,'server-a');assert.equal(other.remote_password,'__YOLO_REMEMBERED_SSH_PASSWORD__')
 assert(!JSON.stringify(other).includes('wrong-host-secret'))
 const same=trainingSessionConnection({remote_host:' SERVER-A ',remote_username:'dell',remote_password:'typed-secret',remember_password:false},target)
 assert.equal(same.remote_password,'typed-secret');assert.equal(same.remember_password,false)
 for (const changes of [{remote_port:2222},{remote_username:'root'}]) {
  const result=trainingSessionConnection({remote_host:'server-a',remote_username:'dell',remote_password:'must-not-forward',...changes},target)
  assert.notEqual(result.remote_password,'must-not-forward')
 }
})
test('OOM label appears for failed OOM tasks, not running or recovered tasks',()=>{
 assert(trainingSessionFailedWithOOM({status:'failed',error:'torch.OutOfMemoryError: CUDA out of memory',message:'失败'}))
 assert(trainingSessionFailedWithOOM({status:'failed',error:null,message:'日志包含 OOM／内存或显存不足错误'}))
 for(const status of ['running','disconnected','completed','stopped']) {
  assert.equal(trainingSessionFailedWithOOM({status,error:'CUDA out of memory',message:'OOM'}),false)
 }
 assert.equal(trainingSessionFailedWithOOM({status:'failed',error:null,message:'模型文件未找到'}),false)
 assert.equal(trainingSessionFailedWithOOM({status:'failed',error:null,message:'退出码 137，可能是系统内存不足，不能仅凭退出码确认 OOM'}),false)
})
