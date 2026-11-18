const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{},require:()=>({REMEMBERED_PASSWORD_VALUE:'__YOLO_REMEMBERED_SSH_PASSWORD__'})}
const source=fs.readFileSync(path.join(__dirname,'../src/tensorRTQueue.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {retainTensorRTQueue,tensorRTReconnectParameters,tensorRTDispatchStatus}=context.exports
test('queued work is scheduled automatically but never reported as already building',()=>{
 const status=tensorRTDispatchStatus({queuedCount:2,runningCount:0,paused:false})
 assert.equal(status.title,'自动执行已开启，等待调度')
 assert.equal(status.blocked,false)
 assert.match(status.detail,/当前没有任务正在构建/)
 assert.equal(tensorRTDispatchStatus({queuedCount:2,runningCount:1,paused:false}).title,'自动执行中')
})
test('paused, closed and unresolved queues visibly say waiting work is not executing',()=>{
 for (const state of [{paused:true,pauseReason:'请输入密码'}, {closed:true}, {disconnectedCount:1}]) {
  const status=tensorRTDispatchStatus({queuedCount:1,runningCount:0,...state})
  assert.equal(status.blocked,true)
  assert.match(status.title,/尚未执行|不会执行|未执行/)
 }
 assert.equal(tensorRTDispatchStatus({queuedCount:1,paused:true,pauseReason:'请输入密码'}).detail,'请输入密码')
 assert.equal(tensorRTDispatchStatus({queuedCount:1,runningCount:1,paused:true}).title,'当前任务运行中，后续已暂停')
})
test('a slow queue poll cannot erase a newly submitted job or a pause action',()=>{
 const newest={instanceId:'a',version:20,items:['submitted'],paused:true}
 assert.equal(retainTensorRTQueue(newest,{instanceId:'a',version:19,items:[]}),newest)
 assert.equal(retainTensorRTQueue(newest,{instanceId:'a',version:20,items:[]}),newest)
 const next={instanceId:'a',version:21,items:['submitted']}
 assert.equal(retainTensorRTQueue(newest,next),next)
})
test('a restarted backend can replace a higher version from the old queue',()=>{
 const restarted={instanceId:'b',version:0,items:[]}
 assert.equal(retainTensorRTQueue({instanceId:'a',version:100},restarted),restarted)
 assert.equal(retainTensorRTQueue(null,restarted),restarted)
})
test('reconnect uses the saved device and never forwards a different device password',()=>{
 const target={host:'box-a',port:22,username:'wel'}
 const other=tensorRTReconnectParameters({host:'box-b',port:22,username:'wel',password:'other-secret'},target)
 assert.equal(other.host,'box-a');assert.equal(other.password,'__YOLO_REMEMBERED_SSH_PASSWORD__')
 assert(!JSON.stringify(other).includes('other-secret'))
 const matched=tensorRTReconnectParameters({host:' BOX-A ',username:'wel',password:'typed',remember_password:false},target)
 assert.equal(matched.password,'typed');assert.equal(matched.remember_password,false)
 const otherUser=tensorRTReconnectParameters({host:'box-a',port:22,username:'root',password:'root-secret'},target)
 assert.notEqual(otherUser.password,'root-secret')
 const otherPort=tensorRTReconnectParameters({host:'box-a',port:2222,username:'wel',password:'port-secret'},target)
 assert.notEqual(otherPort.password,'port-secret')
})
