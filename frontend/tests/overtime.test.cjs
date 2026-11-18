const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/overtime.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {createActivityBuffer,isOvertimeActivity,acceptsOvertimeStatus}=context.exports
function fixture(){
 let now=0,id=0;const timers=new Map(),sent=[]
 const clock={now:()=>now,set:(fn,delay)=>{timers.set(++id,{fn,at:now+delay});return id},clear:id=>timers.delete(id)}
 const buffer=createActivityBuffer(at=>sent.push(at),clock)
 const advance=delta=>{const end=now+delta;for(;;){const next=[...timers].filter(([,v])=>v.at<=end).sort((a,b)=>a[1].at-b[1].at)[0];if(!next)break;now=next[1].at;timers.delete(next[0]);next[1].fn()}now=end}
 return {buffer,advance,sent,timers}
}
test('only trusted input in a visible focused app can extend overtime',()=>{
 assert.equal(isOvertimeActivity({isTrusted:true},true,true),true)
 assert.equal(isOvertimeActivity({isTrusted:false},true,true),false)
 assert.equal(isOvertimeActivity({isTrusted:true},false,true),false)
 assert.equal(isOvertimeActivity({isTrusted:true},true,false),false)
})
test('input is coalesced but reports the last actual input timestamp, not timer firing time',()=>{
 const f=fixture();f.buffer.activity();assert.deepEqual(f.sent,[0]);f.advance(1000);f.buffer.activity();f.advance(1000);f.buffer.activity();f.advance(3000)
 assert.deepEqual(f.sent,[0,2000]);assert.equal(f.timers.size,0)
})
test('idle has no heartbeat and background polling cannot extend activity',()=>{
 const f=fixture();f.buffer.activity();f.advance(3600000);assert.deepEqual(f.sent,[0]);assert.equal(f.timers.size,0)
})
test('many pointer events use only one trailing timer',()=>{
 const f=fixture();for(let i=0;i<1000;i++)f.buffer.activity();assert.equal(f.timers.size,1);f.advance(5000);assert.deepEqual(f.sent,[0,0])
})
test('leaving the app flushes pending input and pagehide can persist exact pending timestamp',()=>{
 const f=fixture();f.buffer.activity();f.advance(1200);f.buffer.activity();f.buffer.flush();assert.deepEqual(f.sent,[0,1200]);f.advance(500);f.buffer.activity()
 assert.equal(f.buffer.takePending(),1700);assert.equal(f.buffer.takePending(),null);assert.equal(f.timers.size,0)
 f.advance(50000);assert.deepEqual(f.sent,[0,1200])
})
test('unmount cancels delayed writes and older responses cannot revert a confirmed record',()=>{
 const f=fixture();f.buffer.activity();f.advance(1000);f.buffer.activity();f.buffer.dispose();f.advance(5000);assert.deepEqual(f.sent,[0])
 const current={revision:5,serverTime:'2026-11-12T10:00:00.000+00:00'}
 assert.equal(acceptsOvertimeStatus(current,{revision:4,serverTime:'2026-11-12T11:00:00.000+00:00'}),false)
 assert.equal(acceptsOvertimeStatus(current,{revision:5,serverTime:'2026-11-12T09:00:00.000+00:00'}),false)
 assert.equal(acceptsOvertimeStatus(current,{revision:6,serverTime:'2026-11-12T10:00:00.000+00:00'}),true)
})
