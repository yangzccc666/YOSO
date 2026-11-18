const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
const source=fs.readFileSync(path.join(__dirname,'../src/runTiming.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {runElapsedSeconds,formatRunDuration}=context.exports
test('duration renders subsecond, minute, hour, and multi-day runs',()=>{
 for(const [seconds,text] of [[0,'0秒'],[.35,'不足1秒'],[59.9,'59秒'],[60,'1分 0秒'],[3661,'1小时 1分 1秒'],[90061,'1天 1小时 1分 1秒']])
  assert.equal(formatRunDuration(seconds),text)
 for(const invalid of [null,NaN,Infinity,-1]) assert.equal(formatRunDuration(invalid),'暂无')
})
test('stored duration is authoritative and does not keep growing after completion',()=>{
 const record={status:'completed',startedAt:'2026-11-06T12:00:00Z',finishedAt:'2026-11-06T13:00:00Z',elapsedSeconds:2.375}
 assert.equal(runElapsedSeconds(record,Date.now()+1e8),2.375)
 assert.equal(runElapsedSeconds({...record,status:'failed'}),2.375)
 assert.equal(runElapsedSeconds({...record,status:'stopped'}),2.375)
})
test('legacy timestamps are supported without fabricating missing or backwards timing',()=>{
 const record={status:'completed',startedAt:'2026-11-06T12:00:00+08:00',finishedAt:'2026-11-06T04:01:00Z'}
 assert.equal(runElapsedSeconds(record),60)
 assert.equal(runElapsedSeconds({...record,status:'running'},Date.parse('2026-11-06T04:02:00Z')),120)
 assert.equal(runElapsedSeconds({...record,startedAt:''}),null)
 assert.equal(runElapsedSeconds({...record,finishedAt:'2025-01-01'}),null)
 assert.equal(runElapsedSeconds({...record,elapsedSeconds:null}),null)
})
