const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/processRegions.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {validRegion,readRegions,rectangle}=context.exports
test('dragging in any direction creates a clamped normalized ROI',()=>{
 assert.deepEqual(Array.from(rectangle([.8,.9],[.2,.1])),[.2,.1,.8,.9])
 assert.deepEqual(Array.from(rectangle([-.3,-.2],[1.4,1.5])),[0,0,1,1])
})
test('empty, inverted, nonfinite and nonnumeric rectangles are rejected',()=>{
 for(const v of [null,[],[0,0,0,1],[.5,0,.2,1],[-1,0,1,1],[0,0,2,1],[0,0,NaN,1],[0,0,Infinity,1],[false,0,.5,1]])assert.equal(validRegion(v),false)
 assert.equal(validRegion([0,0,.5,1]),true)
})
test('region config preserves independent workstation keys and its source root',()=>{
 const config=readRegions(JSON.stringify({version:1,root:'/data/场景',regions:{'工位A/images':[0,0,.5,1],'工位B/images':[.5,0,1,1]}}))
 assert.equal(config.root,'/data/场景')
 assert.deepEqual(Array.from(config.regions['工位B/images']),[.5,0,1,1])
})
test('malformed or stale format is not interpreted as a full-frame ROI',()=>{
 for(const value of ['', '{}','null','[]','bad',JSON.stringify({version:0,root:'/',regions:{}}),JSON.stringify({version:1,root:'/',regions:[]}),JSON.stringify({version:1,root:'/',regions:{'.':[0,0,0,1]}})])assert.equal(readRegions(value),null)
})
