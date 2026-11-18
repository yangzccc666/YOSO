const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
const source=fs.readFileSync(path.join(__dirname,'../src/helpTip.ts'),'utf8')
vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {helpTipPosition}=context.exports
test('help opens below a trigger with room and stays inside the right edge',()=>{
 const pos=helpTipPosition({left:900,top:30,bottom:56},{width:360,height:160},{width:1000,height:800})
 assert.equal(pos.left,628)
 assert.equal(pos.top,64)
 assert(pos.top+160<=788)
})
test('bottom-of-window help flips above, without clipping a scrolled panel',()=>{
 const pos=helpTipPosition({left:4,top:740,bottom:766},{width:360,height:180},{width:1000,height:800})
 assert.equal(pos.left,12)
 assert.equal(pos.top,552)
 assert(pos.top+180<740)
})
test('long help is height-limited and scrollable even in a small viewport',()=>{
 for (const viewport of [{width:390,height:600},{width:800,height:260}]) {
  const pos=helpTipPosition({left:viewport.width-30,top:100,bottom:126},{width:Math.min(360,viewport.width-24),height:1200},viewport)
  assert(pos.left>=12)
  assert(pos.top>=12)
  assert(pos.maxHeight>0 && pos.maxHeight<=400)
  assert(pos.top+pos.maxHeight<=viewport.height-12)
 }
})
