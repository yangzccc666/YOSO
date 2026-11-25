const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
function render(props, current) {
 let next
 const jsx=(type,props)=>({type,props})
 const context={exports:{},require:id=>{
  if(id==='react')return {useState:value=>[current===undefined?value:current,v=>{next=v}]}
  if(id==='react/jsx-runtime')return {jsx,jsxs:jsx}
  if(id==='../icons')return {icons:{ChevronDown:'arrow'}}
  throw Error(id)
 }}
 const source=fs.readFileSync(path.join(__dirname,'../src/components/CollapsibleSection.tsx'),'utf8')
 vm.runInNewContext(ts.transpileModule(source,{compilerOptions:{module:ts.ModuleKind.CommonJS,jsx:ts.JsxEmit.ReactJSX,target:ts.ScriptTarget.ES2020}}).outputText,context)
 const section=context.exports.CollapsibleSection(props)
 return {section,details:section.props.children,next:()=>next}
}
test('optional sections start collapsed and keep native keyboard disclosure semantics',()=>{
 const f=render({heading:'运行参数',children:'table',className:'workspace-section',count:12})
 assert.equal(f.section.props.className,'workspace-section')
 assert.equal(f.details.type,'details');assert.equal(f.details.props.open,false)
 assert.equal(f.details.props.children[0].type,'summary')
 f.details.props.onToggle({currentTarget:{open:false}});assert.equal(f.next(),false)
})
test('collapsed content remains in the tree instead of clearing or unmounting inputs',()=>{
 const child={type:'expensive-form',props:{value:'keep me'}}
 const f=render({heading:'SSH',children:child},false)
 assert.equal(f.details.props.open,false)
 assert.equal(f.details.props.children[1].props.children,child)
})
test('zero item count and heading help controls are preserved',()=>{
 const heading={type:'heading-with-help'}
 const f=render({heading,children:'content',count:0})
 assert.equal(f.details.props.open,false)
 const summary=f.details.props.children[0]
 assert.equal(summary.props.children[0],heading)
 assert.equal(summary.props.children[1].props.children[0].props.children[0],0)
 f.details.props.onToggle({currentTarget:{open:true}});assert.equal(f.next(),true)
})

test('fixed sections expose the form without a disclosure control',()=>{
 const {section}=render({fixed:true,heading:'数据路径',children:'form'})
 const [heading,body]=section.props.children
 assert.equal(heading,'数据路径')
 assert.equal(body.type,'div')
 assert.equal(body.props.children,'form')
})
