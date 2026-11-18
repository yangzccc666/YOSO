const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const path=require('node:path')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{},Date}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/inferenceVideos.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
const {videoBasename,videoRecordLabel,filterInferenceVideos}=context.exports
const rows=[
 {id:'old',scene:'磁探3号线',station:'工位2',path:'/videos/中文.mp4',updatedAt:'2026-11-16T08:00:00+00:00',available:true},
 {id:'new',scene:'TCL',station:'工位1',path:'C:\\videos\\昼夜.mp4',updatedAt:'2026-11-17T08:00:00+00:00',available:false},
]
test('video records sort newest first without mutating shared source',()=>{
 assert.equal(filterInferenceVideos(rows,'').map(row=>row.id).join(','),'new,old')
 assert.equal(rows[0].id,'old')
})
test('search supports Chinese scene, workstation, filename and case insensitive multiple terms',()=>{
 assert.equal(filterInferenceVideos(rows,'磁探 工位2 中文').length,1)
 assert.equal(filterInferenceVideos(rows,'tcl 昼夜')[0].id,'new')
 assert.equal(filterInferenceVideos(rows,'TCL 工位2').length,0)
})
test('dropdown labels distinguish scene station filename date and missing file',()=>{
 assert.equal(videoBasename(rows[0].path),'中文.mp4')
 assert.equal(videoBasename(rows[1].path),'昼夜.mp4')
 const label=videoRecordLabel(rows[1]);for(const text of ['TCL','工位1','昼夜.mp4','2026','文件不存在'])assert(label.includes(text))
 assert(videoRecordLabel({...rows[0],scene:'',station:''}).includes('未分类场景 · 未分类工位'))
})
