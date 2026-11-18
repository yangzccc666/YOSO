const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs')
const vm=require('node:vm')
const ts=require('typescript')
const context={exports:{}}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(require('node:path').join(__dirname,'../src/runHistoryLabel.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020}}).outputText,context)
const {runHistoryLabel}=context.exports
const record=(details={},extra={})=>({id:'abcdef123456',functionId:'video_clip',name:'视频裁剪',details,outputs:[],...extra})
test('old histories distinguish source file and preserve its full path',()=>{
 const label=runHistoryLabel(record({'单个视频（可选）':'/项目/工位2.mp4','输出文件夹':'/out'}))
 assert.equal(label.title,'视频裁剪 · 工位2.mp4');assert.equal(label.context,'/项目/工位2.mp4')
 assert(!label.title.includes('out'))
})
test('input priority, Windows paths, multi-model count and safe missing data',()=>{
 const label=runHistoryLabel(record({'模型文件':'C:\\models\\one.pt\nC:\\models\\two.pt\nC:\\models\\three.pt','视频文件':'C:\\videos\\sample.mp4'}))
 assert(label.title.includes('sample.mp4 + one.pt 等4项'))
 assert.equal(runHistoryLabel(record({}, {details:undefined})).title,'视频裁剪 · #123456')
})
test('existing training name and frame task name are preserved without duplication',()=>{
 assert.equal(runHistoryLabel(record({'模型权重':'/x/model.pt'},{functionId:'yolo_model_training',name:'YOLO 训练 · yolo26_1113_1030'})).title,'YOLO 训练 · yolo26_1113_1030')
 assert.equal(runHistoryLabel(record({'视频文件':'/x/工位.mp4'},{name:'视频抽帧 · 工位.mp4 · scene'})).title,'视频抽帧 · 工位.mp4 · scene')
})
test('parameters are shown from saved snapshots, not current form values',()=>{
 const label=runHistoryLabel(record({'关键参数':'张/秒=5 · 开始=01:30'}))
 assert.equal(label.parameters,'张/秒=5 · 开始=01:30')
 assert.equal(runHistoryLabel(record({'构建精度':'FP16'})).parameters,'FP16')
})
test('outputs are only a fallback and code-like filenames remain plain strings',()=>{
 const label=runHistoryLabel(record({}, {outputs:['/out/<script>.star']}))
 assert.equal(label.title,'视频裁剪 · <script>.star')
 assert.equal(runHistoryLabel(record({'模型文件':'/x/model.pt'}, {outputs:['/out/new.star']})).title,'视频裁剪 · model.pt')
})
