const {test}=require('node:test')
const assert=require('node:assert/strict')
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),ts=require('typescript')
const context={exports:{}}
vm.runInNewContext(ts.transpileModule(fs.readFileSync(path.join(__dirname,'../src/videoClipOptions.ts'),'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText,context)
test('replace confirmation identifies folder batch and single video with backup and cancellation semantics',()=>{
 const text=context.exports.videoReplacementConfirmation({video_folder:'C:\\生产线\\工位2',video_file:'/videos/工位1.mp4'})
 for(const phrase of ['C:\\生产线\\工位2','其中所有支持的视频','/videos/工位1.mp4','备份','失败不会替换','不会撤销已完成','.video_clip_backups'])assert(text.includes(phrase))
 assert.notEqual(context.exports.CLIP_SAVE_AS,context.exports.CLIP_REPLACE_SOURCE)
})
