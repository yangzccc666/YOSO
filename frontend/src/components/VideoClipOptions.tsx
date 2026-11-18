import { CLIP_REPLACE_SOURCE, CLIP_SAVE_AS } from '../videoClipOptions'
import { HelpTip } from './HelpTip'
import type { WorkingValues } from '../types'

type Props = { values: WorkingValues; disabled: boolean; onChange: (id: string, value: string | boolean) => void }
export function VideoClipOptions({values, disabled, onChange}: Props) {
  const replace = values.parameters.clip_save_mode === CLIP_REPLACE_SOURCE
  const exact = replace || values.parameters.clip_exact_time !== false
  return <section className="clip-save-options" aria-label="视频裁剪保存设置">
    <div className="help-heading"><strong>裁剪保存方式</strong><HelpTip label="原位裁剪与另存说明">另存：保留原视频，可填写输出目录及文件名。替换原视频：保持原路径和文件名，先在临时文件中精确裁剪，验证时长与完整解码，再备份原视频并原子替换。支持MP4、M4V、MOV、MKV；其他格式请另存MP4。原始备份保存在源目录的.video_clip_backups子目录，运行日志和历史结果可查看路径；需额外磁盘空间。恢复时停止所有使用该视频的任务，再将备份复制回原位置。批量逐个提交，取消不回滚已成功的视频。</HelpTip></div>
    <div className="clip-save-buttons" role="group" aria-label="选择视频保存方式">
      {[CLIP_SAVE_AS, CLIP_REPLACE_SOURCE].map(mode => <button key={mode} type="button" aria-pressed={(mode === CLIP_REPLACE_SOURCE) === replace} disabled={disabled} onClick={() => {
        onChange('clip_save_mode', mode)
        if (mode === CLIP_REPLACE_SOURCE) { onChange('clip_exact_time', true); onChange('split_train_test', false) }
      }}>{mode}</button>)}
    </div>
    <div className="help-heading"><label className="clip-exact-toggle"><input type="checkbox" checked={exact} disabled={disabled || replace} onChange={event => onChange('clip_exact_time', event.target.checked)} />按帧精确裁剪时间（推荐）</label><HelpTip label="裁剪时间精度说明">保留原视频时间轴上开始时间至结束时间之前的已有帧，结束边界不包含。精确模式需要重新编码，不改变分辨率，但文件体积可能变化；精确度受原帧间隔限制，例如25FPS约0.04秒一帧，不能生成不存在的半帧。可变帧率和多段拼接存在逐帧取整；结果会检查时长，无法满足校验时不替换原视频。关闭后可使用不重新编码的快速模式，但起止时间不能严格保证。替换原视频始终启用精确模式。</HelpTip></div>
    <small>{replace ? '运行前再次确认；保留原路径和文件名，自动备份。不支持同时拆分训练集／测试集。' : exact ? '另存新视频，原文件保留；按帧精确裁剪并校验时间。' : '快速模式可能按关键帧偏移；要求时间严格对应时请开启精确裁剪。'}</small>
  </section>
}
