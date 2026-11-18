export const CLIP_SAVE_AS = '另存为新视频'
export const CLIP_REPLACE_SOURCE = '替换原视频（自动备份）'
export function videoReplacementConfirmation(paths: Record<string, string>) {
  const targets = [paths.video_folder ? `文件夹：${paths.video_folder}（其中所有支持的视频）` : '', paths.video_file ? `单个视频：${paths.video_file}` : ''].filter(Boolean).join('\n')
  return `确认在原位置替换视频吗？\n${targets}\n\n将先精确裁剪、验证结果并备份原视频，再替换原路径。失败不会替换原文件；每个成功视频保留独立备份，需额外磁盘空间。批量中途停止不会撤销已完成的替换。\n备份位置：原目录/.video_clip_backups/日期时间_编号/原文件名。\n是否继续？`
}
