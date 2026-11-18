export type InferenceVideoRecord = {
  id: string; path: string; scene: string; station: string; createdAt: string; updatedAt: string
  lastUsedAt: string | null; useCount: number; available: boolean
}
export const videoBasename = (path: string) => path.split(/[\\/]/).pop() || path
export const videoRecordDate = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false })
export const videoRecordLabel = (row: InferenceVideoRecord) => `${row.scene || '未分类场景'} · ${row.station || '未分类工位'} · ${videoBasename(row.path)} · ${videoRecordDate(row.updatedAt)}${row.available ? '' : '（文件不存在）'}`
export function filterInferenceVideos(rows: InferenceVideoRecord[], query: string) {
  const tokens = query.toLocaleLowerCase().trim().split(/\s+/).filter(Boolean)
  return rows.filter(row => tokens.every(token => `${row.scene} ${row.station} ${row.path}`.toLocaleLowerCase().includes(token)))
    .sort((a,b) => Date.parse(b.updatedAt) - Date.parse(a.updatedAt))
}
