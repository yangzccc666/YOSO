import type { RunHistoryRecord } from './types'

type TimingRecord = Pick<RunHistoryRecord, 'status' | 'startedAt' | 'finishedAt' | 'elapsedSeconds'>

export function runElapsedSeconds(record: TimingRecord, now = Date.now()): number | null {
  const recorded = record.elapsedSeconds
  if (recorded === null) return null
  if (typeof recorded === 'number' && Number.isFinite(recorded) && recorded >= 0) return recorded
  const start = Date.parse(record.startedAt)
  const end = record.status === 'running' ? now : Date.parse(record.finishedAt)
  return Number.isFinite(start) && Number.isFinite(end) && end >= start ? (end - start) / 1000 : null
}

export function formatRunDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return '暂无'
  if (seconds > 0 && seconds < 1) return '不足1秒'
  const total = Math.floor(seconds)
  const days = Math.floor(total / 86400)
  const hours = Math.floor(total % 86400 / 3600)
  const minutes = Math.floor(total % 3600 / 60)
  const remaining = total % 60
  if (days) return `${days}天 ${hours}小时 ${minutes}分 ${remaining}秒`
  if (hours) return `${hours}小时 ${minutes}分 ${remaining}秒`
  if (minutes) return `${minutes}分 ${remaining}秒`
  return `${remaining}秒`
}
