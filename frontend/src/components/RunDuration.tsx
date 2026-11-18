import { useEffect, useMemo, useState } from 'react'
import { formatRunDuration, runElapsedSeconds } from '../runTiming'
import type { RunHistoryRecord } from '../types'

export function RunDuration({ record, showLabel = true }: { record: RunHistoryRecord; showLabel?: boolean }) {
  const running = record.status === 'running'
  const anchor = useMemo(() => ({
    seconds: runElapsedSeconds(record), at: performance.now(),
    // Only timing primitives invalidate the anchor; polling logs must not reset it.
  }), [record.elapsedSeconds, record.startedAt, record.finishedAt, record.status])
  const [now, setNow] = useState(() => performance.now())
  useEffect(() => {
    if (!running) return
    const timer = window.setInterval(() => setNow(performance.now()), 1000)
    return () => window.clearInterval(timer)
  }, [running])
  const seconds = anchor.seconds === null ? null : anchor.seconds + (running ? Math.max(0, now - anchor.at) / 1000 : 0)
  return <span className="history-duration" title="从任务启动到平台确认结束的总时长，含准备、传输和运行中暂停；不含排队等待。断线重连按原始启动时间统计。">
    {showLabel ? running ? '已运行 ' : '耗时 ' : ''}{formatRunDuration(seconds)}
  </span>
}
