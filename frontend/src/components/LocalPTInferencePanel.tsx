import { useEffect, useRef, useState } from 'react'
import { getLocalPTInferenceStatus, seekLocalPTInference, setLocalPTInferencePaused, startLocalPTInference, stopLocalPTInference } from '../api'
import { icons } from '../icons'
import type { PlatformTaskStatus, RemoteInferenceStatus, WorkingValues } from '../types'

type Props = { itemId: string; values: WorkingValues; platformTask: PlatformTaskStatus | null }
const ended = new Set(['completed', 'stopped', 'failed'])
const formatTime = (seconds: number) => {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0))
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

export function LocalPTInferencePanel({ itemId, values, platformTask }: Props) {
  const key = `processing-view:local-pt-session:${itemId}`
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem(key) || '')
  const [session, setSession] = useState<RemoteInferenceStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [expanded, setExpanded] = useState(false)
  const [visible, setVisible] = useState(true)
  const [scrubbing, setScrubbing] = useState<number | null>(null)
  const hideTimer = useRef<number | undefined>(undefined)
  const scrubRef = useRef<number | null>(null)
  const reveal = () => {
    setVisible(true)
    window.clearTimeout(hideTimer.current)
    hideTimer.current = window.setTimeout(() => setVisible(false), 3000)
  }

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const next = await getLocalPTInferenceStatus(sessionId)
        if (cancelled) return
        setSession(next)
        if (!ended.has(next.status)) timer = window.setTimeout(poll, 700)
      } catch (cause) {
        if (cancelled) return
        setError(cause instanceof Error ? cause.message : '读取本地任务失败')
        sessionStorage.removeItem(key)
        setSessionId('')
      }
    }
    void poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [key, sessionId])

  useEffect(() => {
    if (!expanded) return
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') setExpanded(false) }
    document.body.classList.add('remote-preview-open')
    window.addEventListener('keydown', escape)
    return () => { document.body.classList.remove('remote-preview-open'); window.removeEventListener('keydown', escape) }
  }, [expanded])
  useEffect(() => () => window.clearTimeout(hideTimer.current), [])

  const active = Boolean(session && !ended.has(session.status))
  const duration = session?.durationSeconds || 0
  const seekable = Boolean(active && duration > 0 && (session?.status === 'running' || session?.status === 'paused'))
  const pausable = Boolean(active && !session?.atEnd && session?.frameCount && (session?.status === 'running' || session?.status === 'paused'))

  const start = async () => {
    setBusy(true); setError(''); setSession(null)
    sessionStorage.removeItem(key)
    try {
      const next = await startLocalPTInference(values)
      sessionStorage.setItem(key, next.id)
      setSessionId(next.id)
      setSession(next)
    } catch (cause) { setError(cause instanceof Error ? cause.message : '启动失败') }
    finally { setBusy(false) }
  }
  const stop = async () => {
    if (!sessionId) return
    setBusy(true); setError('')
    try { setSession(await stopLocalPTInference(sessionId)) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '停止失败') }
    finally { setBusy(false) }
  }
  const toggle = async () => {
    if (!sessionId || !session) return
    setBusy(true); setError('')
    try { setSession(await setLocalPTInferencePaused(sessionId, session.status !== 'paused')) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '播放控制失败') }
    finally { setBusy(false) }
  }
  const commitSeek = async () => {
    const target = scrubRef.current
    if (target === null || !sessionId) return
    scrubRef.current = null
    setScrubbing(null)
    setError('')
    try { await seekLocalPTInference(sessionId, target) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '跳转失败') }
  }
  const status = session?.atEnd ? '已到结尾 · 可回看' : session?.status === 'paused' ? '视频已暂停' : session?.status === 'running' ? '本地检测中' : session?.message || '尚未开始'

  return <section className="remote-inference-section">
    <div className="remote-run-actions">
      <button className="run-button" disabled={busy || (!active && Boolean(platformTask))} onClick={() => void (active ? stop() : start())}>
        {active ? <icons.X size={18} /> : <icons.Play size={18} fill="currentColor" />}{active ? session?.atEnd ? '结束回看并清理' : '停止检测' : busy ? '正在启动…' : '开始本地检测'}
      </button>
      <span className={`remote-status ${session?.status || 'idle'}`}><i />{status}</span>
      <small>直接使用本机视频和 PT 权重；类别名称从权重读取，不需要 SSH。</small>
    </div>
    {error || session?.error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error || session?.error}</span></div> : null}
    <div className={`remote-preview ${expanded ? 'expanded' : ''}`} onMouseMove={reveal} onMouseLeave={() => setVisible(false)} onDoubleClick={() => setExpanded((current) => !current)} title="双击切换撑满窗口">
      <button className="preview-expand-button" onClick={() => setExpanded((current) => !current)} onDoubleClick={(event) => event.stopPropagation()} aria-label={expanded ? '退出撑满窗口' : '撑满整个窗口'}>{expanded ? '退出全窗口' : '撑满窗口'}</button>
      {session?.frameCount ? <img src={session.streamUrl} alt="本地 PT 检测预览画面" draggable={false} /> : <div className="remote-preview-empty"><icons.Play size={34} /><strong>{session?.message || '选择 PT 权重和视频，然后开始本地检测'}</strong></div>}
      {session ? <div className={`remote-preview-timeline ${visible ? '' : 'is-hidden'}`} onDoubleClick={(event) => event.stopPropagation()} onMouseMove={reveal}>
        <button className="remote-preview-playback-button" disabled={!pausable || busy} onClick={() => void toggle()} onPointerUp={(event) => event.currentTarget.blur()} aria-label={session.status === 'paused' ? '继续视频' : '暂停视频'}>{session.status === 'paused' ? <icons.Play size={16} fill="currentColor" /> : <icons.Pause size={16} fill="currentColor" />}</button>
        <span>{formatTime(scrubbing ?? session.positionSeconds)}</span>
        <input type="range" aria-label="视频进度，拖动后跳转" min={0} max={Math.max(duration, 0.01)} step="0.1" value={Math.min(scrubbing ?? session.positionSeconds, Math.max(duration, 0.01))} disabled={!seekable} onFocus={reveal} onChange={(event) => { const next = Number(event.target.value); scrubRef.current = next; setScrubbing(next) }} onPointerUp={() => void commitSeek()} onKeyUp={() => void commitSeek()} onBlur={() => void commitSeek()} />
        <span>{formatTime(duration)}</span>
      </div> : null}
    </div>
    <div className="remote-session-summary"><span>本地预览：{session?.frameCount || 0} 帧</span>{session && session.sourceFps > 0 ? <span>源视频：{session.sourceFps.toFixed(2)} FPS</span> : null}{duration > 0 ? <span>视频进度：{formatTime(session?.positionSeconds || 0)} / {formatTime(duration)}</span> : null}{session?.atEnd ? <span>剩余回看时间：{session.replayRemainingSeconds} 秒</span> : null}</div>
    <details className="run-output" open><summary><icons.ChevronRight size={18} />本地运行日志</summary><pre>{session?.logs.length ? session.logs.join('\n') : '启动后将在这里显示模型加载和处理结果'}</pre></details>
  </section>
}
