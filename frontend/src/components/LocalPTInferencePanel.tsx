import { useEffect, useRef, useState } from 'react'
import { confirmStop } from '../confirmActions'
import { detectLocalModelLabels, getLocalPTInferenceStatus, seekLocalPTInference, setLocalPTInferencePaused, startLocalPTInference, stepLocalPTInference, stopLocalPTInference } from '../api'
import { icons } from '../icons'
import { HelpTip } from './HelpTip'
import { useVideoPlaybackShortcut } from '../hooks/useVideoPlaybackShortcut'
import { FrameStepControls } from './FrameStepControls'
import { retainNewestPlaybackStatus } from '../playbackStatus'
import type { PlatformTaskStatus, RemoteInferenceStatus, WorkingValues } from '../types'

type Props = { itemId: string; values: WorkingValues; platformTask: PlatformTaskStatus | null; onParameterChange: (id: string, value: string | number | boolean) => void }
const ended = new Set(['completed', 'stopped', 'failed'])
const formatTime = (seconds: number) => {
  const total = Math.max(0, Math.floor(Number.isFinite(seconds) ? seconds : 0))
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
}

export function LocalPTInferencePanel({ itemId, values, platformTask, onParameterChange }: Props) {
  const key = `processing-view:local-pt-session:${itemId}`
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem(key) || '')
  const [session, setSession] = useState<RemoteInferenceStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [readingLabels, setReadingLabels] = useState(false)
  const [labelNotice, setLabelNotice] = useState<{ model: string; message: string } | null>(null)
  const modelFile = String(values.paths.model_file || '').trim()
  const labelOwner = String(values.parameters.labels_model_file || '').trim()
  const labels = !labelOwner || labelOwner === modelFile ? String(values.parameters.labels || '') : ''
  const latestLabels = useRef({ modelFile, labels })
  latestLabels.current = { modelFile, labels }
  const readRequest = useRef(0)
  const [expanded, setExpanded] = useState(false)
  const [visible, setVisible] = useState(true)
  const [scrubbing, setScrubbing] = useState<number | null>(null)
  const hideTimer = useRef<number | undefined>(undefined)
  const previewRef = useRef<HTMLDivElement>(null)
  const playbackRequest = useRef(false)
  const scrubRef = useRef<number | null>(null)
  const frameHolding = useRef(false)
  const reveal = () => {
    setVisible(true)
    window.clearTimeout(hideTimer.current)
    if (!frameHolding.current) hideTimer.current = window.setTimeout(() => setVisible(false), 3000)
  }

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const next = await getLocalPTInferenceStatus(sessionId)
        if (cancelled) return
        setSession((current) => retainNewestPlaybackStatus(current, next))
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
  const imageMode = session?.inputMode ? session.inputMode === 'images' : values.parameters.input_mode === '图片文件夹检测'
  const duration = session?.durationSeconds || 0
  const playbackBusy = busy || Boolean(session?.frameStepPending)
  const seekable = Boolean(!imageMode && active && !playbackBusy && duration > 0 && (session?.status === 'running' || session?.status === 'paused'))
  const pausable = Boolean(!imageMode && active && !session?.atEnd && session?.frameCount && (session?.status === 'running' || session?.status === 'paused'))
  const steppable = Boolean(!imageMode && active && session?.frameStepSupported && (session?.status === 'running' || session?.status === 'paused'))

  const changeLabels = (text: string) => {
    readRequest.current += 1
    onParameterChange('labels', text)
    onParameterChange('labels_model_file', modelFile)
    setLabelNotice(null)
  }
  const readLabels = async () => {
    const request = ++readRequest.current
    const originalLabels = labels
    setReadingLabels(true); setError(''); setLabelNotice(null)
    try {
      const result = await detectLocalModelLabels(modelFile)
      if (request !== readRequest.current || latestLabels.current.modelFile !== modelFile || latestLabels.current.labels !== originalLabels) return
      if (result.found) {
        onParameterChange('labels', result.labels.join('\n'))
        onParameterChange('labels_model_file', modelFile)
      }
      setLabelNotice({ model: modelFile, message: result.message })
    } catch (cause) {
      if (request === readRequest.current && latestLabels.current.modelFile === modelFile) setError(cause instanceof Error ? cause.message : '读取类别失败')
    } finally { setReadingLabels(false) }
  }

  const start = async () => {
    setBusy(true); setError(''); setSession(null)
    sessionStorage.removeItem(key)
    try {
      const next = await startLocalPTInference(values)
      sessionStorage.setItem(key, next.id)
      setSessionId(next.id)
      setSession(next)
      previewRef.current?.focus({ preventScroll: true })
    } catch (cause) { setError(cause instanceof Error ? cause.message : '启动失败') }
    finally { setBusy(false) }
  }
  const stop = async () => {
    if (!sessionId || busy || session?.status === 'stopping') return
    if (!confirmStop(imageMode ? '本地图片数据集检测' : session?.atEnd ? '本地推理回看并清理' : '本地视频检测', '将结束当前检测或回看；已保存的检测结果会保留。')) return
    setBusy(true); setError('')
    try { setSession(await stopLocalPTInference(sessionId)) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '停止失败') }
    finally { setBusy(false) }
  }
  const toggle = async () => {
    if (!sessionId || !session || !pausable || playbackBusy || playbackRequest.current) return
    playbackRequest.current = true
    reveal()
    setBusy(true); setError('')
    try { setSession(await setLocalPTInferencePaused(sessionId, session.status !== 'paused')) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '播放控制失败') }
    finally { playbackRequest.current = false; setBusy(false) }
  }
  const stepFrame = async (direction: -1 | 1) => {
    if (!sessionId || !session || !steppable || playbackBusy || playbackRequest.current || scrubRef.current !== null) return false
    if (direction < 0 ? session.sourceFrameIndex <= 0 : session.sourceFrameIndex >= session.sourceFrameCount - 1) return false
    playbackRequest.current = true
    reveal()
    setBusy(true); setError('')
    try { setSession(await stepLocalPTInference(sessionId, direction)); return true }
    catch (cause) { setError(cause instanceof Error ? cause.message : '逐帧定位失败'); return false }
    finally { playbackRequest.current = false; setBusy(false) }
  }
  useVideoPlaybackShortcut(previewRef, pausable, playbackBusy, toggle, steppable, stepFrame)
  const commitSeek = async () => {
    if (playbackBusy || playbackRequest.current) { scrubRef.current = null; setScrubbing(null); return }
    const target = scrubRef.current
    if (target === null || !sessionId) return
    scrubRef.current = null
    setScrubbing(null)
    setError('')
    try { await seekLocalPTInference(sessionId, target) }
    catch (cause) { setError(cause instanceof Error ? cause.message : '跳转失败') }
  }
  const status = imageMode ? session?.message || '尚未开始图片检测' : session?.atEnd ? '已到结尾 · 可回看' : session?.status === 'paused' ? '视频已暂停' : session?.status === 'running' ? '本地检测中' : session?.message || '尚未开始'

  return <section className="remote-inference-section">
    <div className="local-model-labels">
      <div className="local-model-labels-heading"><span className="help-heading"><label htmlFor={`local-labels-${itemId}`}>类别名称（自动读取，也可手动修改）</label><HelpTip label="本地模型类别说明">支持 PT 权重和标准 YOLO ONNX。ONNX 未内嵌 names 时需手动填写；切换模型后不会沿用上一个模型的类别。ONNX 的 auto 默认使用 CPU。</HelpTip></span><button className="secondary-button" disabled={readingLabels || active || busy || !modelFile} onClick={() => void readLabels()}>{readingLabels ? '正在读取类别…' : '读取模型类别'}</button></div>
      <textarea id={`local-labels-${itemId}`} rows={4} value={labels} disabled={active || busy} onChange={(event) => changeLabels(event.target.value)} placeholder={'留空：开始检测时自动从 PT / ONNX 读取\n手动填写：每行一个类别，按 ID 0、1、2…排列'} />
      {labelNotice?.model === modelFile ? <p role="status">{labelNotice.message}</p> : null}
    </div>
    <div className="remote-run-actions">
      <button className="run-button" disabled={busy || readingLabels || (!active && Boolean(platformTask))} onClick={() => void (active ? stop() : start())}>
        {active ? <icons.X size={18} /> : <icons.Play size={18} fill="currentColor" />}{active ? session?.atEnd ? '结束回看并清理' : '停止检测' : busy ? '正在启动…' : '开始本地检测'}
      </button>
      <span className={`remote-status ${session?.status || 'idle'}`}><i />{status}</span>
      <HelpTip label="本地检测与播放说明">{imageMode ? '每张图片均检测并保存；预览只展示最近处理的图片，不影响完整检测。停止后保留已保存结果。' : '本机 PT / ONNX 检测。空格播放 / 暂停；, 上一帧，. 下一帧（兼容 ，。）。'}</HelpTip>
    </div>
    {error || session?.error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error || session?.error}</span></div> : null}
    {imageMode && session ? <div className="local-image-progress" role="region" aria-label="图片检测进度">
      <div><strong>图片检测进度</strong><span>{session.processedImages || 0} / {session.totalImages || 0} 张 · {session.progressPercent || 0}%</span></div>
      <progress aria-label="已处理图片进度" value={session.processedImages || 0} max={Math.max(session.totalImages || 0, 1)} />
      <div><span>已保存：{session.savedImages || 0} 张</span><span>读取失败：{session.failedImages || 0} 张</span></div>
      {session.currentImage ? <p>最近处理：{session.currentImage}</p> : null}
      {session.outputFolder ? <p>保存目录：<code>{session.outputFolder}</code></p> : null}
    </div> : null}
    <div ref={previewRef} tabIndex={0} role="region" aria-label={imageMode ? '本地图片检测预览' : '本地视频预览'} aria-keyshortcuts={imageMode ? undefined : 'Space , .'} className={`remote-preview ${expanded ? 'expanded' : ''}`} onMouseMove={reveal} onPointerLeave={(event) => { if (event.pointerType === 'mouse') setVisible(false) }} onPointerDown={reveal} onDoubleClick={() => setExpanded((current) => !current)} title={imageMode ? '显示最近检测的图片；双击切换撑满窗口' : '空格播放/暂停；, 上一帧，. 下一帧；逐帧按钮支持长按；双击切换撑满窗口'}>
      <button className="preview-expand-button" onClick={() => setExpanded((current) => !current)} onDoubleClick={(event) => event.stopPropagation()} aria-label={expanded ? '退出撑满窗口' : '撑满整个窗口'}>{expanded ? '退出全窗口' : '撑满窗口'}</button>
      {session?.frameCount ? <img src={session.streamUrl} alt="本地 PT / ONNX 检测预览画面" draggable={false} /> : <div className="remote-preview-empty"><icons.Play size={34} /><strong>{session?.message || (imageMode ? '选择 PT / ONNX 模型、图片文件夹与输出目录，然后开始检测' : '选择 PT / ONNX 模型和视频，然后开始本地检测')}</strong></div>}
      {!imageMode && session ? <div className={`remote-preview-timeline ${visible ? '' : 'is-hidden'}`} onDoubleClick={(event) => event.stopPropagation()} onMouseMove={reveal}>
        <button className="remote-preview-playback-button" disabled={!pausable || playbackBusy} onClick={() => void toggle()} onPointerUp={(event) => event.currentTarget.blur()} aria-keyshortcuts="Space" title="空格播放/暂停" aria-label={session.status === 'paused' ? '继续视频' : '暂停视频'}>{session.status === 'paused' ? <icons.Play size={16} fill="currentColor" /> : <icons.Pause size={16} fill="currentColor" />}</button>
        <FrameStepControls key={session.id} supported={steppable} busy={playbackBusy} index={session.sourceFrameIndex} count={session.sourceFrameCount} onStep={stepFrame} onHoldingChange={(held) => { frameHolding.current = held; reveal() }} />
        <span>{formatTime(scrubbing ?? session.positionSeconds)}</span>
        <input type="range" aria-label="视频进度，拖动后跳转" min={0} max={Math.max(duration, 0.01)} step="0.1" value={Math.min(scrubbing ?? session.positionSeconds, Math.max(duration, 0.01))} disabled={!seekable} onFocus={reveal} onChange={(event) => { const next = Number(event.target.value); scrubRef.current = next; setScrubbing(next) }} onPointerUp={() => void commitSeek()} onKeyUp={() => void commitSeek()} onBlur={() => void commitSeek()} />
        <span>{formatTime(duration)}</span>
      </div> : null}
    </div>
    {!imageMode ? <div className="remote-session-summary"><span>本地预览：{session?.frameCount || 0} 帧</span>{session && session.sourceFps > 0 ? <span>源视频：{session.sourceFps.toFixed(2)} FPS</span> : null}{duration > 0 ? <span>视频进度：{formatTime(session?.positionSeconds || 0)} / {formatTime(duration)}</span> : null}{session?.atEnd ? <span>剩余回看时间：{session.replayRemainingSeconds} 秒</span> : null}</div> : null}
    <details className="run-output" open><summary><icons.ChevronRight size={18} />本地运行日志</summary><pre>{session?.logs.length ? session.logs.join('\n') : '启动后将在这里显示模型加载和处理结果'}</pre></details>
  </section>
}
