import { useEffect, useRef, useState } from 'react'
import { choosePaths, forgetRemotePassword, getRemoteCredentialStatus, getRemoteInferenceStatus, seekRemoteInference, setRemoteInferencePaused, startRemoteInference, stopRemoteInference, testRemoteConnection } from '../api'
import { icons } from '../icons'
import { parseStarModels } from '../remoteStarModels'
import type { StarModel } from '../remoteStarModels'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { ConnectionTestResult, PlatformTaskStatus, RemoteInferenceStatus, WorkingValues } from '../types'

type Props = {
  itemId: string
  values: WorkingValues
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  platformTask: PlatformTaskStatus | null
  onCompleted?: () => void
}

const terminalStates = new Set<RemoteInferenceStatus['status']>(['completed', 'stopped', 'failed'])
const TIMELINE_HIDE_DELAY_MS = 3000
const statusLabels: Record<RemoteInferenceStatus['status'], string> = {
  connecting: '正在连接',
  uploading: '正在上传文件',
  starting: '正在加载模型',
  running: '实时推理中',
  paused: '视频已暂停',
  stopping: '正在停止',
  completed: '处理完成',
  stopped: '已停止',
  failed: '运行失败',
}

function formatVideoTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '00:00'
  const total = Math.floor(seconds)
  const minutes = Math.floor(total / 60)
  const formatted = `${String(minutes % 60).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`
  return minutes >= 60 ? `${String(Math.floor(minutes / 60)).padStart(2, '0')}:${formatted}` : formatted
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: ConnectionTestResult
  error?: string
}

export function RemoteInferencePanel({ itemId, values, onParameterChange, platformTask, onCompleted }: Props) {
  const storageKey = `processing-view:remote-session:${itemId}`
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem(storageKey) || '')
  const [session, setSession] = useState<RemoteInferenceStatus | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [playbackSubmitting, setPlaybackSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [credentialNotice, setCredentialNotice] = useState('')
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [previewExpanded, setPreviewExpanded] = useState(false)
  const [scrubPosition, setScrubPosition] = useState<number | null>(null)
  const [pendingSeek, setPendingSeek] = useState<number | null>(null)
  const [timelineVisible, setTimelineVisible] = useState(true)
  const scrubPositionRef = useRef<number | null>(null)
  const seekTimer = useRef<number | undefined>(undefined)
  const timelineHideTimer = useRef<number | undefined>(undefined)
  const timelineHolding = useRef(false)
  const checkedCredentialIdentity = useRef('')
  const completedSession = useRef('')

  const clearTimelineTimer = () => {
    if (timelineHideTimer.current !== undefined) window.clearTimeout(timelineHideTimer.current)
    timelineHideTimer.current = undefined
  }

  const revealTimeline = () => {
    setTimelineVisible(true)
    clearTimelineTimer()
    if (!timelineHolding.current) timelineHideTimer.current = window.setTimeout(() => setTimelineVisible(false), TIMELINE_HIDE_DELAY_MS)
  }

  const leaveTimeline = () => {
    clearTimelineTimer()
    if (!timelineHolding.current) setTimelineVisible(false)
  }

  const releaseTimeline = () => {
    timelineHolding.current = false
    revealTimeline()
  }

  const rememberPassword = Boolean(values.parameters.remember_password)
  const host = String(values.parameters.host || '')
  const port = Number(values.parameters.port || 22)
  const username = String(values.parameters.username || '')
  const password = String(values.parameters.password || '')
  const identitySignature = JSON.stringify([host, port, username])
  const passwordSignature = password === REMEMBERED_PASSWORD_VALUE ? '' : password
  const connectionSignature = JSON.stringify([
    host,
    port,
    username,
    passwordSignature,
    rememberPassword,
  ])
  const currentConnectionCheck = connectionCheck?.signature === connectionSignature ? connectionCheck : null
  const connectionReady = currentConnectionCheck?.state === 'success'
  const testingConnection = currentConnectionCheck?.state === 'checking'
  const starModels = parseStarModels(values.parameters.star_models, String(values.paths.local_model_file || ''), String(values.parameters.labels || ''), Number(values.parameters.conf ?? 0.5))

  const saveStarModels = (models: StarModel[]) => onParameterChange('star_models', JSON.stringify(models))
  const updateStarModel = (index: number, patch: Partial<StarModel>) => {
    saveStarModels(starModels.map((model, position) => position === index ? { ...model, ...patch } : model))
  }
  const removeStarModel = (index: number) => {
    if (starModels.length <= 1) {
      setError('远程实时 AI 推理至少需要保留一个 STAR 模型。')
      return
    }
    saveStarModels(starModels.filter((_, position) => position !== index))
    setError('')
  }
  const clearAllStarModels = () => {
    const configuredConfidence = Number(values.parameters.conf ?? 0.5)
    const confidence = Number.isFinite(configuredConfidence) ? configuredConfidence : 0.5
    saveStarModels([{ path: '', labels: '', conf: confidence }])
    setError('')
  }
  const browseStarModel = async (index: number) => {
    try {
      const paths = await choosePaths('file')
      if (paths[0]) updateStarModel(index, { path: paths[0] })
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法选择模型文件')
    }
  }

  useEffect(() => {
    if (!rememberPassword) {
      checkedCredentialIdentity.current = ''
      if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('password', '')
      return
    }
    if (!host || !username) return
    if (password === REMEMBERED_PASSWORD_VALUE) {
      if (checkedCredentialIdentity.current === identitySignature) return
      checkedCredentialIdentity.current = ''
      onParameterChange('password', '')
      return
    }
    if (password || checkedCredentialIdentity.current === identitySignature) return

    let cancelled = false
    checkedCredentialIdentity.current = identitySignature
    getRemoteCredentialStatus({ host, port, username }).then((result) => {
      if (cancelled || !result.remembered) return
      onParameterChange('password', REMEMBERED_PASSWORD_VALUE)
      setCredentialNotice('已自动填充此设备上次保存的 SSH 密码。')
    }).catch(() => {
      if (!cancelled) checkedCredentialIdentity.current = ''
    })
    return () => { cancelled = true }
  }, [host, identitySignature, password, port, rememberPassword, username])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let timer: number | undefined

    const refresh = async () => {
      try {
        const next = await getRemoteInferenceStatus(sessionId)
        if (cancelled) return
        setSession(next)
        if (!terminalStates.has(next.status)) timer = window.setTimeout(refresh, 700)
      } catch (refreshError) {
        if (cancelled) return
        sessionStorage.removeItem(storageKey)
        setSessionId('')
        setSession(null)
        setError(refreshError instanceof Error ? refreshError.message : '无法读取远端任务状态')
      }
    }

    refresh()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [sessionId, storageKey])

  useEffect(() => {
    if (session?.status !== 'completed' || completedSession.current === session.id) return
    completedSession.current = session.id
    onCompleted?.()
  }, [onCompleted, session?.id, session?.status])

  useEffect(() => {
    if (!previewExpanded) return
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPreviewExpanded(false)
    }
    document.body.classList.add('remote-preview-open')
    window.addEventListener('keydown', closeOnEscape)
    return () => {
      document.body.classList.remove('remote-preview-open')
      window.removeEventListener('keydown', closeOnEscape)
    }
  }, [previewExpanded])

  useEffect(() => {
    if (pendingSeek === null || !session) return
    if (Math.abs(session.positionSeconds - pendingSeek) < 0.8) {
      setPendingSeek(null)
      if (seekTimer.current !== undefined) window.clearTimeout(seekTimer.current)
    }
  }, [pendingSeek, session?.positionSeconds])

  useEffect(() => () => {
    if (seekTimer.current !== undefined) window.clearTimeout(seekTimer.current)
    clearTimelineTimer()
  }, [])

  useEffect(() => {
    if (!session?.frameCount) return
    setTimelineVisible(true)
    timelineHideTimer.current = window.setTimeout(() => setTimelineVisible(false), TIMELINE_HIDE_DELAY_MS)
    return clearTimelineTimer
  }, [Boolean(session?.frameCount), session?.id])

  const testConnection = async () => {
    const signature = connectionSignature
    if (session && terminalStates.has(session.status)) {
      sessionStorage.removeItem(storageKey)
      setSessionId('')
      setSession(null)
      setScrubPosition(null)
      setPendingSeek(null)
      scrubPositionRef.current = null
    }
    setConnectionCheck({ signature, state: 'checking' })
    setError('')
    setCredentialNotice('')
    try {
      const result = await testRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        checkedCredentialIdentity.current = identitySignature
        const rememberedSignature = JSON.stringify([
          host,
          port,
          username,
          '',
          true,
        ])
        setConnectionCheck({ signature: rememberedSignature, state: 'success', result })
        onParameterChange('password', REMEMBERED_PASSWORD_VALUE)
      } else {
        setConnectionCheck({ signature, state: 'success', result })
      }
    } catch (connectionError) {
      setConnectionCheck({
        signature,
        state: 'failed',
        error: connectionError instanceof Error ? connectionError.message : 'SSH 连接测试失败',
      })
    }
  }

  const forgetPassword = async () => {
    setSubmitting(true)
    setError('')
    setCredentialNotice('')
    try {
      const result = await forgetRemotePassword(values.parameters)
      onParameterChange('password', '')
      onParameterChange('remember_password', false)
      checkedCredentialIdentity.current = ''
      setConnectionCheck(null)
      setCredentialNotice(result.message)
    } catch (forgetError) {
      setError(forgetError instanceof Error ? forgetError.message : '清除保存的密码失败')
    } finally {
      setSubmitting(false)
    }
  }

  const start = async () => {
    if (!connectionReady) {
      setError('请先测试 SSH 连接，连接成功后再开始实时检测。')
      return
    }
    setSubmitting(true)
    setError('')
    setScrubPosition(null)
    setPendingSeek(null)
    scrubPositionRef.current = null
    try {
      const next = await startRemoteInference(values)
      sessionStorage.setItem(storageKey, next.id)
      setSession(next)
      setSessionId(next.id)
      onParameterChange('password', rememberPassword ? REMEMBERED_PASSWORD_VALUE : '')
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : '远端推理启动失败')
    } finally {
      setSubmitting(false)
    }
  }

  const stop = async () => {
    if (!sessionId) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await stopRemoteInference(sessionId))
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : '停止远端推理失败')
    } finally {
      setSubmitting(false)
    }
  }

  const commitSeek = async () => {
    const target = scrubPositionRef.current
    if (target === null || !sessionId) return
    scrubPositionRef.current = null
    setScrubPosition(null)
    setPendingSeek(target)
    if (seekTimer.current !== undefined) window.clearTimeout(seekTimer.current)
    seekTimer.current = window.setTimeout(() => setPendingSeek(null), 3000)
    setError('')
    try {
      await seekRemoteInference(sessionId, target)
    } catch (seekError) {
      setPendingSeek(null)
      setError(seekError instanceof Error ? seekError.message : '视频跳转失败')
    }
  }

  const togglePlayback = async () => {
    if (!sessionId || !session) return
    setPlaybackSubmitting(true)
    setError('')
    try {
      setSession(await setRemoteInferencePaused(sessionId, session.status !== 'paused'))
    } catch (playbackError) {
      setError(playbackError instanceof Error ? playbackError.message : '暂停或继续视频失败')
    } finally {
      setPlaybackSubmitting(false)
    }
  }

  const active = Boolean(session && !terminalStates.has(session.status))
  const streamUrl = session && session.frameCount > 0 ? `${session.streamUrl}?session=${session.id}` : ''
  const duration = session?.durationSeconds || 0
  const displayedPosition = scrubPosition ?? pendingSeek ?? session?.positionSeconds ?? 0
  const canSeek = Boolean(session && (session.status === 'running' || session.status === 'paused') && duration > 0)
  const canPause = Boolean(session && (session.status === 'running' || session.status === 'paused') && session.frameCount > 0 && !session.atEnd)
  const connectionError = currentConnectionCheck?.state === 'failed' ? currentConnectionCheck.error || 'SSH 连接测试失败' : ''
  const displayedStatus = active
    ? { className: session?.status || 'idle', label: session?.atEnd ? '已到结尾 · 可回看' : session ? statusLabels[session.status] : '正在运行' }
    : connectionReady
      ? { className: 'connected', label: '连接成功' }
      : testingConnection
        ? { className: 'connecting', label: '正在测试连接' }
        : session && terminalStates.has(session.status)
          ? { className: session.status, label: statusLabels[session.status] }
          : { className: 'idle', label: '尚未测试连接' }

  return (
    <section className="remote-inference-section">
      <div className="remote-multi-models">
        <div className="section-title"><icons.Settings size={19} /><h2>同一视频使用多个 STAR 模型</h2></div>
        <p className="section-description">所有 STAR 模型都在这里统一配置。每个模型分别填写文件、类别名和置信度，检测框会叠加在同一视频上。多个模型会降低推理帧率；开启“保持原视频速度”后，平台会自动跳过已经落后的源帧，避免画面变成慢动作。</p>
        {starModels.map((model, index) => <div className="remote-model-card" key={index}>
          <div className="remote-model-card-title"><strong>模型 {index + 1}</strong><button type="button" onClick={() => removeStarModel(index)} title={starModels.length === 1 ? '至少需要保留一个模型' : `移除模型 ${index + 1}`}>移除</button></div>
          <label><span>模型文件</span><div className="remote-model-path"><input value={model.path} onChange={(event) => updateStarModel(index, { path: event.target.value })} placeholder="选择本机 .star / .plan / .engine 文件" /><button type="button" onClick={() => void browseStarModel(index)}>浏览</button></div></label>
          <div className="remote-model-options">
            <label><span>类别名称（英文逗号分隔）</span><input value={model.labels} onChange={(event) => updateStarModel(index, { labels: event.target.value })} placeholder="例如：person,car" /></label>
            <label><span>置信度阈值</span><input type="number" min="0" max="1" step="0.01" value={model.conf} onChange={(event) => updateStarModel(index, { conf: Number(event.target.value) })} /></label>
          </div>
        </div>)}
        <div className="remote-model-actions">
          <button type="button" className="text-action" disabled={starModels.length >= 8} onClick={() => saveStarModels([...starModels, { path: '', labels: '', conf: Number(values.parameters.conf ?? 0.5) }])}><icons.Plus size={16} />添加 STAR 模型（最多 8 个）</button>
          <button type="button" className="text-action danger" disabled={starModels.length === 1 && !starModels[0].path.trim() && !starModels[0].labels.trim()} onClick={clearAllStarModels} title="删除全部模型配置并保留一个空白上传窗口"><icons.Trash2 size={16} />清空全部模型</button>
        </div>
      </div>
      <div className="remote-run-actions">
        {!active ? <button className="connection-test-button" onClick={testConnection} disabled={testingConnection || submitting}>
          <icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试连接'}
        </button> : null}
        <button className="run-button" onClick={active ? stop : start} disabled={submitting || testingConnection || (!active && (!connectionReady || Boolean(platformTask)))}>
          {active ? <icons.X size={18} /> : <icons.Play size={18} fill="currentColor" />}
          {submitting ? '请稍候…' : active ? (session?.atEnd ? '结束回看并清理' : '停止实时检测') : platformTask ? '正在运行' : '开始实时检测'}
        </button>
        <span className={`remote-status ${displayedStatus.className}`}><i />{displayedStatus.label}</span>
        <small>{rememberPassword ? '已开启密码记忆；首次连接成功后无需再次输入。' : '先测试 SSH 连接；连接参数改变后需要重新测试。'}</small>
        {rememberPassword ? <button className="forget-password-button" onClick={forgetPassword} disabled={submitting || testingConnection}>清除已保存密码</button> : null}
      </div>

      {connectionReady && currentConnectionCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>{currentConnectionCheck.result.message}<small>设备指纹：{currentConnectionCheck.result.fingerprint}</small></span></div> : null}
      {credentialNotice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{credentialNotice}</span></div> : null}
      {connectionError || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{connectionError || error}</span></div> : null}

      <div className={`remote-preview ${previewExpanded ? 'expanded' : ''}`} onMouseMove={revealTimeline} onMouseLeave={leaveTimeline} onPointerDown={revealTimeline} onDoubleClick={() => setPreviewExpanded((current) => !current)} title="双击切换撑满窗口">
        <button className="preview-expand-button" onClick={() => setPreviewExpanded((current) => !current)} onDoubleClick={(event) => event.stopPropagation()} aria-label={previewExpanded ? '退出撑满窗口' : '撑满整个窗口'} aria-pressed={previewExpanded}>
          {previewExpanded ? <icons.Minimize2 size={17} /> : <icons.Maximize2 size={17} />}
          {previewExpanded ? '退出全窗口' : '撑满窗口'}
        </button>
        {streamUrl ? <img src={streamUrl} alt="远端 AI 检测实时画面" draggable={false} /> : <div className="remote-preview-empty"><icons.Play size={34} /><strong>{session?.message || '先测试连接，再点击“开始实时检测”'}</strong><span>开始后会自动上传内置推理脚本、所选模型和视频，带框画面会显示在这里</span></div>}
        {session ? <div className={`remote-preview-timeline ${timelineVisible ? '' : 'is-hidden'}`} onPointerDown={() => { timelineHolding.current = true; clearTimelineTimer() }} onPointerUp={releaseTimeline} onPointerCancel={releaseTimeline} onDoubleClick={(event) => event.stopPropagation()}>
          <button className="remote-preview-playback-button" type="button" onClick={() => void togglePlayback()} onPointerUp={(event) => event.currentTarget.blur()} disabled={!canPause || playbackSubmitting} aria-label={session.status === 'paused' ? '继续视频' : '暂停视频'} title={session.status === 'paused' ? '继续视频' : '暂停视频'}>
            {session.status === 'paused' ? <icons.Play size={16} fill="currentColor" /> : <icons.Pause size={16} fill="currentColor" />}
          </button>
          <span>{formatVideoTime(displayedPosition)}</span>
          <input aria-label="视频进度，拖动后跳转" type="range" min={0} max={Math.max(duration, 0.01)} step="0.1" value={Math.min(displayedPosition, Math.max(duration, 0.01))} disabled={!canSeek} onFocus={revealTimeline} onChange={(event) => { const next = Number(event.target.value); scrubPositionRef.current = next; setScrubPosition(next) }} onPointerUp={(event) => { void commitSeek(); event.currentTarget.blur() }} onKeyUp={() => void commitSeek()} onBlur={() => { releaseTimeline(); void commitSeek() }} />
          <span>{formatVideoTime(duration)}</span>
        </div> : null}
      </div>

      <div className="remote-session-summary">
        <span>设备：{session ? `${session.username}@${session.host}` : String(values.parameters.host || '未填写')}</span>
        {session?.status === 'uploading' ? <span>上传：{session.transferStage} {session.uploadProgress}%</span> : null}
        <span>已接收预览：{session?.frameCount || 0} 帧</span>
        {session && session.sourceFps > 0 ? <span>源视频：{session.sourceFps.toFixed(2)} FPS</span> : null}
        {duration > 0 ? <span>视频进度：{formatVideoTime(session?.positionSeconds || 0)} / {formatVideoTime(duration)}{!canSeek ? '（运行时可拖动）' : ''}</span> : null}
        {active && session?.atEnd ? <span>已到结尾；{session.replayRemainingSeconds} 秒内可拖动回看，或点击“结束回看并清理”。</span> : null}
        {session?.error ? <span className="summary-error">原因：{session.error}</span> : null}
      </div>

      <details className="run-output" open>
        <summary><icons.ChevronRight size={18} />远端运行日志</summary>
        <pre>{session?.logs.length ? session.logs.join('\n') : '连接后将在这里显示设备环境、模型加载和处理进度'}</pre>
      </details>
    </section>
  )
}
