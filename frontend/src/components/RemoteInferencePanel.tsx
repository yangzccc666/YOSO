import { useEffect, useState } from 'react'
import { forgetRemotePassword, getRemoteInferenceStatus, startRemoteInference, stopRemoteInference, testRemoteConnection } from '../api'
import { icons } from '../icons'
import type { ConnectionTestResult, RemoteInferenceStatus, WorkingValues } from '../types'

type Props = {
  itemId: string
  values: WorkingValues
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
}

const terminalStates = new Set<RemoteInferenceStatus['status']>(['completed', 'stopped', 'failed'])
const statusLabels: Record<RemoteInferenceStatus['status'], string> = {
  connecting: '正在连接',
  uploading: '正在上传文件',
  starting: '正在加载模型',
  running: '实时推理中',
  stopping: '正在停止',
  completed: '处理完成',
  stopped: '已停止',
  failed: '运行失败',
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: ConnectionTestResult
  error?: string
}

export function RemoteInferencePanel({ itemId, values, onParameterChange }: Props) {
  const storageKey = `processing-view:remote-session:${itemId}`
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem(storageKey) || '')
  const [session, setSession] = useState<RemoteInferenceStatus | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const [credentialNotice, setCredentialNotice] = useState('')
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [previewExpanded, setPreviewExpanded] = useState(false)

  const rememberPassword = Boolean(values.parameters.remember_password)
  const connectionSignature = JSON.stringify([
    values.parameters.host || '',
    values.parameters.port || 22,
    values.parameters.username || '',
    values.parameters.password || '',
    rememberPassword,
  ])
  const currentConnectionCheck = connectionCheck?.signature === connectionSignature ? connectionCheck : null
  const connectionReady = currentConnectionCheck?.state === 'success'
  const testingConnection = currentConnectionCheck?.state === 'checking'

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let timer: number | undefined

    const refresh = async () => {
      try {
        const next = await getRemoteInferenceStatus(sessionId)
        if (cancelled) return
        setSession(next)
        setError('')
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

  const testConnection = async () => {
    const signature = connectionSignature
    if (session && terminalStates.has(session.status)) {
      sessionStorage.removeItem(storageKey)
      setSessionId('')
      setSession(null)
    }
    setConnectionCheck({ signature, state: 'checking' })
    setError('')
    setCredentialNotice('')
    try {
      const result = await testRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        const rememberedSignature = JSON.stringify([
          values.parameters.host || '',
          values.parameters.port || 22,
          values.parameters.username || '',
          '',
          true,
        ])
        setConnectionCheck({ signature: rememberedSignature, state: 'success', result })
        onParameterChange('password', '')
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
    try {
      const next = await startRemoteInference(values)
      sessionStorage.setItem(storageKey, next.id)
      setSession(next)
      setSessionId(next.id)
      onParameterChange('password', '')
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

  const active = Boolean(session && !terminalStates.has(session.status))
  const streamUrl = session && session.frameCount > 0 ? `${session.streamUrl}?session=${session.id}` : ''
  const connectionError = currentConnectionCheck?.state === 'failed' ? currentConnectionCheck.error || 'SSH 连接测试失败' : ''
  const displayedStatus = active
    ? { className: session?.status || 'idle', label: session ? statusLabels[session.status] : '正在运行' }
    : connectionReady
      ? { className: 'connected', label: '连接成功' }
      : testingConnection
        ? { className: 'connecting', label: '正在测试连接' }
        : session && terminalStates.has(session.status)
          ? { className: session.status, label: statusLabels[session.status] }
          : { className: 'idle', label: '尚未测试连接' }

  return (
    <section className="remote-inference-section">
      <div className="remote-run-actions">
        {!active ? <button className="connection-test-button" onClick={testConnection} disabled={testingConnection || submitting}>
          <icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试连接'}
        </button> : null}
        <button className="run-button" onClick={active ? stop : start} disabled={submitting || testingConnection || (!active && !connectionReady)}>
          {active ? <icons.X size={18} /> : <icons.Play size={18} fill="currentColor" />}
          {submitting ? '请稍候…' : active ? '停止实时检测' : '开始实时检测'}
        </button>
        <span className={`remote-status ${displayedStatus.className}`}><i />{displayedStatus.label}</span>
        <small>{rememberPassword ? '已开启密码记忆；首次连接成功后无需再次输入。' : '先测试 SSH 连接；连接参数改变后需要重新测试。'}</small>
        {rememberPassword ? <button className="forget-password-button" onClick={forgetPassword} disabled={submitting || testingConnection}>清除已保存密码</button> : null}
      </div>

      {connectionReady && currentConnectionCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>{currentConnectionCheck.result.message}<small>设备指纹：{currentConnectionCheck.result.fingerprint}</small></span></div> : null}
      {credentialNotice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{credentialNotice}</span></div> : null}
      {connectionError || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{connectionError || error}</span></div> : null}

      <div className={`remote-preview ${previewExpanded ? 'expanded' : ''}`} onDoubleClick={() => setPreviewExpanded((current) => !current)} title="双击切换撑满窗口">
        <button className="preview-expand-button" onClick={() => setPreviewExpanded((current) => !current)} onDoubleClick={(event) => event.stopPropagation()} aria-label={previewExpanded ? '退出撑满窗口' : '撑满整个窗口'} aria-pressed={previewExpanded}>
          {previewExpanded ? <icons.Minimize2 size={17} /> : <icons.Maximize2 size={17} />}
          {previewExpanded ? '退出全窗口' : '撑满窗口'}
        </button>
        {streamUrl ? <img src={streamUrl} alt="远端 AI 检测实时画面" /> : <div className="remote-preview-empty"><icons.Play size={34} /><strong>{session?.message || '先测试连接，再点击“开始实时检测”'}</strong><span>开始后会自动上传内置推理脚本、所选模型和视频，带框画面会显示在这里</span></div>}
      </div>

      <div className="remote-session-summary">
        <span>设备：{session ? `${session.username}@${session.host}` : String(values.parameters.host || '未填写')}</span>
        {session?.status === 'uploading' ? <span>上传：{session.transferStage} {session.uploadProgress}%</span> : null}
        <span>已接收预览：{session?.frameCount || 0} 帧</span>
        {session?.error ? <span className="summary-error">原因：{session.error}</span> : null}
      </div>

      <details className="run-output" open>
        <summary><icons.ChevronRight size={18} />远端运行日志</summary>
        <pre>{session?.logs.length ? session.logs.join('\n') : '连接后将在这里显示设备环境、模型加载和处理进度'}</pre>
      </details>
    </section>
  )
}
