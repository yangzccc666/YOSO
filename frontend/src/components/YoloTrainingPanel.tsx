import { useEffect, useRef, useState } from 'react'
import {
  choosePaths,
  deleteYoloTrainingSession,
  forgetYoloRemotePassword,
  getYoloRemoteCredentialStatus,
  getYoloTrainingProfiles,
  getYoloTrainingSessions,
  getYoloTrainingStatus,
  reconnectYoloTraining,
  startYoloTraining,
  stopYoloTraining,
  testYoloRemoteConnection,
} from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { WorkingValues, YoloConnectionTestResult, YoloTrainingRecommendation, YoloTrainingSession, YoloTrainingSummary, YoloTrainingValues } from '../types'
import { SshConnectionHistoryInput } from './SshConnectionHistoryInput'

type Props = {
  values: WorkingValues
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  recommendation: YoloTrainingRecommendation | null
  onCompleted?: () => void
}

type TrainingField = {
  id: string
  label: string
  type?: 'text' | 'number' | 'boolean' | 'select'
  options?: string[]
  step?: string
  browse?: 'file' | 'directory'
  help: string
  range?: string
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: YoloConnectionTestResult
  error?: string
}

const fieldGroups: Array<{ title: string; description: string; defaultOpen?: boolean; fields: TrainingField[] }> = [
  {
    title: '常用训练配置',
    description: '填写远端 data.yaml，并配置轮次、批量大小、尺寸、GPU 和输出位置。训练固定使用远端源码与 yolo26n.pt。',
    defaultOpen: true,
    fields: [
      { id: 'data', label: 'data.yaml 路径', browse: 'file', help: '数据集配置文件，定义 train/val/test 路径、类别数量和类别名称。' },
      { id: 'model', label: '预训练模型 model（固定）', help: '固定使用远端 Ultralytics 源码目录中的 yolo26n.pt，避免误选其他模型。' },
      { id: 'epochs', label: '训练轮次 epochs', type: 'number', help: '完整遍历训练集的最大次数；与 patience 共同决定实际训练时长。', range: '建议 50–500，当前工业任务默认 200' },
      { id: 'patience', label: '早停等待 patience', type: 'number', help: '验证指标连续多少轮没有改善后停止；0 表示关闭早停。', range: '≥ 0，常用 30–100' },
      { id: 'imgsz', label: '输入尺寸 imgsz', type: 'number', help: '训练输入图像尺寸。更大有利于小目标，但显存和训练时间明显增加。', range: '通常为 32 的倍数，如 640、960、1280' },
      { id: 'batch', label: '批量大小 batch', type: 'number', step: 'any', help: '每批图像数；-1 自动估算，0–1 可表示显存占用比例，正整数为固定批量。', range: '-1、0–1 或正整数' },
      { id: 'device', label: '训练设备 device', help: '本项每次训练都必须手动填写，不保留默认值。填写 GPU 编号或列表，例如 0、0,1；也可填写 cpu。' },
      { id: 'workers', label: '数据线程 workers', type: 'number', help: '每个训练进程的数据加载线程数。过高可能占满 CPU 或共享内存。', range: '≥ 0，常用 4–16' },
      { id: 'project', label: '训练输出目录 project', browse: 'directory', help: '训练结果的父目录；最终结果保存在 project/name 下。' },
      { id: 'run_name', label: '本次训练名称 name', help: '本次实验子目录名。留空时平台自动生成 yolo26_日期_时间。' },
    ],
  },
]

const terminalStates = new Set<YoloTrainingSession['status']>(['completed', 'stopped', 'failed'])


function automaticTrainingRunName(date = new Date()): string {
  const twoDigits = (value: number) => String(value).padStart(2, '0')
  return `yolo26_${twoDigits(date.getMonth() + 1)}${twoDigits(date.getDate())}_${twoDigits(date.getHours())}${twoDigits(date.getMinutes())}`
}

export function YoloTrainingPanel({ values, onParameterChange, recommendation, onCompleted }: Props) {
  const [trainingValues, setTrainingValues] = useState<YoloTrainingValues>({})
  const [profilesLoading, setProfilesLoading] = useState(true)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [credentialNotice, setCredentialNotice] = useState('')
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem('processing-view:yolo-training-session') || '')
  const [session, setSession] = useState<YoloTrainingSession | null>(null)
  const [sessions, setSessions] = useState<YoloTrainingSummary[]>([])
  const [submitting, setSubmitting] = useState(false)
  const checkedCredentialIdentity = useRef('')
  const appliedRecommendation = useRef('')
  const completedSession = useRef('')

  const host = String(values.parameters.remote_host || '').trim()
  const port = Number(values.parameters.remote_port || 22)
  const username = String(values.parameters.remote_username || '').trim()
  const password = String(values.parameters.remote_password || '')
  const rememberPassword = values.parameters.remember_password !== false
  const remoteRequested = Boolean(host || username || password)
  const identitySignature = JSON.stringify([host, port, username])
  const passwordSignature = password === REMEMBERED_PASSWORD_VALUE ? '' : password
  const connectionSignature = JSON.stringify([host, port, username, passwordSignature, rememberPassword])
  const currentConnectionCheck = connectionCheck?.signature === connectionSignature ? connectionCheck : null
  const connectionReady = currentConnectionCheck?.state === 'success'
  const testingConnection = currentConnectionCheck?.state === 'checking'
  const activeSession = Boolean(session && !terminalStates.has(session.status))

  useEffect(() => {
    let cancelled = false
    const refresh = () => getYoloTrainingSessions().then((items) => {
      if (cancelled) return
      setSessions(items)
      if (!sessionStorage.getItem('processing-view:yolo-training-session') && items[0]) {
        sessionStorage.setItem('processing-view:yolo-training-session', items[0].id)
        setSessionId(items[0].id)
      }
    }).catch(() => { /* The selected-session request reports failures. */ })
    void refresh()
    const timer = window.setInterval(refresh, 2000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    getYoloTrainingProfiles().then((payload) => {
      const initialValues = payload.profiles[0] ? payload.profiles[0].values : payload.defaults
      setTrainingValues({ ...initialValues, device: '' })
    }).catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : '训练参数加载失败')
    }).finally(() => setProfilesLoading(false))
  }, [])

  useEffect(() => {
    if (profilesLoading || !recommendation || appliedRecommendation.current === recommendation.updatedAt) return
    appliedRecommendation.current = recommendation.updatedAt
    setTrainingValues((current) => ({
      ...current,
      data: recommendation.data,
      project: recommendation.project,
      run_name: '',
    }))
    setNotice('已自动填入刚刚划分的数据集路径和训练输出目录；训练名称将在启动时自动生成。')
    setError('')
  }, [profilesLoading, recommendation])

  useEffect(() => {
    if (!rememberPassword) {
      checkedCredentialIdentity.current = ''
      if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('remote_password', '')
      return
    }
    if (!host || !username) return
    if (password === REMEMBERED_PASSWORD_VALUE) {
      if (checkedCredentialIdentity.current === identitySignature) return
      checkedCredentialIdentity.current = ''
      onParameterChange('remote_password', '')
      return
    }
    if (password || checkedCredentialIdentity.current === identitySignature) return
    let cancelled = false
    checkedCredentialIdentity.current = identitySignature
    getYoloRemoteCredentialStatus(values.parameters).then((result) => {
      if (cancelled || !result.remembered) return
      onParameterChange('remote_password', REMEMBERED_PASSWORD_VALUE)
      setCredentialNotice('已自动填充该服务器上次保存的 SSH 密码。')
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
        const next = await getYoloTrainingStatus(sessionId)
        if (cancelled) return
        setSession(next)
        if (!terminalStates.has(next.status)) timer = window.setTimeout(refresh, 800)
      } catch (refreshError) {
        if (cancelled) return
        sessionStorage.removeItem('processing-view:yolo-training-session')
        setSessionId('')
        setSession(null)
        setError(refreshError instanceof Error ? refreshError.message : '无法读取训练状态')
      }
    }
    refresh()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [sessionId])

  useEffect(() => {
    if (session?.status !== 'completed' || completedSession.current === session.id) return
    completedSession.current = session.id
    onCompleted?.()
  }, [onCompleted, session?.id, session?.status])

  const changeTrainingValue = (fieldId: string, value: string | number | boolean) => {
    setTrainingValues((current) => ({ ...current, [fieldId]: value }))
    setNotice('')
  }

  const browseTrainingPath = async (field: TrainingField) => {
    if (!field.browse || remoteRequested) return
    try {
      const paths = await choosePaths(field.browse)
      if (paths[0]) changeTrainingValue(field.id, paths[0])
    } catch (browseError) {
      setError(browseError instanceof Error ? browseError.message : '无法打开选择窗口')
    }
  }

  const testConnection = async () => {
    const signature = connectionSignature
    setConnectionCheck({ signature, state: 'checking' })
    setCredentialNotice('')
    setError('')
    try {
      const result = await testYoloRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        checkedCredentialIdentity.current = identitySignature
        setConnectionCheck({ signature: JSON.stringify([host, port, username, '', true]), state: 'success', result })
        onParameterChange('remote_password', REMEMBERED_PASSWORD_VALUE)
      } else {
        setConnectionCheck({ signature, state: 'success', result })
      }
    } catch (connectionError) {
      setConnectionCheck({ signature, state: 'failed', error: connectionError instanceof Error ? connectionError.message : 'SSH 连接测试失败' })
    }
  }

  const forgetPassword = async () => {
    setSubmitting(true)
    setError('')
    try {
      const result = await forgetYoloRemotePassword(values.parameters)
      onParameterChange('remote_password', '')
      onParameterChange('remember_password', false)
      checkedCredentialIdentity.current = ''
      setConnectionCheck(null)
      setCredentialNotice(result.message)
    } catch (forgetError) {
      setError(forgetError instanceof Error ? forgetError.message : '清除保存密码失败')
    } finally {
      setSubmitting(false)
    }
  }

  const start = async () => {
    if (!host || !username) {
      setError('当前训练流程固定在远程服务器运行，请填写服务器 IP 和 SSH 用户名。')
      return
    }
    if (!String(trainingValues.device || '').trim()) {
      setError('请先填写本次训练使用的设备 device，例如 0、1、0,1 或 cpu。')
      return
    }
    if (!connectionReady || !currentConnectionCheck?.result) {
      setError('请先测试 SSH 连接，连接成功后再开始远程模型训练。')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const runName = String(trainingValues.run_name || '').trim() || automaticTrainingRunName()
      const preparedTrainingValues = { ...trainingValues, run_name: runName }
      setTrainingValues(preparedTrainingValues)
      const parameters: YoloTrainingValues = {
        ...preparedTrainingValues,
        remote_host: host,
        remote_port: port,
        remote_username: username,
        remote_password: password,
        remember_password: rememberPassword,
      }
      if (remoteRequested && currentConnectionCheck?.result) {
        parameters.remote_connection_token = currentConnectionCheck.result.connectionToken
      }
      const next = await startYoloTraining(parameters)
      sessionStorage.setItem('processing-view:yolo-training-session', next.id)
      setSession(next)
      setSessionId(next.id)
      const summary: YoloTrainingSummary = {
        id: next.id, remote: next.remote, model: next.model, device: next.device,
        output: next.output, host: next.host, port: next.port, username: next.username,
        status: next.status, message: next.message,
        startedAt: next.startedAt, finishedAt: next.finishedAt, error: next.error, result: next.result,
      }
      setSessions((current) => [summary, ...current.filter((item) => item.id !== next.id)])
      setTrainingValues((current) => ({ ...current, device: '' }))
      if (remoteRequested) onParameterChange('remote_password', rememberPassword ? REMEMBERED_PASSWORD_VALUE : '')
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : '训练启动失败')
    } finally {
      setSubmitting(false)
    }
  }

  const stop = async () => {
    if (!sessionId) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await stopYoloTraining(sessionId))
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : '停止训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  const deleteSelectedSession = async () => {
    if (!session || !terminalStates.has(session.status)) return
    if (!window.confirm(`删除训练任务“${session.output.split('/').pop() || session.id}”的界面记录？服务器上的训练输出和文件不会删除。`)) return
    setSubmitting(true)
    setError('')
    try {
      await deleteYoloTrainingSession(session.id)
      const remaining = sessions.filter((item) => item.id !== session.id)
      setSessions(remaining)
      const nextId = remaining[0]?.id || ''
      if (nextId) sessionStorage.setItem('processing-view:yolo-training-session', nextId)
      else sessionStorage.removeItem('processing-view:yolo-training-session')
      setSessionId(nextId)
      setSession(null)
      setNotice('训练任务记录已删除，服务器上的训练文件未改动。')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '删除训练任务记录失败')
    } finally {
      setSubmitting(false)
    }
  }

  const reconnect = async () => {
    if (!sessionId || !session) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await reconnectYoloTraining(sessionId, {
        remote_host: session.host, remote_port: session.port, remote_username: session.username,
        remote_password: password, remember_password: rememberPassword,
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '重新连接训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  const selectSession = (id: string) => {
    sessionStorage.setItem('processing-view:yolo-training-session', id)
    setSessionId(id)
    setSession(null)
    setError('')
  }

  const renderTrainingField = (field: TrainingField) => {
    const value = trainingValues[field.id] ?? ''
    if (field.type === 'boolean') {
      return <label className="switch-control"><input type="checkbox" checked={Boolean(value)} onChange={(event) => changeTrainingValue(field.id, event.target.checked)} /><span /><em>{value ? '开启' : '关闭'}</em></label>
    }
    if (field.type === 'select') {
      return <span className="workspace-select"><select value={String(value)} onChange={(event) => changeTrainingValue(field.id, event.target.value)}>{field.options?.map((option) => <option value={option} key={option}>{option}</option>)}</select><icons.ChevronDown size={16} /></span>
    }
    const input = <input type={field.type === 'number' ? 'number' : 'text'} step={field.step} value={String(value)} disabled={field.id === 'model'} placeholder={field.id === 'device' ? '必填，例如 0、1、0,1 或 cpu' : undefined} onChange={(event) => changeTrainingValue(field.id, field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
    if (!field.browse) return input
    return <span className="training-path-input">{input}<button onClick={() => browseTrainingPath(field)} disabled={remoteRequested}>{remoteRequested ? '远程路径' : '浏览'}</button></span>
  }

  const displayedPassword = password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password
  const connectionError = currentConnectionCheck?.state === 'failed' ? currentConnectionCheck.error : ''
  const statusLabel = session ? {
    starting: '正在准备训练', running: '训练运行中', disconnected: '等待重新连接', stopping: '正在终止', completed: '训练完成', stopped: '已停止', failed: '训练失败',
  }[session.status] : connectionReady ? 'SSH 连接成功' : '等待连接测试'

  return (
    <div className="yolo-training-workspace">
      {notice ? <div className="credential-notice training-parameter-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      {fieldGroups.map((group) => <details className="training-parameter-group" open={group.defaultOpen} key={group.title}>
        <summary><span><strong>{group.title}</strong><small>{group.description}</small></span><icons.ChevronRight size={18} /></summary>
        <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{group.fields.map((field) => <div className="parameter-row" key={field.id}><span className="parameter-label"><span>{field.label}</span><button type="button" className="parameter-help" aria-label={`查看 ${field.label} 说明`} data-tooltip={`${field.help}${field.range ? `\n范围：${field.range}` : ''}`}><icons.CircleHelp size={15} /></button></span>{renderTrainingField(field)}</div>)}</div>
      </details>)}

      <section className="workspace-section training-connection-section">
        <div className="section-title"><icons.Code2 size={21} /><h2>训练位置与 SSH</h2></div>
        <p className="section-description">本流程固定在远程服务器执行，使用 <code>/home/dell/lqy/lqy/code/ultralytics</code> 下的源码和 <code>yolo26n.pt</code>；开始前需要测试 SSH 连接。</p>
        <div className="training-connection-grid">
          <label><span>服务器 IP / 主机名</span><SshConnectionHistoryInput host={host} disabled={testingConnection || submitting} placeholder="训练服务器，例如 192.168.21.5" onHostChange={(value) => onParameterChange('remote_host', value)} onSelect={(profile) => {
            onParameterChange('remote_port', profile.port); onParameterChange('remote_username', profile.username); onParameterChange('remember_password', true); onParameterChange('remote_password', profile.passwordRemembered ? REMEMBERED_PASSWORD_VALUE : ''); setConnectionCheck(null); setCredentialNotice(profile.passwordRemembered ? '已选择历史服务器，并填充保存的 SSH 密码。' : '已选择历史服务器，请输入 SSH 密码。')
          }} /></label>
          <label><span>SSH 端口</span><input type="number" value={port} onChange={(event) => onParameterChange('remote_port', Number(event.target.value))} /></label>
          <label><span>SSH 用户名</span><input value={username} onChange={(event) => onParameterChange('remote_username', event.target.value)} /></label>
          <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={displayedPassword} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('remote_password', '') }} onChange={(event) => onParameterChange('remote_password', event.target.value)} placeholder="首次连接或未保存时输入" /></label>
          <label className="training-remember-password"><span>记住 SSH 密码</span><span className="switch-control"><input type="checkbox" checked={rememberPassword} onChange={(event) => onParameterChange('remember_password', event.target.checked)} /><span /><em>{rememberPassword ? '开启' : '关闭'}</em></span></label>
        </div>
        <div className="remote-run-actions training-connection-actions">
          <button className="connection-test-button" onClick={testConnection} disabled={!remoteRequested || testingConnection || submitting}><icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试 SSH 连接'}</button>
          <span className={`remote-status ${connectionReady ? 'connected' : testingConnection ? 'connecting' : 'idle'}`}><i />{connectionReady ? '连接成功' : '尚未测试连接'}</span>
          {remoteRequested && rememberPassword ? <button className="forget-password-button" onClick={forgetPassword} disabled={submitting || testingConnection}>清除已保存密码</button> : null}
        </div>
        {connectionReady && currentConnectionCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>{currentConnectionCheck.result.message}<small>设备指纹：{currentConnectionCheck.result.fingerprint}</small></span></div> : null}
        {credentialNotice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{credentialNotice}</span></div> : null}
      </section>

      <section className="remote-inference-section training-run-section">
        <div className="section-title"><icons.Play size={21} /><h2>训练任务</h2></div>
        <p className="section-description">可同时启动多个训练；每个任务独立运行、查看日志和终止。请为每个任务设置不同的输出名称，并按显存容量分配 GPU。</p>
        {sessions.length ? <div className="training-session-list" aria-label="训练任务列表">{sessions.map((item) => <button type="button" key={item.id} className={item.id === sessionId ? 'selected' : ''} aria-pressed={item.id === sessionId} onClick={() => selectSession(item.id)}>
          <strong>{item.output.split('/').pop() || item.id}</strong><span>{item.model.split('/').pop()} · {item.remote ? item.host : '本地'} · GPU {item.device}</span><em>{item.status === 'running' ? '训练中' : item.status === 'starting' ? '准备中' : item.status === 'disconnected' ? '待重新连接' : item.status === 'stopping' ? '终止中' : item.status === 'completed' ? '已完成' : item.status === 'failed' ? '失败' : '已停止'}</em>
        </button>)}</div> : <p className="section-empty">当前没有训练任务。</p>}
        <div className="remote-run-actions">
          <button className="run-button" onClick={start} disabled={submitting || testingConnection || profilesLoading || !connectionReady}>
            <icons.Play size={18} fill="currentColor" />
            {submitting ? '请稍候…' : '再开一个远程训练'}
          </button>
          {session?.status === 'disconnected' ? <button onClick={reconnect} disabled={submitting}>重新连接远端训练</button> : null}
          {activeSession && session?.status !== 'disconnected' ? <button className="stop-run-button" onClick={stop} disabled={submitting || session?.status === 'stopping'}><icons.X size={17} />终止选中任务</button> : null}
          {session && terminalStates.has(session.status) ? <button className="stop-run-button" onClick={deleteSelectedSession} disabled={submitting}><icons.X size={17} />删除选中任务记录</button> : null}
          <span className={`remote-status ${session?.status || (connectionReady ? 'connected' : 'idle')}`}><i />{statusLabel}</span>
          <small>{session?.status === 'disconnected'
            ? `将使用任务保存的连接信息 ${session.username}@${session.host}:${session.port}；若密码未保存，请先在上方 SSH 密码框输入。`
            : '远程训练在独立 tmux 会话中运行；关闭平台不会终止，重开后可重新连接查看日志。'}</small>
        </div>
        {connectionError || error || session?.error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{connectionError || error || session?.error}</span></div> : null}
        <details className="run-output" open>
          <summary><icons.ChevronRight size={18} />训练日志</summary>
          <pre>{session?.logs.length ? session.logs.join('\n') : '开始训练后，将在这里持续显示 Ultralytics YOLO 的训练日志'}</pre>
        </details>
      </section>
    </div>
  )
}
