import { useEffect, useRef, useState } from 'react'
import {
  choosePaths,
  createYoloTrainingProfile,
  deleteYoloTrainingProfile,
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
  updateYoloTrainingProfile,
} from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { WorkingValues, YoloConnectionTestResult, YoloTrainingProfile, YoloTrainingRecommendation, YoloTrainingSession, YoloTrainingSummary, YoloTrainingValues } from '../types'

type Props = {
  values: WorkingValues
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  recommendation: YoloTrainingRecommendation | null
}

type TrainingField = {
  id: string
  label: string
  type?: 'text' | 'number' | 'boolean' | 'select'
  options?: string[]
  step?: string
  browse?: 'file' | 'directory'
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: YoloConnectionTestResult
  error?: string
}

const fieldGroups: Array<{ title: string; description: string; fields: TrainingField[] }> = [
  {
    title: '基础训练配置',
    description: '数据、模型、训练轮次、批量大小、GPU 和输出目录。',
    fields: [
      { id: 'yolo_executable', label: 'YOLO 命令或可执行文件路径' },
      { id: 'task', label: '训练任务类型', type: 'select', options: ['detect', 'segment', 'classify', 'pose', 'obb'] },
      { id: 'data', label: 'data.yaml 路径', browse: 'file' },
      { id: 'model', label: '预训练模型或模型配置' },
      { id: 'epochs', label: '训练轮次 epochs', type: 'number' },
      { id: 'patience', label: '早停等待 patience', type: 'number' },
      { id: 'imgsz', label: '输入尺寸 imgsz', type: 'number' },
      { id: 'batch', label: '批量大小 batch', type: 'number', step: 'any' },
      { id: 'nbs', label: '标称批量 nbs', type: 'number' },
      { id: 'workers', label: '数据线程 workers', type: 'number' },
      { id: 'device', label: '训练设备 device' },
      { id: 'project', label: '训练输出目录 project', browse: 'directory' },
      { id: 'run_name', label: '本次训练名称 name' },
    ],
  },
  {
    title: '数据增强',
    description: '控制拼图、混合、旋转、位移、缩放、透视和翻转。',
    fields: [
      { id: 'mosaic', label: 'mosaic', type: 'number', step: '0.01' },
      { id: 'mixup', label: 'mixup', type: 'number', step: '0.01' },
      { id: 'copy_paste', label: 'copy_paste', type: 'number', step: '0.01' },
      { id: 'degrees', label: '旋转角度 degrees', type: 'number', step: '0.01' },
      { id: 'translate', label: '平移比例 translate', type: 'number', step: '0.01' },
      { id: 'scale', label: '缩放比例 scale', type: 'number', step: '0.01' },
      { id: 'shear', label: '剪切角度 shear', type: 'number', step: '0.01' },
      { id: 'perspective', label: '透视 perspective', type: 'number', step: '0.0001' },
      { id: 'fliplr', label: '水平翻转 fliplr', type: 'number', step: '0.01' },
      { id: 'flipud', label: '垂直翻转 flipud', type: 'number', step: '0.01' },
      { id: 'multi_scale', label: '多尺度 multi_scale', type: 'number', step: '0.01' },
    ],
  },
  {
    title: '损失与优化器',
    description: '损失权重、学习率、预热、权重衰减和优化器。',
    fields: [
      { id: 'cls', label: '分类损失 cls', type: 'number', step: '0.01' },
      { id: 'box', label: '框损失 box', type: 'number', step: '0.01' },
      { id: 'dfl', label: 'DFL 损失 dfl', type: 'number', step: '0.01' },
      { id: 'cos_lr', label: '余弦学习率 cos_lr', type: 'boolean' },
      { id: 'lr0', label: '初始学习率 lr0', type: 'number', step: '0.0001' },
      { id: 'lrf', label: '最终学习率比例 lrf', type: 'number', step: '0.001' },
      { id: 'warmup_epochs', label: '预热轮次 warmup_epochs', type: 'number', step: '0.1' },
      { id: 'weight_decay', label: '权重衰减 weight_decay', type: 'number', step: '0.0001' },
      { id: 'optimizer', label: '优化器 optimizer' },
    ],
  },
]

const terminalStates = new Set<YoloTrainingSession['status']>(['completed', 'stopped', 'failed'])

export function YoloTrainingPanel({ values, onParameterChange, recommendation }: Props) {
  const [profiles, setProfiles] = useState<YoloTrainingProfile[]>([])
  const [selectedProfileId, setSelectedProfileId] = useState('')
  const [profileName, setProfileName] = useState('')
  const [profileDescription, setProfileDescription] = useState('')
  const [trainingValues, setTrainingValues] = useState<YoloTrainingValues>({})
  const [profilesLoading, setProfilesLoading] = useState(true)
  const [savingProfile, setSavingProfile] = useState(false)
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

  const applyProfile = (profile: YoloTrainingProfile) => {
    setSelectedProfileId(profile.id)
    setProfileName(profile.name)
    setProfileDescription(profile.description)
    setTrainingValues({ ...profile.values })
    setNotice(`已加载训练场景“${profile.name}”。`)
    setError('')
  }

  useEffect(() => {
    getYoloTrainingProfiles().then((payload) => {
      setProfiles(payload.profiles)
      if (payload.profiles[0]) applyProfile(payload.profiles[0])
      else setTrainingValues(payload.defaults)
    }).catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : '训练场景加载失败')
    }).finally(() => setProfilesLoading(false))
  }, [])

  useEffect(() => {
    if (profilesLoading || !recommendation || appliedRecommendation.current === recommendation.updatedAt) return
    appliedRecommendation.current = recommendation.updatedAt
    setTrainingValues((current) => ({
      ...current,
      data: recommendation.data,
      project: recommendation.project,
      ...(recommendation.runName ? { run_name: recommendation.runName } : {}),
    }))
    setNotice('已自动填入刚刚划分的数据集路径、训练输出目录和本次训练名称。')
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

  const selectProfile = (profileId: string) => {
    const profile = profiles.find((item) => item.id === profileId)
    if (profile) applyProfile(profile)
  }

  const createProfile = async () => {
    setSavingProfile(true)
    setError('')
    try {
      const created = await createYoloTrainingProfile({ name: profileName, description: profileDescription, values: trainingValues })
      setProfiles((current) => [...current, created])
      setSelectedProfileId(created.id)
      setNotice(`训练场景“${created.name}”已保存。`)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : '训练场景保存失败')
    } finally {
      setSavingProfile(false)
    }
  }

  const updateProfile = async () => {
    if (!selectedProfileId) {
      setError('请先选择场景，或点击“另存为新场景”。')
      return
    }
    setSavingProfile(true)
    setError('')
    try {
      const updated = await updateYoloTrainingProfile(selectedProfileId, { name: profileName, description: profileDescription, values: trainingValues })
      setProfiles((current) => current.map((item) => item.id === updated.id ? updated : item))
      setNotice(`训练场景“${updated.name}”已更新。`)
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : '训练场景更新失败')
    } finally {
      setSavingProfile(false)
    }
  }

  const deleteProfile = async () => {
    if (!selectedProfileId) return
    if (!window.confirm(`确定删除训练场景“${profileName}”吗？`)) return
    setSavingProfile(true)
    setError('')
    try {
      await deleteYoloTrainingProfile(selectedProfileId)
      const remaining = profiles.filter((item) => item.id !== selectedProfileId)
      setProfiles(remaining)
      if (remaining[0]) applyProfile(remaining[0])
      else {
        setSelectedProfileId('')
        setProfileName('')
        setProfileDescription('')
      }
      setNotice('训练场景已删除。')
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : '训练场景删除失败')
    } finally {
      setSavingProfile(false)
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
    if (remoteRequested && (!connectionReady || !currentConnectionCheck?.result)) {
      setError('请先测试 SSH 连接，连接成功后再开始远程模型训练。')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const parameters: YoloTrainingValues = {
        ...trainingValues,
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
        output: next.output, host: next.host, status: next.status, message: next.message,
        startedAt: next.startedAt, finishedAt: next.finishedAt, error: next.error, result: next.result,
      }
      setSessions((current) => [summary, ...current.filter((item) => item.id !== next.id)])
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
    if (!sessionId) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await reconnectYoloTraining(sessionId, {
        remote_host: host, remote_port: port, remote_username: username,
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
    const input = <input type={field.type === 'number' ? 'number' : 'text'} step={field.step} value={String(value)} onChange={(event) => changeTrainingValue(field.id, field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
    if (!field.browse) return input
    return <span className="training-path-input">{input}<button onClick={() => browseTrainingPath(field)} disabled={remoteRequested}>{remoteRequested ? '远程路径' : '浏览'}</button></span>
  }

  const displayedPassword = password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password
  const connectionError = currentConnectionCheck?.state === 'failed' ? currentConnectionCheck.error : ''
  const statusLabel = session ? {
    starting: '正在准备训练', running: '训练运行中', disconnected: '等待重新连接', stopping: '正在终止', completed: '训练完成', stopped: '已停止', failed: '训练失败',
  }[session.status] : remoteRequested ? (connectionReady ? 'SSH 连接成功' : '等待连接测试') : '本地训练'

  return (
    <div className="yolo-training-workspace">
      <section className="workspace-section training-profile-section">
        <div className="section-title"><icons.Settings size={21} /><h2>训练场景</h2></div>
        <p className="section-description">保存一整套训练参数，并用容易理解的场景名称复用，例如“光线变化”“小目标加强”或“快速验证”。SSH 密码不会保存在场景中。</p>
        <div className="training-profile-grid">
          <label><span>已保存场景</span><select value={selectedProfileId} onChange={(event) => selectProfile(event.target.value)} disabled={profilesLoading}><option value="">请选择训练场景</option>{profiles.map((profile) => <option value={profile.id} key={profile.id}>{profile.name}</option>)}</select></label>
          <label><span>场景名称</span><input value={profileName} onChange={(event) => setProfileName(event.target.value)} placeholder="例如：光线变化" /></label>
          <label className="profile-description"><span>适用场景说明</span><textarea value={profileDescription} onChange={(event) => setProfileDescription(event.target.value)} placeholder="说明这组参数适合什么数据、目标或训练阶段" /></label>
        </div>
        <div className="profile-actions">
          <button onClick={createProfile} disabled={savingProfile || profilesLoading}><icons.Plus size={16} />另存为新场景</button>
          <button onClick={updateProfile} disabled={savingProfile || !selectedProfileId}>更新当前场景</button>
          <button className="danger-action" onClick={deleteProfile} disabled={savingProfile || !selectedProfileId}>删除场景</button>
        </div>
        {notice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      </section>

      {fieldGroups.map((group, index) => <details className="training-parameter-group" open={index === 0} key={group.title}>
        <summary><span><strong>{group.title}</strong><small>{group.description}</small></span><icons.ChevronRight size={18} /></summary>
        <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{group.fields.map((field) => <label className="parameter-row" key={field.id}><span>{field.label}</span>{renderTrainingField(field)}</label>)}</div>
      </details>)}

      <section className="workspace-section training-connection-section">
        <div className="section-title"><icons.Code2 size={21} /><h2>训练位置与 SSH</h2></div>
        <p className="section-description">服务器信息全部留空时在本机执行；填写 IP 和用户名后在远程服务器执行。远程模式需要先测试连接。</p>
        <div className="training-connection-grid">
          <label><span>服务器 IP / 主机名</span><input value={host} onChange={(event) => onParameterChange('remote_host', event.target.value)} placeholder="留空则本地训练" /></label>
          <label><span>SSH 端口</span><input type="number" value={port} onChange={(event) => onParameterChange('remote_port', Number(event.target.value))} /></label>
          <label><span>SSH 用户名</span><input value={username} onChange={(event) => onParameterChange('remote_username', event.target.value)} /></label>
          <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={displayedPassword} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('remote_password', '') }} onChange={(event) => onParameterChange('remote_password', event.target.value)} placeholder="首次连接或未保存时输入" /></label>
          <label className="training-remember-password"><span>记住 SSH 密码</span><span className="switch-control"><input type="checkbox" checked={rememberPassword} onChange={(event) => onParameterChange('remember_password', event.target.checked)} /><span /><em>{rememberPassword ? '开启' : '关闭'}</em></span></label>
        </div>
        <div className="remote-run-actions training-connection-actions">
          <button className="connection-test-button" onClick={testConnection} disabled={!remoteRequested || testingConnection || submitting}><icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试 SSH 连接'}</button>
          <span className={`remote-status ${connectionReady ? 'connected' : testingConnection ? 'connecting' : 'idle'}`}><i />{remoteRequested ? (connectionReady ? '连接成功' : '尚未测试连接') : '本地训练'}</span>
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
          <button className="run-button" onClick={start} disabled={submitting || testingConnection || profilesLoading || (remoteRequested && !connectionReady)}>
            <icons.Play size={18} fill="currentColor" />
            {submitting ? '请稍候…' : remoteRequested ? '再开一个远程训练' : '再开一个本地训练'}
          </button>
          {session?.status === 'disconnected' ? <button onClick={reconnect} disabled={submitting || !host || !username}>重新连接远端训练</button> : null}
          {activeSession && session?.status !== 'disconnected' ? <button className="stop-run-button" onClick={stop} disabled={submitting || session?.status === 'stopping'}><icons.X size={17} />终止选中任务</button> : null}
          {session && terminalStates.has(session.status) ? <button className="stop-run-button" onClick={deleteSelectedSession} disabled={submitting}><icons.X size={17} />删除选中任务记录</button> : null}
          <span className={`remote-status ${session?.status || (connectionReady ? 'connected' : 'idle')}`}><i />{statusLabel}</span>
          <small>远程训练在独立 tmux 会话中运行；关闭平台不会终止，重开后可重新连接查看日志。本地训练仍随平台关闭而停止。</small>
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
