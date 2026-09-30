import { useEffect, useRef, useState } from 'react'
import { choosePaths, forgetYoloRemotePassword, getYoloRemoteCredentialStatus, testYoloRemoteConnection } from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { PlatformTaskStatus, WorkingValues, YoloConnectionTestResult } from '../types'

type Props = {
  values: WorkingValues
  output: string
  running: boolean
  platformTask: PlatformTaskStatus | null
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  onRun: (parameterOverrides?: Record<string, string | number | boolean>) => void
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: YoloConnectionTestResult
  error?: string
}

export function RemoteCalibrationPanel({ values, output, running, platformTask, onParameterChange, onRun }: Props) {
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [forgetting, setForgetting] = useState(false)
  const checkedIdentity = useRef('')
  const parameter = (id: string) => values.parameters[id]
  const set = onParameterChange
  const host = String(parameter('remote_host') || '').trim()
  const executionLocation = String(parameter('execution_location') || '本地运行')
  const remoteRequested = executionLocation === 'SSH 远程服务器'
  const port = Number(parameter('remote_port') || 22)
  const username = String(parameter('remote_username') || '').trim()
  const password = String(parameter('remote_password') || '')
  const remember = parameter('remember_password') !== false
  const identity = JSON.stringify([host, port, username])
  const signature = JSON.stringify([host, port, username, password === REMEMBERED_PASSWORD_VALUE ? '' : password, remember])
  const currentCheck = connectionCheck?.signature === signature ? connectionCheck : null
  const ready = currentCheck?.state === 'success'
  const checking = currentCheck?.state === 'checking'

  useEffect(() => {
    if (!remoteRequested) return
    if (!remember) {
      checkedIdentity.current = ''
      if (password === REMEMBERED_PASSWORD_VALUE) set('remote_password', '')
      return
    }
    if (!host || !username) return
    if (password === REMEMBERED_PASSWORD_VALUE) {
      if (checkedIdentity.current === identity) return
      checkedIdentity.current = ''
      set('remote_password', '')
      return
    }
    if (password || checkedIdentity.current === identity) return
    let cancelled = false
    checkedIdentity.current = identity
    getYoloRemoteCredentialStatus(values.parameters).then((result) => {
      if (!cancelled && result.remembered) {
        set('remote_password', REMEMBERED_PASSWORD_VALUE)
        setNotice('已自动填充该服务器保存的 SSH 密码。')
      }
    }).catch(() => { if (!cancelled) checkedIdentity.current = '' })
    return () => { cancelled = true }
  }, [host, port, username, password, remember, identity, remoteRequested])

  const chooseDirectory = async (fieldId: string, append = false) => {
    setError('')
    try {
      const paths = await choosePaths('directory')
      if (!paths[0]) return
      const current = String(parameter(fieldId) || '').trim()
      set(fieldId, append && current ? `${current}\n${paths[0]}` : paths[0])
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法选择文件夹')
    }
  }

  const testConnection = async () => {
    if (!remoteRequested) return
    setConnectionCheck({ signature, state: 'checking' })
    setError('')
    setNotice('')
    try {
      const result = await testYoloRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        checkedIdentity.current = identity
        setConnectionCheck({ signature: JSON.stringify([host, port, username, '', true]), state: 'success', result })
        set('remote_password', REMEMBERED_PASSWORD_VALUE)
      } else {
        setConnectionCheck({ signature, state: 'success', result })
      }
    } catch (cause) {
      setConnectionCheck({ signature, state: 'failed', error: cause instanceof Error ? cause.message : 'SSH 连接失败' })
    }
  }

  const forgetPassword = async () => {
    setForgetting(true)
    setError('')
    try {
      const result = await forgetYoloRemotePassword(values.parameters)
      checkedIdentity.current = ''
      set('remote_password', '')
      set('remember_password', false)
      setConnectionCheck(null)
      setNotice(result.message)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '清除密码失败')
    } finally {
      setForgetting(false)
    }
  }

  const run = () => {
    setError('')
    if (!String(parameter('image_dirs') || '').trim()) {
      setError(`请填写${remoteRequested ? '服务器' : '本机'}上的图片目录，每行一个绝对路径。`)
      return
    }
    if (remoteRequested && (!ready || !currentCheck?.result)) {
      setError('请先测试 SSH 连接，成功后再开始制作。')
      return
    }
    onRun(remoteRequested && currentCheck?.result ? { remote_connection_token: currentCheck.result.connectionToken } : {})
    if (remoteRequested) set('remote_password', remember ? REMEMBERED_PASSWORD_VALUE : '')
  }

  return <>
    <section className="workspace-section calibration-section">
      <div className="section-title"><icons.Settings size={21} /><h2>运行位置</h2></div>
      <div className="calibration-location-switch" role="group" aria-label="量化数据集运行位置">
        {['本地运行', 'SSH 远程服务器'].map((location) => <button type="button" className={executionLocation === location ? 'active' : ''} aria-pressed={executionLocation === location} onClick={() => { set('execution_location', location); setError('') }} key={location}>{location === '本地运行' ? <icons.Box size={17} /> : <icons.Code2 size={17} />}{location}</button>)}
      </div>
      <p className="section-description calibration-location-description">{remoteRequested ? '目录填写远程服务器上的绝对路径；执行前需要先测试 SSH 连接。' : '直接处理本机目录，使用平台当前 Python 环境，不需要 SSH 连接。'}</p>
    </section>
    <section className="workspace-section calibration-section">
      <div className="section-title"><icons.Folder size={21} /><h2>{remoteRequested ? '服务器' : '本机'}数据路径</h2></div>
      <div className="calibration-form">
        <label className="wide"><span>图片目录 <em>必填；每行一个，可填写多个</em></span><span className="calibration-directory-input"><textarea rows={3} value={String(parameter('image_dirs') || '')} onChange={(event) => set('image_dirs', event.target.value)} placeholder={remoteRequested ? '/home/dell/dataset/images\n/home/dell/dataset/images_extra' : '/mnt/data/dataset/images'} />{!remoteRequested ? <button type="button" onClick={() => void chooseDirectory('image_dirs', true)}>添加文件夹</button> : null}</span></label>
        <label className="wide"><span>标注目录 <em>可选；按顺序对应图片目录，不填则在图片目录查找</em></span><span className="calibration-directory-input"><textarea rows={3} value={String(parameter('annotation_dirs') || '')} onChange={(event) => set('annotation_dirs', event.target.value)} placeholder={remoteRequested ? '/home/dell/dataset/annotations' : '/mnt/data/dataset/annotations'} />{!remoteRequested ? <button type="button" onClick={() => void chooseDirectory('annotation_dirs', true)}>添加文件夹</button> : null}</span></label>
        <label className="wide"><span>负样本图片目录 <em>可选；每行一个，不需要标注</em></span><span className="calibration-directory-input"><textarea rows={2} value={String(parameter('negative_dirs') || '')} onChange={(event) => set('negative_dirs', event.target.value)} placeholder={remoteRequested ? '/home/dell/dataset/negative' : '/mnt/data/dataset/negative'} />{!remoteRequested ? <button type="button" onClick={() => void chooseDirectory('negative_dirs', true)}>添加文件夹</button> : null}</span></label>
        <label className="wide"><span>输出目录 <em>留空时默认在第一个图片目录旁创建 calib_dataset；已有文件不会覆盖</em></span><span className="calibration-directory-input single-line"><input value={String(parameter('output_dir') || '')} onChange={(event) => set('output_dir', event.target.value)} placeholder={remoteRequested ? '/home/dell/dataset/calib_dataset' : '/mnt/data/dataset/calib_dataset'} />{!remoteRequested ? <button type="button" onClick={() => void chooseDirectory('output_dir')}>选择文件夹</button> : null}</span></label>
        <label><span>标注格式</span><select value={String(parameter('annotation_format') || 'auto')} onChange={(event) => set('annotation_format', event.target.value)}><option value="auto">自动识别</option><option value="xml">VOC XML</option><option value="json">LabelMe JSON</option></select></label>
        <label><span>抽取图片数量</span><input type="number" min="1" max="100000" value={String(parameter('num_samples') ?? 128)} onChange={(event) => set('num_samples', event.target.value === '' ? '' : Number(event.target.value))} /></label>
        <label><span>随机种子</span><input type="number" value={String(parameter('random_seed') ?? 42)} onChange={(event) => set('random_seed', event.target.value === '' ? '' : Number(event.target.value))} /></label>
        <p className="wide calibration-hint">抽样顺序：先保留全部非 copy 原始图片；数量不足时，再从文件名含 copy 的扩充图片和负样本中按比例补齐。</p>
      </div>
    </section>
    {remoteRequested ? <section className="workspace-section calibration-section">
      <div className="section-title"><icons.Settings size={21} /><h2>SSH 服务器</h2></div>
      <div className="calibration-form">
        <label><span>服务器 IP / 主机名</span><input value={host} onChange={(event) => set('remote_host', event.target.value)} /></label>
        <label><span>SSH 端口</span><input type="number" value={String(parameter('remote_port') ?? 22)} onChange={(event) => set('remote_port', event.target.value === '' ? '' : Number(event.target.value))} /></label>
        <label><span>用户名</span><input value={username} onChange={(event) => set('remote_username', event.target.value)} /></label>
        <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) set('remote_password', '') }} onChange={(event) => set('remote_password', event.target.value)} placeholder="首次连接时输入" /></label>
        <label className="wide"><span>远程 Python 解释器 <em>需要已安装 Pillow；可填虚拟环境 Python 的绝对路径</em></span><input value={String(parameter('remote_python') || '')} onChange={(event) => set('remote_python', event.target.value)} placeholder="python3" /></label>
      </div>
      <label className="calibration-remember"><input type="checkbox" checked={remember} onChange={(event) => set('remember_password', event.target.checked)} />记住 SSH 密码</label>
    </section> : null}
    <section className="remote-inference-section yolo-run-section calibration-run-section">
      <div className="remote-run-actions">
        {remoteRequested ? <button className="connection-test-button" onClick={testConnection} disabled={!host || !username || checking || running || forgetting}><icons.Settings size={17} />{checking ? '正在连接…' : ready ? '重新测试连接' : '测试 SSH 连接'}</button> : null}
        <button className="run-button" onClick={run} disabled={running || checking || (remoteRequested && !ready)}><icons.Play size={18} fill="currentColor" />{platformTask ? '正在运行' : running ? '正在启动' : '开始制作'}</button>
        <span className={`remote-status ${remoteRequested ? (ready ? 'connected' : checking ? 'connecting' : 'idle') : 'connected'}`}><i />{remoteRequested ? (ready ? 'SSH 连接成功' : checking ? '正在测试连接' : '尚未测试连接') : '本地运行'}</span>
        {remoteRequested && remember ? <button className="forget-password-button" onClick={forgetPassword} disabled={forgetting || checking}>清除已保存密码</button> : null}
      </div>
      {remoteRequested && ready && currentCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>SSH 连接成功，可以开始制作。<small>设备指纹：{currentCheck.result.fingerprint}</small></span></div> : null}
      {notice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      {currentCheck?.state === 'failed' || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error || currentCheck?.error}</span></div> : null}
      <details className="run-output" open><summary><icons.ChevronRight size={18} />运行输出</summary><pre>{output || `开始制作后，这里会显示${remoteRequested ? '远程' : '本机'}图片统计、抽取进度和输出目录`}</pre></details>
    </section>
  </>
}
