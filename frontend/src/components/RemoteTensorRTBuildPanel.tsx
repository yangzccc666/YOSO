import { useEffect, useRef, useState } from 'react'
import { forgetRemotePassword, getRemoteCredentialStatus, testRemoteConnection } from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { ConnectionTestResult, PlatformTaskStatus, WorkingValues } from '../types'

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
  result?: ConnectionTestResult
  error?: string
}

const PATH_LINE_BREAK = /\r?\n/

function onnxPathText(raw: string | number | boolean | undefined, legacyPath: string): string {
  if (typeof raw === 'string' && raw.trim()) {
    try {
      const values: unknown = JSON.parse(raw)
      if (Array.isArray(values)) return values.map(String).join('\n')
    } catch { return raw }
    return raw
  }
  return legacyPath
}

function parseOnnxFiles(raw: string): string[] {
  const paths = raw.split(PATH_LINE_BREAK).map((path) => path.trim()).filter(Boolean)
  return [...new Set(paths)]
}

export function RemoteTensorRTBuildPanel(props: Props) {
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [forgetting, setForgetting] = useState(false)
  const checkedIdentity = useRef('')
  const parameter = (id: string) => props.values.parameters[id]
  const set = props.onParameterChange
  const host = String(parameter('host') || '').trim()
  const port = Number(parameter('port') || 22)
  const username = String(parameter('username') || '').trim()
  const password = String(parameter('password') || '')
  const remember = parameter('remember_password') !== false
  const identity = JSON.stringify([host, port, username])
  const passwordSignature = password === REMEMBERED_PASSWORD_VALUE ? '' : password
  const signature = JSON.stringify([host, port, username, passwordSignature, remember])
  const currentCheck = connectionCheck?.signature === signature ? connectionCheck : null
  const ready = currentCheck?.state === 'success'
  const checking = currentCheck?.state === 'checking'
  const onnxText = onnxPathText(parameter('onnx_files'), props.values.paths.input_onnx || '')
  const onnxFiles = parseOnnxFiles(onnxText)

  const changeOnnxPaths = (value: string) => {
    set('onnx_files', value)
  }

  useEffect(() => {
    if (!remember) {
      checkedIdentity.current = ''
      if (password === REMEMBERED_PASSWORD_VALUE) set('password', '')
      return
    }
    if (!host || !username) return
    if (password === REMEMBERED_PASSWORD_VALUE) {
      if (checkedIdentity.current === identity) return
      checkedIdentity.current = ''
      set('password', '')
      return
    }
    if (password || checkedIdentity.current === identity) return
    let cancelled = false
    checkedIdentity.current = identity
    getRemoteCredentialStatus(props.values.parameters).then((result) => {
      if (!cancelled && result.remembered) {
        set('password', REMEMBERED_PASSWORD_VALUE)
        setNotice('已自动填充该设备保存的 SSH 密码。')
      }
    }).catch(() => { if (!cancelled) checkedIdentity.current = '' })
    return () => { cancelled = true }
  }, [host, port, username, password, remember, identity])

  const testConnection = async () => {
    setConnectionCheck({ signature, state: 'checking' })
    setError('')
    setNotice('')
    try {
      const result = await testRemoteConnection(props.values.parameters)
      if (result.passwordRemembered) {
        checkedIdentity.current = identity
        setConnectionCheck({ signature: JSON.stringify([host, port, username, '', true]), state: 'success', result })
        set('password', REMEMBERED_PASSWORD_VALUE)
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
      const result = await forgetRemotePassword(props.values.parameters)
      checkedIdentity.current = ''
      set('password', '')
      set('remember_password', false)
      setConnectionCheck(null)
      setNotice(result.message)
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '清除密码失败')
    } finally {
      setForgetting(false)
    }
  }

  const run = (operation: '检查远程环境' | '构建 TensorRT 引擎') => {
    setError('')
    if (!ready) {
      setError('请先测试 SSH 连接，成功后再执行。')
      return
    }
    if (operation === '构建 TensorRT 引擎' && onnxFiles.length === 0) {
      setError('请至少填写 1 个远端量化 ONNX 文件路径。')
      return
    }
    props.onRun({ operation })
    set('password', remember ? REMEMBERED_PASSWORD_VALUE : '')
  }

  return <>
    <section className="workspace-section calibration-section">
      <div className="section-title"><icons.Folder size={21} /><h2>模型与输出</h2></div>
      <div className="calibration-form remote-onnx-form">
        <label className="wide"><span>远端量化 ONNX 路径 <em>必填；每行一个，从上到下顺序执行</em></span><textarea rows={6} value={onnxText} onChange={(event) => changeOnnxPaths(event.target.value)} placeholder={'/home/wel/Downloads/onnx_quant/model_a_quant.onnx\n/home/wel/Downloads/onnx_quant/model_b_quant.onnx'} /></label>
        <p className="wide calibration-hint">路径必须是当前 SSH 设备上的绝对路径。每个结果会在 ONNX 同级目录生成同名的 <strong>.plan</strong> 和 <strong>.trtexec.log</strong>。当前共 <strong>{onnxFiles.length}</strong> 个有效路径。</p>
      </div>
      <div className="calibration-form">
        <label><span>构建精度</span><select value={String(parameter('build_mode') || '自动最佳（--best）')} onChange={(event) => set('build_mode', event.target.value)}><option>自动最佳（--best）</option><option>FP16</option><option>FP32</option></select></label>
        <label className="wide"><span>远端 trtexec 路径</span><input value={String(parameter('trtexec_path') || '')} onChange={(event) => set('trtexec_path', event.target.value)} /></label>
        <label className="wide"><span>远端后台任务根目录 <em>仅用于保存运行状态</em></span><input value={String(parameter('remote_workspace') || '')} onChange={(event) => set('remote_workspace', event.target.value)} /></label>
      </div>
      <div className="remote-build-options">
        <label><input type="checkbox" checked={parameter('run_benchmark') !== false} onChange={(event) => set('run_benchmark', event.target.checked)} />构建后执行性能测试</label>
        <label><input type="checkbox" checked={parameter('keep_remote_files') !== false} onChange={(event) => set('keep_remote_files', event.target.checked)} />保留远端后台任务状态目录</label>
        <label><input type="checkbox" checked={Boolean(parameter('overwrite_output'))} onChange={(event) => set('overwrite_output', event.target.checked)} />允许覆盖远端同名结果</label>
      </div>
    </section>
    <section className="workspace-section calibration-section">
      <div className="section-title"><icons.Settings size={21} /><h2>AI 推理盒子 SSH</h2></div>
      <div className="calibration-form">
        <label><span>设备 IP / 主机名</span><input value={host} onChange={(event) => set('host', event.target.value)} /></label>
        <label><span>SSH 端口</span><input type="number" value={String(parameter('port') ?? 22)} onChange={(event) => set('port', event.target.value === '' ? '' : Number(event.target.value))} /></label>
        <label><span>用户名</span><input value={username} onChange={(event) => set('username', event.target.value)} /></label>
        <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) set('password', '') }} onChange={(event) => set('password', event.target.value)} placeholder="首次连接时输入" /></label>
      </div>
      <label className="calibration-remember"><input type="checkbox" checked={remember} onChange={(event) => set('remember_password', event.target.checked)} />记住 SSH 密码</label>
    </section>
    <section className="remote-inference-section yolo-run-section calibration-run-section">
      <div className="remote-run-actions remote-build-actions">
        <button className="connection-test-button" onClick={testConnection} disabled={!host || !username || checking || props.running || forgetting}><icons.Settings size={17} />{checking ? '正在连接…' : ready ? '重新测试连接' : '测试 SSH 连接'}</button>
        <button className="connection-test-button" onClick={() => run('检查远程环境')} disabled={!ready || props.running}><icons.SlidersHorizontal size={17} />检查 TensorRT 环境</button>
        <button className="run-button" onClick={() => run('构建 TensorRT 引擎')} disabled={!ready || props.running}><icons.Play size={18} fill="currentColor" />{props.platformTask ? '正在构建' : props.running ? '正在启动' : '开始构建'}</button>
        <span className={`remote-status ${ready ? 'connected' : checking ? 'connecting' : 'idle'}`}><i />{ready ? 'SSH 连接成功' : checking ? '正在测试连接' : '尚未测试连接'}</span>
        {remember ? <button className="forget-password-button" onClick={forgetPassword} disabled={forgetting || checking}>清除已保存密码</button> : null}
      </div>
      {ready && currentCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>SSH 连接成功，可以检查环境或开始构建。<small>设备指纹：{currentCheck.result.fingerprint}</small></span></div> : null}
      {notice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      {currentCheck?.state === 'failed' || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error || currentCheck?.error}</span></div> : null}
      <details className="run-output" open><summary><icons.ChevronRight size={18} />构建日志</summary><pre>{props.output || '开始后会显示远端文件检查、TensorRT 构建日志、引擎大小和远端保存路径'}</pre></details>
    </section>
  </>
}
