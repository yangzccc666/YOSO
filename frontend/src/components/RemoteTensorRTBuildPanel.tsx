import { useEffect, useRef, useState } from 'react'
import { CollapsibleSection } from './CollapsibleSection'
import { confirmStop } from '../confirmActions'
import { clearTensorRTWaiting, discoverTensorRTBuilds, enqueueTensorRTBuild, forgetRemotePassword, getRemoteCredentialStatus, getTensorRTQueue, pauseTensorRTQueue, reconnectTensorRTQueueItem, removeTensorRTQueueItem, stopTensorRTQueueItem, testRemoteConnection } from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { ConnectionTestResult, PlatformTaskStatus, TensorRTQueueStatus, WorkingValues } from '../types'
import { SshConnectionHistoryInput } from './SshConnectionHistoryInput'
import { TensorRTQueuePanel } from './TensorRTQueuePanel'
import { HelpTip } from './HelpTip'
import { retainTensorRTQueue, tensorRTReconnectParameters } from '../tensorRTQueue'

type Props = {
  itemId: string
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
  const [submitting, setSubmitting] = useState(false)
  const [queueBusy, setQueueBusy] = useState(false)
  const [queue, setQueue] = useState<TensorRTQueueStatus | null>(null)
  const [queueError, setQueueError] = useState('')
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
  const latestInput = useRef({ signature, onnxText })
  latestInput.current = { signature, onnxText }
  const applyQueue = (next: TensorRTQueueStatus) => setQueue((current) => retainTensorRTQueue(current, next))

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const next = await getTensorRTQueue()
        if (!cancelled) applyQueue(next)
      } catch (cause) {
        if (!cancelled) setQueueError(cause instanceof Error ? cause.message : '读取构建队列失败')
      } finally { if (!cancelled) timer = window.setTimeout(poll, 800) }
    }
    void poll()
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [])

  const queueAction = async (action: () => Promise<TensorRTQueueStatus>) => {
    setQueueBusy(true); setQueueError('')
    try { applyQueue(await action()) }
    catch (cause) { setQueueError(cause instanceof Error ? cause.message : '操作队列失败') }
    finally { setQueueBusy(false) }
  }

  const stopQueueItem = (id: string) => {
    const item = queue?.items.find((entry) => entry.id === id)
    if (!item || queueBusy || item.status !== 'running') return
    if (!confirmStop(`TensorRT 构建：${item.source}`, '将终止该远端构建，并暂停后续队列；尚未完成的引擎可能不可用，已有日志和结果不会主动删除。')) return
    void queueAction(() => stopTensorRTQueueItem(id))
  }

  const reconnectItem = (id: string) => {
    const item = queue?.items.find(entry => entry.id === id)
    if (!item || queueBusy) return
    void queueAction(() => reconnectTensorRTQueueItem(id, tensorRTReconnectParameters(props.values.parameters, item)))
  }

  const discover = () => {
    if (!ready || queueBusy) return
    void queueAction(async () => {
      const result = await discoverTensorRTBuilds(props.itemId, props.values.parameters)
      setNotice(result.message)
      return result
    })
  }

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
      if (latestInput.current.signature !== signature) return
      if (result.passwordRemembered) {
        checkedIdentity.current = identity
        setConnectionCheck({ signature: JSON.stringify([host, port, username, '', true]), state: 'success', result })
        set('password', REMEMBERED_PASSWORD_VALUE)
      } else {
        setConnectionCheck({ signature, state: 'success', result })
      }
    } catch (cause) {
      if (latestInput.current.signature !== signature) return
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
      setError('请至少填写 1 个远端 ONNX 文件路径。')
      return
    }
    props.onRun({ operation })
    set('password', remember ? REMEMBERED_PASSWORD_VALUE : '')
  }

  const enqueue = async () => {
    setError(''); setNotice(''); setQueueError('')
    if (!ready) { setError('请先测试 SSH 连接，成功后再加入队列。'); return }
    if (!onnxFiles.length) { setError('请至少填写 1 个远端 ONNX 文件路径。'); return }
    setSubmitting(true)
    try {
      const result = await enqueueTensorRTBuild(props.itemId, props.values)
      applyQueue(result)
      const unchanged = latestInput.current.onnxText === onnxText
      setNotice(result.message + (unchanged ? ' 已提交的路径已从输入框清空，可继续填写下一批。' : ' 当前新编辑的路径已保留。'))
      if (unchanged) set('onnx_files', '[]')
      if (latestInput.current.signature === signature) set('password', remember ? REMEMBERED_PASSWORD_VALUE : '')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '加入队列失败') }
    finally { setSubmitting(false) }
  }

  return <>
    <CollapsibleSection className="workspace-section calibration-section" count={7} heading={<div className="section-title"><icons.Folder size={21} /><h2>模型与输出</h2></div>}>
      <div className="calibration-form remote-onnx-form">
        <label className="wide"><span>远端 ONNX 路径 <em>必填 · 每行一个 · 当前 {onnxFiles.length} 个</em><HelpTip label="远端 ONNX 路径说明">路径必须是当前 SSH 设备上的绝对路径，从上到下顺序执行。每个结果会在 ONNX 同级目录生成同名的 .plan 和 .trtexec.log。FP16/FP32 请使用未量化的 FP32 ONNX，不要选择之前的 _quant.onnx。更换精度时使用不同源文件名或目录，避免覆盖已有对照引擎。</HelpTip></span><textarea rows={6} value={onnxText} onChange={(event) => changeOnnxPaths(event.target.value)} placeholder={'/srv/models/model_a_cut.onnx\n/srv/models/model_b_cut.onnx'} /></label>
      </div>
      <div className="calibration-form">
        <label><span className="help-heading">构建精度<HelpTip label="TensorRT 构建精度说明">FP16：优先测试的性能/数值精度折中，允许 FP32 回退；FP32：禁用 FP16/INT8 构建开关，并通过 --noTF32 禁用 TF32，通常更慢、占用更多资源。两者都必须使用未量化 ONNX；已含 Q/DQ 量化算子时会阻止执行，因为切换精度不会恢复已损失的信息。“自动最佳 --best”偏向速度，不是检测效果最佳，可能使用低精度。更高数值精度不保证检出率提高，应对同一验证集比较漏检/误检。</HelpTip></span><select aria-label="TensorRT 构建精度" value={String(parameter('build_mode') || '自动最佳（--best）')} onChange={(event) => set('build_mode', event.target.value)}><option>自动最佳（--best）</option><option>FP16</option><option>FP32</option></select></label>
        <label className="wide"><span>远端 trtexec 路径</span><input value={String(parameter('trtexec_path') || '')} onChange={(event) => set('trtexec_path', event.target.value)} /></label>
        <label className="wide"><span>远端后台任务根目录 <em>仅用于保存运行状态</em></span><input value={String(parameter('remote_workspace') || '')} onChange={(event) => set('remote_workspace', event.target.value)} /></label>
      </div>
      <div className="remote-build-options">
        <label><input type="checkbox" checked={parameter('run_benchmark') !== false} onChange={(event) => set('run_benchmark', event.target.checked)} />构建后执行性能测试</label>
        <label><input type="checkbox" checked={parameter('keep_remote_files') !== false} onChange={(event) => set('keep_remote_files', event.target.checked)} />保留远端后台任务状态目录</label>
        <label><input type="checkbox" checked={Boolean(parameter('overwrite_output'))} onChange={(event) => set('overwrite_output', event.target.checked)} />允许覆盖远端同名结果</label>
      </div>
    </CollapsibleSection>
    <CollapsibleSection className="workspace-section calibration-section" count={5} heading={<div className="section-title"><icons.Settings size={21} /><h2>AI 推理盒子 SSH</h2></div>}>
      <div className="calibration-form">
        <label><span>设备 IP / 主机名</span><SshConnectionHistoryInput host={host} disabled={checking || submitting} onHostChange={(value) => set('host', value)} onSelect={(profile) => {
          set('port', profile.port); set('username', profile.username); set('remember_password', true); set('password', profile.passwordRemembered ? REMEMBERED_PASSWORD_VALUE : ''); setConnectionCheck(null); setNotice(profile.passwordRemembered ? '已选择历史设备，并填充保存的 SSH 密码。' : '已选择历史设备，请输入 SSH 密码。')
        }} /></label>
        <label><span>SSH 端口</span><input type="number" value={String(parameter('port') ?? 22)} onChange={(event) => set('port', event.target.value === '' ? '' : Number(event.target.value))} /></label>
        <label><span>用户名</span><input value={username} onChange={(event) => set('username', event.target.value)} /></label>
        <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) set('password', '') }} onChange={(event) => set('password', event.target.value)} placeholder="首次连接时输入" /></label>
      </div>
      <label className="calibration-remember"><input type="checkbox" checked={remember} onChange={(event) => set('remember_password', event.target.checked)} />记住 SSH 密码</label>
    </CollapsibleSection>
    <section className="remote-inference-section yolo-run-section calibration-run-section">
      <div className="remote-run-actions remote-build-actions">
        <button className="connection-test-button" onClick={testConnection} disabled={!host || !username || checking || submitting || forgetting}><icons.Settings size={17} />{checking ? '正在连接…' : ready ? '重新测试连接' : '测试 SSH 连接'}</button>
        <button className="connection-test-button" onClick={() => run('检查远程环境')} disabled={!ready || props.running || Boolean(queue?.runningCount)}><icons.SlidersHorizontal size={17} />检查 TensorRT 环境</button>
        <button className="connection-test-button" onClick={discover} disabled={!ready || queueBusy}>查找远端任务</button><HelpTip label="找回远端任务说明">旧版本丢失的构建可点击“查找远端任务”。若旧任务没有描述文件，请在上方 ONNX 路径框填入原远端路径，并保持原任务根目录后再查找。远端会保留用于重连的最小状态文件。</HelpTip>
        <button className="run-button" onClick={() => void enqueue()} disabled={!ready || submitting || checking || forgetting}><icons.Play size={18} fill="currentColor" />{submitting ? '正在提交…' : '加入队列并自动执行'}</button>
        <span className={`remote-status ${ready ? 'connected' : checking ? 'connecting' : 'idle'}`}><i />{ready ? 'SSH 连接成功' : checking ? '正在测试连接' : '尚未测试连接'}</span>
        {remember ? <button className="forget-password-button" onClick={forgetPassword} disabled={forgetting || checking}>清除已保存密码</button> : null}
      </div>
      {ready && currentCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>SSH 连接成功，可以检查环境或开始构建。<small>设备指纹：{currentCheck.result.fingerprint}</small></span></div> : null}
      {notice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      {currentCheck?.state === 'failed' || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error || currentCheck?.error}</span></div> : null}
      {queueError ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{queueError}</span></div> : null}
      <TensorRTQueuePanel queue={queue} busy={queueBusy} onReconnect={reconnectItem} onPause={(paused) => {
        if (paused && !window.confirm('确定暂停后续构建队列吗？\n当前构建会继续运行；等待项将保留，点击“继续队列”后恢复调度。')) return
        void queueAction(() => pauseTensorRTQueue(paused, paused ? undefined : props.values.parameters))
      }} onClear={() => {
        if (window.confirm('确定清空所有等待中的构建任务吗？\n当前运行的构建不受影响，清空后需要重新添加等待任务。')) void queueAction(clearTensorRTWaiting)
      }} onRemove={(id) => {
        const item = queue?.items.find((entry) => entry.id === id)
        if (item?.status === 'disconnected' && !window.confirm(`确定仅移除“${item.source}”的本地记录吗？\n这不会停止远端构建，之后继续队列可能占用同一设备。请先确认远端已结束；需要时可通过“查找远端任务”找回。`)) return
        if (item?.status === 'queued' && !window.confirm(`确定移出等待任务“${item.source}”吗？\n移出后不会自动运行，需要重新添加。`)) return
        void queueAction(() => removeTensorRTQueueItem(id, item?.status === 'disconnected'))
      }} onStop={stopQueueItem} />
      <details className="run-output" open><summary><icons.ChevronRight size={18} />构建日志</summary><pre>{props.output || '开始后会显示远端文件检查、TensorRT 构建日志、引擎大小和远端保存路径'}</pre></details>
    </section>
  </>
}
