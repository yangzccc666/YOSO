import { useEffect, useRef, useState } from 'react'
import { detectRemoteStarLabels, forgetYoloRemotePassword, getYoloRemoteCredentialStatus, testYoloRemoteConnection } from '../api'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { WorkingValues, YoloConnectionTestResult } from '../types'
import { icons } from '../icons'
import { HelpTip } from './HelpTip'

type Props = {
  values: WorkingValues
  running: boolean
  output: string
  onParameterChange: (id: string, value: string | number | boolean) => void
  onRun: (overrides?: Record<string, string | number | boolean>) => void
}

export function AutoAnnotationPanel({ values, running, output, onParameterChange: set, onRun }: Props) {
  const remote = values.parameters.annotation_mode === '远程 STAR 标注'
  const p = values.parameters
  const host = String(p.remote_host || '').trim()
  const port = Number(p.remote_port || 22)
  const user = String(p.remote_username || '').trim()
  const password = String(p.remote_password || '')
  const remember = p.remember_password !== false
  const identity = JSON.stringify([host, port, user])
  const signature = JSON.stringify([host, port, user, password === REMEMBERED_PASSWORD_VALUE ? '' : password, remember])
  const [check, setCheck] = useState<{ signature: string; result: YoloConnectionTestResult } | null>(null)
  const [testing, setTesting] = useState(false)
  const [reading, setReading] = useState(false)
  const [forgetting, setForgetting] = useState(false)
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const ready = check?.signature === signature
  const weights = values.paths.weights_file || ''
  const labels = String(p.star_labels || '')
  const latest = useRef({ weights, labels, remote })
  latest.current = { weights, labels, remote }

  useEffect(() => {
    if (!remote) return
    if (!remember) {
      if (password === REMEMBERED_PASSWORD_VALUE) set('remote_password', '')
      return
    }
    if (!host || !user || password) return
    let cancelled = false
    getYoloRemoteCredentialStatus(p).then(result => {
      if (!cancelled && result.remembered) set('remote_password', REMEMBERED_PASSWORD_VALUE)
    }).catch(() => undefined)
    return () => { cancelled = true }
  }, [identity, password, remember, remote])

  const test = async () => {
    setTesting(true)
    setError('')
    setNotice('')
    setCheck(null)
    try {
      const result = await testYoloRemoteConnection(p)
      const checkedSignature = result.passwordRemembered
        ? JSON.stringify([host, port, user, '', true]) : signature
      setCheck({ signature: checkedSignature, result })
      if (result.passwordRemembered) set('remote_password', REMEMBERED_PASSWORD_VALUE)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'SSH 连接失败')
    } finally {
      setTesting(false)
    }
  }

  const readLabels = async () => {
    setReading(true)
    setError('')
    setNotice('')
    try {
      const result = await detectRemoteStarLabels(weights)
      if (latest.current.weights !== weights || latest.current.labels !== labels || !latest.current.remote) return
      set('star_labels', result.labels.join('\n'))
      setNotice(result.message)
    } catch (e) {
      if (latest.current.weights === weights && latest.current.remote) setError(e instanceof Error ? e.message : '读取类别失败，请手动填写')
    } finally {
      setReading(false)
    }
  }

  const forget = async () => {
    setForgetting(true)
    setError('')
    try {
      const result = await forgetYoloRemotePassword(p)
      set('remote_password', '')
      set('remember_password', false)
      setCheck(null)
      setNotice(result.message)
    } catch (e) {
      setError(e instanceof Error ? e.message : '清除密码失败')
    } finally {
      setForgetting(false)
    }
  }

  return <section className="remote-inference-section">
    <div className="remote-run-actions">
      {remote ? <>
        <button className="outline-button" disabled={!weights || reading || running} onClick={() => void readLabels()}>{reading ? '正在读取…' : '读取 STAR 类别'}</button>
        <button className="connection-test-button" disabled={!host || !user || testing || running || forgetting} onClick={() => void test()}>{testing ? '正在连接…' : ready ? '重新测试连接' : '测试 SSH 连接'}</button>
      </> : null}
      <button className="run-button" disabled={running || testing || reading || forgetting || (remote && !ready)} onClick={() => {
        if (remote) {
          if (ready && check) onRun({ remote_connection_token: check.result.connectionToken })
        } else onRun()
      }}><icons.Play size={18} fill="currentColor" />{running ? '正在运行' : remote ? '开始 STAR 标注' : '开始 PT 标注'}</button>
      <HelpTip label="自动标注注意事项">{remote
        ? '图片文件夹和 STAR 都是本机路径。盒子需具备兼容的 TensorRT、CUDA、OpenCV 和 NumPy 环境；本机无需安装 TensorRT。使用与实时推理相同的 YOLO26 分头输出解码，输入尺寸由引擎决定，不跳帧。类别留空时自动读取 STAR；手动填写请按类别 ID 顺序。连接测试只检查 SSH，运行时才检查推理环境。'
        : '使用本机 Ultralytics 检测 PT 模型；类别名来自模型。'}每张原图旁生成同名 XML，不修改图片。已有标注可跳过或备份后覆盖。空标注不等于真实负样本，所有自动标注都请人工复核。</HelpTip>
      {remote && remember ? <button className="forget-password-button" disabled={testing || running || forgetting} onClick={() => void forget()}>清除已保存密码</button> : null}
    </div>
    {remote && ready ? <div className="connection-success" role="status">SSH 连接成功，可以开始 STAR 标注。设备指纹：{check?.result.fingerprint}</div> : null}
    {notice ? <p role="status">{notice}</p> : null}
    {error ? <div className="remote-error" role="alert">{error}</div> : null}
    <details className="run-output" open><summary><icons.ChevronRight size={18} />运行输出</summary><pre>{output || '运行后显示逐张标注进度和结果，XML 保存在本机原图旁。'}</pre></details>
  </section>
}
