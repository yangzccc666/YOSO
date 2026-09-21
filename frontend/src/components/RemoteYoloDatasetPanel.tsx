import { useEffect, useRef, useState } from 'react'
import { forgetYoloRemotePassword, getYoloRemoteCredentialStatus, testYoloRemoteConnection } from '../api'
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

export function RemoteYoloDatasetPanel({ values, output, running, platformTask, onParameterChange, onRun }: Props) {
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [credentialNotice, setCredentialNotice] = useState('')
  const [error, setError] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const checkedCredentialIdentity = useRef('')

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

  const testConnection = async () => {
    const signature = connectionSignature
    setConnectionCheck({ signature, state: 'checking' })
    setCredentialNotice('')
    setError('')
    try {
      const result = await testYoloRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        checkedCredentialIdentity.current = identitySignature
        const rememberedSignature = JSON.stringify([host, port, username, '', true])
        setConnectionCheck({ signature: rememberedSignature, state: 'success', result })
        onParameterChange('remote_password', REMEMBERED_PASSWORD_VALUE)
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
      const result = await forgetYoloRemotePassword(values.parameters)
      onParameterChange('remote_password', '')
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

  const run = () => {
    setError('')
    if (!remoteRequested) {
      onRun()
      return
    }
    if (!connectionReady || !currentConnectionCheck?.result) {
      setError('请先测试 SSH 连接，连接成功后再执行远程数据集划分。')
      return
    }
    onRun({ remote_connection_token: currentConnectionCheck.result.connectionToken })
    onParameterChange('remote_password', rememberPassword ? REMEMBERED_PASSWORD_VALUE : '')
  }

  const connectionError = currentConnectionCheck?.state === 'failed'
    ? currentConnectionCheck.error || 'SSH 连接测试失败'
    : ''
  const status = !remoteRequested
    ? { className: 'local', label: '本地运行' }
    : connectionReady
      ? { className: 'connected', label: 'SSH 连接成功' }
      : testingConnection
        ? { className: 'connecting', label: '正在测试连接' }
        : { className: 'idle', label: '尚未测试连接' }

  return (
    <section className="remote-inference-section yolo-run-section">
      <div className="remote-run-actions">
        <button className="connection-test-button" onClick={testConnection} disabled={!remoteRequested || testingConnection || submitting || running}>
          <icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试 SSH 连接'}
        </button>
        <button className="run-button" onClick={run} disabled={running || testingConnection || (remoteRequested && !connectionReady)}>
          <icons.Play size={18} fill="currentColor" />
          {platformTask ? '正在运行' : running ? '正在启动' : remoteRequested ? '执行远程划分' : '本地一键运行'}
        </button>
        <span className={`remote-status ${status.className}`}><i />{status.label}</span>
        <small>{remoteRequested ? (rememberPassword ? '密码默认记住；连接参数变化后需要重新测试。' : '密码不会保存；请先测试连接再执行。') : 'SSH 信息为空，数据将在本机处理。'}</small>
        {remoteRequested && rememberPassword ? <button className="forget-password-button" onClick={forgetPassword} disabled={submitting || testingConnection}>清除已保存密码</button> : null}
      </div>

      {connectionReady && currentConnectionCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>{currentConnectionCheck.result.message}<small>设备指纹：{currentConnectionCheck.result.fingerprint}</small></span></div> : null}
      {credentialNotice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{credentialNotice}</span></div> : null}
      {connectionError || error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{connectionError || error}</span></div> : null}

      <details className="run-output" open>
        <summary><icons.ChevronRight size={18} />运行输出</summary>
        <pre>{output || (remoteRequested ? '连接成功并执行后，将在这里显示远程划分进度和结果' : '运行后将在这里显示本地处理进度和结果')}</pre>
      </details>
    </section>
  )
}
