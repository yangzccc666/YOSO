import { useEffect, useState, type ReactNode } from 'react'

type ActivationStatus = { activated: boolean; message: string; customer?: string; expiresAt?: string | null }

export function ActivationGate({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<ActivationStatus | null>(null)
  const [key, setKey] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    let disposed = false
    const check = async () => {
      try {
        const response = await fetch('/api/license/status', { cache: 'no-store' })
        if (!response.ok) throw new Error('无法检查激活状态，请重试。')
        const value = await response.json() as ActivationStatus
        if (!disposed) { setStatus(value); setError('') }
      } catch (cause) {
        if (!disposed) { setStatus({ activated: false, message: '无法连接本地平台服务。' }); setError(String(cause)) }
      }
    }
    void check()
    const timer = window.setInterval(() => { void check() }, 30000)
    window.addEventListener('yolo-activation-required', check)
    return () => { disposed = true; window.clearInterval(timer); window.removeEventListener('yolo-activation-required', check) }
  }, [])

  const activate = async () => {
    setBusy(true)
    setError('')
    try {
      const response = await fetch('/api/license/activate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ key }),
      })
      const value = await response.json()
      if (!response.ok) throw new Error(value.error || '激活失败，请检查密钥。')
      setStatus(value)
      setKey('')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '激活失败。') }
    finally { setBusy(false) }
  }

  if (status?.activated) return <>{children}</>
  return <main className="activation-page"><section className="activation-card">
    <div className="activation-brand">YOLO</div>
    <h1>YOLO数据处理平台</h1>
    <h2>{status ? '首次使用 · 密钥激活' : '正在检查激活状态…'}</h2>
    <p>{status?.message || '请稍候。'}</p>
    {status ? <form onSubmit={(event) => { event.preventDefault(); void activate() }}>
      <label htmlFor="activation-key">激活密钥</label>
      <textarea id="activation-key" rows={5} autoComplete="off" spellCheck={false} value={key} onChange={(event) => setKey(event.target.value)} placeholder="请粘贴提供者发给你的完整密钥" disabled={busy} />
      {error ? <p className="activation-error" role="alert">{error}</p> : null}
      <button className="run-button" type="submit" disabled={busy || !key.trim()}>{busy ? '正在验证…' : '激活并进入平台'}</button>
      <small>激活状态保存在本机，下次打开自动验证，无需重复输入。</small>
    </form> : null}
  </section></main>
}
