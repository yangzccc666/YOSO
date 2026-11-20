import { useEffect, useRef, useState, type FormEvent } from 'react'
import { createPortal } from 'react-dom'
import { Timer } from 'lucide-react'
import { getCountdowns, removeCountdown, startCountdown, type Countdown, type CountdownPayload } from '../api'
import { confirmStop } from '../confirmActions'
import { icons } from '../icons'

function timeRemaining(deadlineMs: number, now: number): string {
  const seconds = Math.max(0, Math.ceil((deadlineMs - now) / 1000))
  return [Math.floor(seconds / 3600), Math.floor(seconds / 60) % 60, seconds % 60]
    .map(unit => String(unit).padStart(2, '0')).join(':')
}

export function CountdownControl() {
  const [open, setOpen] = useState(false)
  const [payload, setPayload] = useState<CountdownPayload | null>(null)
  const [reload, setReload] = useState(0)
  const [note, setNote] = useState('')
  const [hours, setHours] = useState('')
  const [minutes, setMinutes] = useState('5')
  const [seconds, setSeconds] = useState('')
  const [now, setNow] = useState(Date.now)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const root = useRef<HTMLDivElement>(null)
  const alert = useRef<HTMLElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const noteInput = useRef<HTMLInputElement>(null)
  const timers = payload?.timers ?? []
  const activeCount = timers.filter(timer => timer.deadlineMs > now).length
  const dueCount = timers.length - activeCount
  const reminder = payload?.nativeNotifications === false ? timers.find(timer => timer.deadlineMs <= now) : undefined
  const syncNativeAcknowledgements = payload?.nativeNotifications === true && dueCount > 0 && !saving
  const ticking = activeCount > 0
  const duration = Number(hours) * 3600 + Number(minutes) * 60 + Number(seconds)

  useEffect(() => {
    const request = new AbortController()
    getCountdowns(request.signal).then(value => {
      if (request.signal.aborted) return
      setNow(Date.now())
      setPayload(value)
    }).catch(cause => {
      if (!request.signal.aborted) setError(cause.message || '读取倒计时失败。')
    })
    return () => request.abort()
  }, [reload])

  useEffect(() => {
    if (!ticking) return
    const tick = () => setNow(Date.now())
    const ticker = window.setInterval(tick, 250)
    document.addEventListener('visibilitychange', tick)
    return () => {
      window.clearInterval(ticker)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [ticking])

  useEffect(() => {
    if (!syncNativeAcknowledgements) return
    const request = new AbortController()
    let refreshing = false
    const refresh = async () => {
      if (refreshing || document.hidden) return
      refreshing = true
      try {
        const next = await getCountdowns(request.signal)
        if (!request.signal.aborted) setPayload(current => current?.timers.length === next.timers.length
          && current.timers.every((timer, index) => timer.id === next.timers[index].id)
          && current.nativeNotifications === next.nativeNotifications ? current : next)
      } catch {
        // Keep the current display when a backend hot reload interrupts a poll.
      } finally {
        refreshing = false
      }
    }
    const ticker = window.setInterval(() => void refresh(), 800)
    document.addEventListener('visibilitychange', refresh)
    return () => {
      request.abort()
      window.clearInterval(ticker)
      document.removeEventListener('visibilitychange', refresh)
    }
  }, [syncNativeAcknowledgements])

  useEffect(() => {
    if (!open) return
    noteInput.current?.focus()
    const outside = (event: PointerEvent) => {
      const target = event.target as Node
      if (!root.current?.contains(target) && !alert.current?.contains(target)) setOpen(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape' || event.isComposing || !root.current?.contains(document.activeElement)) return
      event.stopPropagation()
      trigger.current?.focus()
      setOpen(false)
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [open])

  const save = async (request: () => Promise<CountdownPayload>) => {
    if (saving) return false
    setSaving(true)
    setError('')
    try {
      const value = await request()
      setNow(Date.now())
      setPayload(value)
      return true
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法保存倒计时。')
      return false
    } finally {
      setSaving(false)
    }
  }
  const submit = async (event: FormEvent) => {
    event.preventDefault()
    if (!note.trim() || duration <= 0) return
    if (await save(() => startCountdown(note.trim(), duration))) {
      setNote('')
      const focused = document.activeElement
      if (focused === document.body || root.current?.contains(focused)) noteInput.current?.focus()
    }
  }
  const remove = (timer: Countdown) => {
    if (timer.deadlineMs > Date.now() && !confirmStop(timer.note, '取消后此倒计时将不再提醒。')) return
    void save(() => removeCountdown(timer.id))
  }
  const summary = `倒计时（计时中 ${activeCount} 项，已到期 ${dueCount} 项）`

  return <div className="countdown-control" ref={root}>
    <button ref={trigger} type="button" className={`history-trigger countdown-trigger${dueCount > 0 ? ' due' : ''}`} aria-label={summary}
      title={summary} aria-expanded={open} aria-controls={open ? 'countdown-panel' : undefined}
      onClick={() => setOpen(value => !value)}>
      <Timer size={17} /><span className="countdown-trigger-label">{dueCount ? '倒计时到期' : '倒计时'}</span>
      {timers.length > 0 ? <span className={dueCount ? 'countdown-count due' : 'countdown-count'}>{dueCount || activeCount}</span> : null}
    </button>
    {open ? <section id="countdown-panel" className="countdown-panel" role="dialog" aria-label="倒计时" aria-modal="false">
      <header><strong>倒计时</strong><button type="button" aria-label="收起倒计时" onClick={() => { setOpen(false); trigger.current?.focus() }}><icons.X size={17} /></button></header>
      <div className="countdown-panel-body">
        <form onSubmit={event => void submit(event)}>
          <label>备注<input ref={noteInput} value={note} onChange={event => setNote(event.target.value)} placeholder="例如：检查远程训练结果" maxLength={120} required /></label>
          <div className="countdown-duration">
            <label>小时<input type="number" min={0} max={9999} step={1} value={hours} onChange={event => setHours(event.target.value)} placeholder="0" /></label>
            <label>分钟<input type="number" min={0} max={59} step={1} value={minutes} onChange={event => setMinutes(event.target.value)} placeholder="0" /></label>
            <label>秒<input type="number" min={0} max={59} step={1} value={seconds} onChange={event => setSeconds(event.target.value)} placeholder="0" /></label>
          </div>
          <button type="submit" className="countdown-start" disabled={!payload || saving || !note.trim() || duration <= 0}><icons.Plus size={15} />开始倒计时</button>
        </form>
        {error ? <div className="countdown-error" role="status">{error} {!payload ? <button type="button" onClick={() => { setError(''); setReload(value => value + 1) }}>重试</button> : null}</div> : null}
        {!payload && !error ? <p className="countdown-empty">正在读取倒计时…</p> : null}
        {timers.length ? <ul className="countdown-list" aria-label="倒计时列表">
          {timers.map(timer => {
            const due = timer.deadlineMs <= now
            return <li key={timer.id} className={due ? 'due' : undefined}>
              <span className="countdown-note">{timer.note}</span>
              <span className={due ? 'countdown-remaining due' : 'countdown-remaining'}>{due ? '已到期' : timeRemaining(timer.deadlineMs, now)}</span>
              <button type="button" disabled={saving} aria-label={`${due ? '清除' : '取消'} ${timer.note}`} onClick={() => remove(timer)}>{due ? '清除' : '取消'}</button>
            </li>
          })}
        </ul> : null}
      </div>
    </section> : null}
    {reminder ? createPortal(<aside className={open ? 'countdown-alert below-panel' : 'countdown-alert'} ref={alert}>
      <Timer size={21} aria-hidden="true" />
      <div role="alert"><strong>倒计时到了</strong><p>{reminder.note}</p>{error && !open ? <small>{error}</small> : null}</div>
      <button type="button" disabled={saving} onClick={() => remove(reminder)}>知道了</button>
    </aside>, document.body) : null}
  </div>
}
