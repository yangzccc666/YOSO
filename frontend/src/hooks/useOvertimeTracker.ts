import { useCallback, useEffect, useRef, useState } from 'react'
import { getOvertime, updateOvertime } from '../api'
import { acceptsOvertimeStatus, createActivityBuffer, isOvertimeActivity } from '../overtime'
import type { OvertimeStatus } from '../overtime'

export function useOvertimeTracker() {
  const [status, setStatus] = useState<OvertimeStatus | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const current = useRef<OvertimeStatus | null>(null)
  const alive = useRef(false)
  const chain = useRef<Promise<unknown>>(Promise.resolve())
  const buffer = useRef<ReturnType<typeof createActivityBuffer> | null>(null)
  const apply = useCallback((next: OvertimeStatus) => {
    if (!alive.current || !acceptsOvertimeStatus(current.current, next)) return
    const previous = current.current
    if (next.pendingCount && (!previous || next.records.some(r => r.status === 'pending' && !previous.records.some(p => p.id === r.id && p.status === 'pending')))) {
      setNotice('有加班记录待确认，请核对实际下班时间。')
    } else if (next.active && previous?.active?.id !== next.active.id) setNotice('已开始记录加班，可在右上角问号中查看和结束。')
    current.current = next; setStatus(next); setError('')
  }, [])
  const enqueue = useCallback(<T,>(fn: () => Promise<T>) => {
    const next = chain.current.then(fn, fn); chain.current = next.catch(() => undefined); return next
  }, [])
  const fail = useCallback((reason: unknown) => { if (alive.current) setError(reason instanceof Error ? reason.message : '加班记录暂未保存，请重试') }, [])
  const refresh = useCallback(() => enqueue(async () => { try { apply(await getOvertime()) } catch (reason) { fail(reason) } }), [apply, enqueue, fail])

  useEffect(() => {
    alive.current = true
    void refresh()
    const report = (at: number) => {
      void enqueue(async () => { try { apply(await updateOvertime('activity', { at: new Date(at).toISOString(), offsetMinutes: -new Date(at).getTimezoneOffset() })) } catch (reason) { fail(reason) } })
    }
    const activityBuffer = createActivityBuffer(report)
    buffer.current = activityBuffer
    let point: string | null = null
    const activity = (event: Event) => {
      if (!current.current?.enabled || !isOvertimeActivity(event, document.visibilityState === 'visible', document.hasFocus())) return
      if (event.type === 'pointermove') {
        const pointer = event as PointerEvent
        const next = `${pointer.clientX}:${pointer.clientY}`
        if (point === next) return
        point = next
      }
      activityBuffer.activity()
    }
    const flush = () => activityBuffer.flush()
    const hidden = () => { if (document.visibilityState === 'hidden') flush() }
    const pagehide = () => {
      const at = activityBuffer.takePending()
      if (at !== null) navigator.sendBeacon('/api/overtime/activity', new Blob([JSON.stringify({ at: new Date(at).toISOString(), offsetMinutes: -new Date(at).getTimezoneOffset() })], { type: 'application/json' }))
    }
    const events = ['pointerdown', 'pointermove', 'keydown', 'wheel']
    events.forEach(event => document.addEventListener(event, activity, { capture: true, passive: true }))
    window.addEventListener('blur', flush); window.addEventListener('pagehide', pagehide)
    document.addEventListener('visibilitychange', hidden)
    const poll = window.setInterval(() => void refresh(), 15000)
    return () => {
      alive.current = false; window.clearInterval(poll)
      events.forEach(event => document.removeEventListener(event, activity, true))
      window.removeEventListener('blur', flush); window.removeEventListener('pagehide', pagehide)
      document.removeEventListener('visibilitychange', hidden)
      activityBuffer.dispose(); buffer.current = null
    }
  }, [apply, enqueue, fail, refresh])
  useEffect(() => {
    if (!notice) return
    const timer = window.setTimeout(() => setNotice(''), 6000)
    return () => window.clearTimeout(timer)
  }, [notice])
  const action = async (name: 'start' | 'finish' | 'confirm' | 'continue', values: Record<string, unknown> = {}) => {
    if (busy) return false
    setBusy(true)
    buffer.current?.flush()
    try { await enqueue(async () => apply(await updateOvertime(name, { offsetMinutes: -new Date().getTimezoneOffset(), ...values }))); return true }
    catch (reason) { fail(reason); return false }
    finally { if (alive.current) setBusy(false) }
  }
  return { status, error, busy, notice, dismissNotice: () => setNotice(''), action, refresh }
}
