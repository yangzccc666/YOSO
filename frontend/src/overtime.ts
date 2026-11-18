export type OvertimeRecord = {
  id: string; status: 'running' | 'pending' | 'confirmed'; startAt: string; endAt: string | null
  lastActivityAt: string; seconds: number; reason: string; canContinue: boolean; offsetMinutes: number
}
export type OvertimeStatus = {
  revision: number; serverTime: string; enabled: boolean; idleMinutes: number; offsetMinutes: number
  active: OvertimeRecord | null; records: OvertimeRecord[]; pendingCount: number
  todaySeconds: number; monthSeconds: number; totalSeconds: number; workingNow: boolean; manualStoppedToday: boolean
}

export function acceptsOvertimeStatus(current: OvertimeStatus | null, next: OvertimeStatus) {
  return !current || next.revision > current.revision || next.revision === current.revision && next.serverTime >= current.serverTime
}

/** Only real foreground input counts. Polls, renders and synthetic events do not. */
export function isOvertimeActivity(event: { isTrusted: boolean }, visible: boolean, focused: boolean) {
  return event.isTrusted && visible && focused
}

export function createActivityBuffer(send: (at: number) => void, clock = {
  now: () => Date.now(), set: (fn: () => void, delay: number) => window.setTimeout(fn, delay), clear: (id: number) => window.clearTimeout(id),
}, delay = 5000) {
  let lastSent = -Infinity, pending: number | null = null, timer: number | undefined
  const flush = () => {
    if (timer !== undefined) clock.clear(timer)
    timer = undefined
    if (pending === null) return
    const at = pending; pending = null; lastSent = clock.now(); send(at)
  }
  return {
    activity() {
      pending = clock.now()
      if (clock.now() - lastSent >= delay) flush()
      else if (timer === undefined) timer = clock.set(flush, delay - (clock.now() - lastSent))
    },
    flush,
    takePending() { const at = pending; pending = null; if (timer !== undefined) clock.clear(timer); timer = undefined; return at },
    dispose() { if (timer !== undefined) clock.clear(timer); timer = undefined; pending = null },
  }
}
