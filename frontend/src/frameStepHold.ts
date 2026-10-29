export type FrameDirection = -1 | 1
export type FrameStepState = { supported: boolean; busy: boolean; index: number; count: number }
type Clock = { now: () => number; set: (callback: () => void, delay: number) => unknown; clear: (id: unknown) => void }

/** One in-flight request, no queue; late replies cannot restart a released hold. */
export function createFrameStepHold(
  state: () => FrameStepState,
  step: (direction: FrameDirection) => Promise<boolean | void>,
  changed: (direction: FrameDirection | null) => void,
  clock: Clock = { now: () => performance.now(), set: (callback, delay) => window.setTimeout(callback, delay), clear: (id) => window.clearTimeout(id as number) },
) {
  let direction: FrameDirection | null = null
  let generation = 0
  let inFlight = false
  let timer: unknown
  const allowed = (value: FrameStepState, target: FrameDirection) => value.supported && value.index >= 0 && value.count > 0
    && (target < 0 ? value.index > 0 : value.index < value.count - 1)
  const stop = () => {
    generation++
    if (timer !== undefined) clock.clear(timer)
    timer = undefined
    if (direction !== null) { direction = null; changed(null) }
  }
  const schedule = (delay: number, first: boolean) => {
    timer = clock.set(() => { timer = undefined; void advance(first) }, delay)
  }
  const advance = async (first: boolean) => {
    if (direction === null) return
    const target = direction
    const value = state()
    if (!allowed(value, target)) { stop(); return }
    if (inFlight || value.busy) { schedule(60, first); return }
    const token = generation
    const started = clock.now()
    inFlight = true
    let accepted: boolean | void = false
    try { accepted = await step(target) }
    catch { /* The player reports the request error; stop instead of retrying forever. */ }
    finally { inFlight = false }
    if (generation !== token || direction === null) return
    if (accepted === false) { stop(); return }
    // Give React a commit between replies, even when inference took longer than the interval.
    schedule(Math.max(30, (first ? 350 : 100) - (clock.now() - started)), false)
  }
  return {
    start(target: FrameDirection): boolean {
      stop()
      const value = state()
      if (inFlight || value.busy || !allowed(value, target)) return false
      direction = target
      changed(target)
      void advance(true)
      return true
    },
    stop,
    reconcile() { if (direction !== null && !allowed(state(), direction)) stop() },
  }
}
