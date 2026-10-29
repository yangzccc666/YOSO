import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { icons } from '../icons'
import { createFrameStepHold } from '../frameStepHold'
import type { FrameDirection } from '../frameStepHold'

type Props = {
  supported: boolean
  busy: boolean
  index: number
  count: number
  onStep: (direction: -1 | 1) => Promise<boolean | void>
  onHoldingChange?: (holding: boolean) => void
}

export function FrameStepControls({ supported, busy, index, count, onStep, onHoldingChange }: Props) {
  const current = useRef({ supported, busy, index, count, onStep, onHoldingChange })
  const [holding, setHolding] = useState<FrameDirection | null>(null)
  const pointer = useRef<{ id: number; button: HTMLButtonElement } | null>(null)
  const controller = useRef<ReturnType<typeof createFrameStepHold> | null>(null)
  if (!controller.current) controller.current = createFrameStepHold(
    () => ({ ...current.current, supported: current.current.supported && !document.hidden && !document.querySelector('[aria-modal="true"], dialog[open]') }),
    (direction) => current.current.onStep(direction),
    (direction) => { setHolding(direction); current.current.onHoldingChange?.(direction !== null) },
  )
  useLayoutEffect(() => {
    current.current = { supported, busy, index, count, onStep, onHoldingChange }
    controller.current?.reconcile()
  }, [supported, busy, index, count, onStep, onHoldingChange])
  useEffect(() => {
    const stop = () => {
      const previous = pointer.current
      pointer.current = null
      controller.current?.stop()
      if (previous?.button.hasPointerCapture(previous.id)) previous.button.releasePointerCapture(previous.id)
    }
    const pointerEnd = (event: PointerEvent) => { if (event.pointerId === pointer.current?.id) stop() }
    const pointerMove = (event: PointerEvent) => {
      if (event.pointerId !== pointer.current?.id) return
      const bounds = pointer.current.button.getBoundingClientRect()
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) stop()
    }
    const visibility = () => { if (document.hidden) stop() }
    window.addEventListener('pointerup', pointerEnd, true)
    window.addEventListener('pointercancel', pointerEnd, true)
    window.addEventListener('pointermove', pointerMove, { passive: true })
    window.addEventListener('blur', stop)
    window.addEventListener('keydown', stop, true)
    document.addEventListener('visibilitychange', visibility)
    return () => {
      stop()
      window.removeEventListener('pointerup', pointerEnd, true)
      window.removeEventListener('pointercancel', pointerEnd, true)
      window.removeEventListener('pointermove', pointerMove)
      window.removeEventListener('blur', stop)
      window.removeEventListener('keydown', stop, true)
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [])
  const buttonProps = (direction: FrameDirection) => ({
    className: 'remote-preview-playback-button frame-step-button',
    disabled: !supported || (direction < 0 ? index <= 0 : index >= count - 1) || (busy && holding !== direction),
    'aria-pressed': holding === direction,
    onPointerDown: (event: React.PointerEvent<HTMLButtonElement>) => {
      if (event.button !== 0 || !event.isPrimary) return
      event.preventDefault()
      if (controller.current?.start(direction)) {
        pointer.current = { id: event.pointerId, button: event.currentTarget }
        event.currentTarget.setPointerCapture(event.pointerId)
      }
    },
    onLostPointerCapture: () => { pointer.current = null; controller.current?.stop() },
    onPointerUp: (event: React.PointerEvent<HTMLButtonElement>) => event.currentTarget.blur(),
    onContextMenu: (event: React.MouseEvent) => event.preventDefault(),
    onClick: (event: React.MouseEvent) => {
      // Pointer presses already stepped on pointerdown; preserve keyboard/assistive clicks.
      if (event.detail === 0 && !busy) void onStep(direction)
    },
  })
  return <>
    <button type="button" {...buttonProps(-1)} aria-label="上一帧" aria-keyshortcuts="," title="上一帧（, / ，），点按一帧、长按连续后退，自动暂停"><icons.ChevronRight size={16} style={{ transform: 'rotate(180deg)' }} /></button>
    <button type="button" {...buttonProps(1)} aria-label="下一帧" aria-keyshortcuts="." title="下一帧（. / 。），点按一帧、长按连续前进，自动暂停"><icons.ChevronRight size={16} /></button>
    {index >= 0 && count > 0 ? <span className="remote-preview-frame-index" title="当前原视频帧号 / 总帧数">{index + 1}/{count} 帧</span> : null}
  </>
}
