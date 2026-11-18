import { useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import { createPortal } from 'react-dom'
import { icons } from '../icons'
import { helpTipPosition } from '../helpTip'

/** A shared, keyboard/touch-friendly tooltip; portals escape scroll containers. */
export function HelpTip({ label = '使用说明', children }: { label?: string; children: ReactNode }) {
  const id = useId()
  const trigger = useRef<HTMLButtonElement>(null)
  const tip = useRef<HTMLDivElement>(null)
  const timer = useRef<number>()
  const [mode, setMode] = useState<'closed' | 'hover' | 'pinned'>('closed')
  const [position, setPosition] = useState<ReturnType<typeof helpTipPosition> | null>(null)
  const open = mode !== 'closed'
  const cancelClose = () => window.clearTimeout(timer.current)
  const reveal = () => { cancelClose(); setMode(current => current === 'closed' ? 'hover' : current) }
  const delayClose = () => {
    cancelClose()
    timer.current = window.setTimeout(() => setMode(current => current === 'pinned' ? current : 'closed'), 150)
  }
  useEffect(() => () => window.clearTimeout(timer.current), [])
  useLayoutEffect(() => {
    if (!open) { setPosition(null); return }
    const update = () => {
      if (!trigger.current || !tip.current) return
      const anchor = trigger.current.getBoundingClientRect()
      if (anchor.bottom < 0 || anchor.top > window.innerHeight) { setMode('closed'); return }
      const next = helpTipPosition(anchor, { width: tip.current.offsetWidth, height: tip.current.scrollHeight }, { width: window.innerWidth, height: window.innerHeight })
      setPosition(current => current?.left === next.left && current.top === next.top && current.maxHeight === next.maxHeight ? current : next)
    }
    update()
    window.addEventListener('resize', update)
    window.addEventListener('scroll', update, true)
    return () => { window.removeEventListener('resize', update); window.removeEventListener('scroll', update, true) }
  }, [open, children])
  useEffect(() => {
    if (!open) return
    const otherTip = (event: Event) => { if ((event as CustomEvent<string>).detail !== id) setMode('closed') }
    document.dispatchEvent(new CustomEvent('platform-help-open', { detail: id }))
    document.addEventListener('platform-help-open', otherTip)
    const dismiss = (event: PointerEvent) => {
      if (!trigger.current?.contains(event.target as Node) && !tip.current?.contains(event.target as Node)) setMode('closed')
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setMode('closed') }
    }
    document.addEventListener('pointerdown', dismiss)
    document.addEventListener('keydown', escape, true)
    return () => { document.removeEventListener('platform-help-open', otherTip); document.removeEventListener('pointerdown', dismiss); document.removeEventListener('keydown', escape, true) }
  }, [open, id])
  return <span className="help-tip">
    <button ref={trigger} type="button" className="help-tip-trigger" aria-label={`查看${label}`} aria-expanded={open} aria-describedby={open ? id : undefined}
      onPointerEnter={event => { if (event.pointerType !== 'touch') reveal() }} onPointerLeave={delayClose} onFocus={reveal}
      onBlur={event => { if (!tip.current?.contains(event.relatedTarget as Node)) setMode('closed') }}
      onClick={event => { event.preventDefault(); event.stopPropagation(); cancelClose(); setMode(current => current === 'pinned' ? 'closed' : 'pinned') }}><icons.CircleHelp size={16} aria-hidden="true" /></button>
    {open ? createPortal(<div ref={tip} id={id} role="tooltip" tabIndex={-1} className="help-tip-content" onPointerEnter={cancelClose} onPointerLeave={delayClose} onBlur={event => { if (!trigger.current?.contains(event.relatedTarget as Node)) setMode('closed') }} onClick={event => event.stopPropagation()}
      style={{ ...position, visibility: position ? 'visible' : 'hidden' }}><strong className="help-tip-title">{label}</strong><div>{children}</div></div>, document.body) : null}
  </span>
}
