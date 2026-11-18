import { useState } from 'react'
import type { ReactNode } from 'react'
import { icons } from '../icons'

/** Native disclosure keeps inputs and child effects mounted while collapsed. */
export function CollapsibleSection({ heading, children, className = '', count, defaultOpen = true }: {
  heading: ReactNode
  children: ReactNode
  className?: string
  count?: number
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return <section className={className}>
    <details className="collapsible-section" open={open} onToggle={event => setOpen(event.currentTarget.open)}>
      <summary className="collapsible-section-summary">{heading}<span className="collapsible-section-action">{count === undefined ? null : <small>{count} 项</small>}<span>{open ? '收起' : '展开'}</span><icons.ChevronDown size={17} /></span></summary>
      <div className="collapsible-section-body">{children}</div>
    </details>
  </section>
}
