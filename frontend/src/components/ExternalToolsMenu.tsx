import { useEffect, useRef, useState } from 'react'
import { icons } from '../icons'

type Tool = { id: string; name: string; description: string; busy?: boolean; onLaunch: () => void }

export function ExternalToolsMenu({ tools }: { tools: Tool[] }) {
  const [open, setOpen] = useState(false)
  const container = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!open) return
    const outside = (event: PointerEvent) => {
      if (!container.current?.contains(event.target as Node)) setOpen(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { setOpen(false); trigger.current?.focus() }
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    return () => {
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
    }
  }, [open])
  return <div className="external-tools" ref={container} onBlur={(event) => {
    if (!event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false)
  }}>
    <button ref={trigger} className="external-tools-trigger" aria-expanded={open} aria-controls="external-tools-list" onClick={() => setOpen(!open)}>
      <icons.Box size={17} /><span>外部工具</span><icons.ChevronDown size={14} />
    </button>
    {open ? <div className="external-tools-list" id="external-tools-list" aria-label="外部工具">
      {tools.map((tool) => <button key={tool.id} disabled={tool.busy} onClick={() => { setOpen(false); trigger.current?.focus(); tool.onLaunch() }}>
        <icons.Play size={17} /><span><strong>{tool.busy ? '正在启动…' : tool.name}</strong><small>{tool.description}</small></span>
      </button>)}
    </div> : null}
  </div>
}
