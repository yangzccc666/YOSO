import {useEffect, useRef, type ReactNode} from 'react'

export function ProjectDialog({title, close, children, wide = false}: {
  title: string; close: () => void; children: ReactNode; wide?: boolean
}) {
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const dialog = ref.current!
    dialog.showModal()
    return () => dialog.close()
  }, [])
  return <dialog className={'project-dialog' + (wide ? ' project-wide' : '')} ref={ref}
    aria-label={title} onCancel={event => {event.preventDefault(); event.stopPropagation(); close()}}>
    <header><h2>{title}</h2><button type="button" onClick={close} aria-label={'关闭' + title}>关闭</button></header>
    {children}
  </dialog>
}
