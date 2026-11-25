import { useEffect, useRef, type ReactNode } from 'react'
import { Plus, Search, Pencil, Trash2 } from 'lucide-react'
import './workspace.css'

type LayoutProps = {
  title: string; detailKey:string; createLabel: string; onCreate: () => void; busy?: boolean
  search: string; onSearch: (value: string) => void; searchLabel: string
  filters?: ReactNode; list: ReactNode; children: ReactNode
}

export function MasterDetailLayout(props: LayoutProps) {
  const pane = useRef<HTMLElement>(null)
  useEffect(()=>{pane.current?.scrollTo(0,0)},[props.detailKey])
  return <section className="master-detail" aria-label={props.title}>
    <aside className="master-index">
      <div className="sidebar-heading"><h1>{props.title}</h1>
        <button disabled={props.busy} className="add-group-button" onClick={props.onCreate}><Plus size={15}/>{props.createLabel}</button></div>
      {props.filters}
      <label className="search-box"><Search size={16}/><input aria-label={props.searchLabel} placeholder={props.searchLabel}
        value={props.search} onChange={e=>props.onSearch(e.target.value)}/></label>
      <nav aria-label={props.title+'索引'} className="master-items">{props.list}</nav>
    </aside>
    <main ref={pane} className="master-detail-pane"><div className="master-content">{props.children}</div></main>
  </section>
}

export function MasterItem({name,selected,disabled,icon,badge,select,rename,remove,deleteLabel}: {
  name:string; selected:boolean; disabled?:boolean; icon:ReactNode; badge?:ReactNode
  select:()=>void; rename?:()=>void; remove?:()=>void; deleteLabel?:string
}) {
  return <div className={'master-item'+(selected?' selected':'')}>
    <button className="master-select" disabled={disabled} aria-pressed={selected} aria-label={name} title={name} onClick={select}>
      {icon}<span>{name}</span>{badge}</button>
    <div className="group-actions">{rename && <button disabled={disabled} aria-label={'更改 '+name} title={'更改 '+name} onClick={rename}><Pencil size={14}/></button>}
      {remove && <button disabled={disabled} className="danger-action" aria-label={(deleteLabel||'删除')+' '+name}
        title={(deleteLabel||'删除')+' '+name} onClick={remove}><Trash2 size={14}/></button>}</div>
  </div>
}
