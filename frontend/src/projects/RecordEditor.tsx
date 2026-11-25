import {useEffect, useRef, useState} from 'react'
import {ApiError, change, scopeName, type Entry, type Detail, type Catalog} from './model'
import {ProjectDialog} from './ProjectDialog'

type Control = {key: string; label: string; choices?: string[]; multiline?: boolean; suggest?: string[]}
export function RecordEditor({initial, project, detail, catalog, onSaved, close, reload}: {
  initial: Entry; project: string; detail: Detail; catalog: Catalog
  onSaved: (entry: Entry) => void; close: () => void; reload: () => Promise<void>
}) {
  const [draft, setDraft] = useState(initial)
  const [status, setStatus] = useState(initial.id ? '已保存' : '待填写')
  const [error, setError] = useState('')
  const [conflict, setConflict] = useState(false)
  const [composing, setComposing] = useState(false)
  const current = useRef(initial), saved = useRef(initial)
  const busy = useRef(false), timer = useRef<ReturnType<typeof setTimeout>>()
  const ime = useRef(false)
  const dirty = () => current.current !== saved.current
  const assessment = initial.kind === 'assessment'
  const suggestions = (key: string, defaults: string[]) => Array.from(new Set([...defaults,
    ...detail.records.filter(r => r.kind === initial.kind).map(r => r.data[key]).filter(Boolean)]))
  const controls: Control[] = assessment ? [
    {key:'category',label:'评估维度',suggest:suggestions('category',catalog.dimensions)},
    {key:'question',label:'关键问题描述',multiline:true}, {key:'solution',label:'评估结论与解决方案',multiline:true},
    {key:'risk',label:'风险等级',choices:['待评估','高','中','低']}, {key:'status',label:'对齐状态',choices:['待确认','已对齐','方案已落地']},
  ] : [
    {key:'name',label:'路径名称'}, {key:'type',label:'路径类型',suggest:suggestions('type',catalog.types)},
    {key:'environment',label:'存储环境',suggest:suggestions('environment',['本地工作站','远程训练服务器','边缘设备','NAS/对象存储'])},
    {key:'host',label:'主机或存储位置'}, {key:'path',label:'完整路径'}, {key:'note',label:'版本 / 阶段备注',multiline:true},
  ]
  for (const field of detail.fields.filter(f => f.kind === initial.kind)) {
    const choices = field.format === 'select' ? [...new Set([...field.options, ...(draft.data[field.id] ? [draft.data[field.id]] : [])])] : undefined
    controls.push({key:field.id,label:field.name,multiline:field.format === 'multiline',choices})
  }
  function edit(next: Entry) {
    current.current = next
    setDraft(next)
    setStatus('待保存')
    setError('')
  }
  function validation(entry: Entry) {
    if (assessment) return entry.data.category?.trim() ? '' : '请填写评估维度'
    if (!entry.data.name?.trim() || !entry.data.type?.trim()) return '请填写路径名称和类型'
    if ((!saved.current.id || saved.current.data.path) && !entry.data.path?.trim()) return '请填写完整路径'
    return ''
  }
  async function save(): Promise<boolean> {
    clearTimeout(timer.current)
    if (busy.current || ime.current || conflict) return false
    if (!dirty()) return !!saved.current.id
    busy.current = true
    try {
      while (dirty() && !ime.current) {
        const snapshot = current.current
        const message = validation(snapshot)
        if (message) {setStatus(message); return false}
        setStatus('正在保存…')
        const result = await change<{id:string;revision:number}>({action:'saveRecord',project,
          ...snapshot,id:saved.current.id || undefined,revision:saved.current.revision})
        const persisted = {...snapshot,...result}
        saved.current = persisted
        if (current.current === snapshot) current.current = persisted
        onSaved(persisted)
      }
      if (dirty()) return false
      setError(''); setStatus('已保存')
      return true
    } catch (e) {
      setError(String((e as Error).message)); setStatus('未保存')
      if (e instanceof ApiError && e.status === 409) setConflict(true)
      return false
    } finally {busy.current = false}
  }
  useEffect(() => {
    if (!composing && !conflict && dirty()) timer.current = setTimeout(() => {void save()}, 700)
    return () => clearTimeout(timer.current)
    // Saving uses the latest draft and acknowledged revision via refs.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft, composing, conflict])
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {if (dirty()) {e.preventDefault(); e.returnValue = ''}}
    window.addEventListener('beforeunload', warn)
    return () => window.removeEventListener('beforeunload', warn)
  }, [])
  async function leave() {
    if (busy.current) {setStatus('正在保存…'); return}
    if (!dirty() || await save()) {close(); return}
    if (window.confirm('仍有未保存的修改。放弃这些修改并关闭？')) close()
  }
  return <ProjectDialog title={assessment ? '编辑评估记录' : '编辑路径资产'} close={() => void leave()}>
    <div className="project-form" onCompositionStart={() => {ime.current=true;setComposing(true);clearTimeout(timer.current)}}
      onCompositionEnd={() => {ime.current=false;setComposing(false)}}>
      <label>记录归属<select value={draft.node} onChange={e => edit({...draft,node:e.target.value})}>
        {detail.nodes.map(n => <option key={n.id} value={n.id}>{scopeName(detail.nodes,n.id)}</option>)}</select></label>
      {controls.map(c => {
        const value = draft.data[c.key] || ''
        const update = (text:string) => edit({...current.current,data:{...current.current.data,[c.key]:text}})
        return <label key={c.key}>{c.label}
          {c.multiline ? <textarea rows={3} value={value} onChange={e=>update(e.target.value)}/> :
            c.choices ? <select value={value} onChange={e=>update(e.target.value)}>
              {(!value || detail.fields.some(f=>f.id===c.key)) && <option value="">请选择</option>}
              {c.choices.map(v=><option key={v}>{v}</option>)}</select> :
            <input value={value} list={c.suggest ? 'project-list-'+c.key : undefined} onChange={e=>update(e.target.value)}/>}
          {c.suggest && <datalist id={'project-list-'+c.key}>{c.suggest.map(v=><option key={v} value={v}/>)}</datalist>}
        </label>
      })}
    </div>
    <footer><span role="status">{error || status}</span>
      {conflict ? <button onClick={async()=>{if(window.confirm('放弃本次未保存的修改并载入最新记录？')) {
        try {await reload();close()} catch(e) {setError((e as Error).message)}
      }}}>载入最新记录</button> : error && <button onClick={()=>void save()}>重试保存</button>}
      <button onClick={()=>void leave()}>完成</button></footer>
  </ProjectDialog>
}
