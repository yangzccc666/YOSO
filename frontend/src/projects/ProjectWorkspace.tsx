import {useEffect, useState} from 'react'
import {Folder, GripVertical, Plus} from 'lucide-react'
import {api, change, lineage, levels, scopeName, tabs, type Catalog, type Detail, type Entry, type Field, type Kind, type Node} from './model'
import {ProjectDialog} from './ProjectDialog'
import {RecordEditor} from './RecordEditor'
import {MasterDetailLayout, MasterItem} from '../workspaces/MasterDetailLayout'
import './projects.css'

type NodeForm = {action:'createProject'|'createNode'|'renameNode'; title:string; name:string; owner?:string}
const blankField = (): Field => ({id:'',name:'',kind:'assessment',format:'text',options:[]})

export function ProjectWorkspace() {
  const [catalog,setCatalog] = useState<Catalog>({projects:[],dimensions:[],types:[]})
  const [project,setProject] = useState('')
  const [detail,setDetail] = useState<Detail|null>(null)
  const [scope,setScope] = useState('')
  const [kind,setKind] = useState<Kind>('assessment')
  const [shared,setShared] = useState(false)
  const [editing,setEditing] = useState<Entry|null>(null)
  const [nodeForm,setNodeForm] = useState<NodeForm|null>(null)
  const [fieldsOpen,setFieldsOpen] = useState(false)
  const [field,setField] = useState<Field>(blankField)
  const [optionsText,setOptionsText] = useState('')
  const [busy,setBusy] = useState(false)
  const [error,setError] = useState('')
  const [notice,setNotice] = useState('')
  const [search,setSearch] = useState('')
  const [ready,setReady] = useState(false)
  const loadCatalog = async () => {setCatalog(await api<Catalog>('/api/projects'));setReady(true)}
  const load = async (id=project) => setDetail(await api<Detail>('/api/projects/'+id))
  useEffect(()=>{
    let active = true
    async function initial() {
      const data = await api<Catalog>('/api/projects')
      if (!active) return
      setCatalog(data);setReady(true)
      if (data.projects.length) {
        const id = data.projects[0].id
        const details = await api<Detail>('/api/projects/'+id)
        if (active) {setDetail(details);setProject(id);setScope(id)}
      }
    }
    setBusy(true)
    initial().catch(e=>{if(active)setError(e.message)}).finally(()=>{if(active)setBusy(false)})
    return ()=>{active=false}
  },[])
  useEffect(()=>{if (!notice) return; const timer=setTimeout(()=>setNotice(''),2400);return ()=>clearTimeout(timer)},[notice])
  async function run(action:()=>Promise<void>) {
    if (busy) return
    setBusy(true);setError('')
    try {await action()} catch(e) {setError((e as Error).message)} finally {setBusy(false)}
  }
  async function openProject(id:string) {
    await load(id);setProject(id);setScope(id);setShared(false);setKind('assessment')
  }
  const nodes = detail?.nodes || []
  const selected = nodes.find(n=>n.id===scope)
  const ancestors = lineage(nodes,scope).map(n=>n.id)
  const visible = detail?.records.filter(r=>r.kind===kind && (lineage(nodes,r.node).some(n=>n.id===scope) || (shared && ancestors.includes(r.node)))) || []
  function newEntry() {
    const data: Record<string,string> = kind==='assessment' ? {category:'',question:'',solution:'',risk:'待评估',status:'待确认'}
      : {name:'',type:'',environment:'',host:'',path:'',note:''}
    setEditing({id:'',node:scope,revision:0,kind,data})
  }
  function onSaved(entry:Entry) {
    void loadCatalog().catch(e=>setError(e.message))
    setDetail(d=>d && ({...d,records:d.records.some(r=>r.id===entry.id)
      ? d.records.map(r=>r.id===entry.id ? entry : r) : [...d.records,entry]}))
  }
  async function saveNode() {
    if (!nodeForm) return
    const {id} = await change({action:nodeForm.action,project:nodeForm.owner || project,node:nodeForm.owner || scope,name:nodeForm.name})
    if (nodeForm.action==='createProject') {await loadCatalog();await openProject(id);setSearch('')}
    else if (nodeForm.owner) {await loadCatalog();if(project===nodeForm.owner) await load()}
    else {await load();if(nodeForm.action==='createNode') setScope(id);await loadCatalog()}
    setNodeForm(null)
  }
  function deleteNode(owner = project, target = scope) {
    void run(async()=>{
      const data = owner === project && detail ? detail : await api<Detail>('/api/projects/'+owner)
      const affected = data.nodes.filter(n=>lineage(data.nodes,n.id).some(v=>v.id===target))
      const count = data.records.filter(r=>affected.some(n=>n.id===r.node)).length
      const node = data.nodes.find(n=>n.id===target)
      if (!node || !window.confirm(`确定删除“${node.name}”？涉及 ${affected.length-1} 个下级节点和 ${count} 条台账记录。历史工时保留。`)) return
      await change({action:'deleteNode',project:owner,node:target})
      if (owner===target) {
        const updated = await api<Catalog>('/api/projects')
        setCatalog(updated)
        if (owner===project) {
          setProject('');setScope('');setDetail(null)
          if(updated.projects.length) await openProject(updated.projects[0].id)
        }
      } else {await load();await loadCatalog();setScope(node.parent || project)}
    })
  }
  function removeRecord(r:Entry) {
    if (!window.confirm('确定删除“'+(r.data.name || r.data.category)+'”这条记录？')) return
    void run(async()=>{await change({action:'deleteRecord',project,id:r.id,revision:r.revision});await load();await loadCatalog()})
  }
  async function copy(path:string) {
    try {await navigator.clipboard.writeText(path)}
    catch {
      const input=document.createElement('textarea')
      input.value=path
      document.body.appendChild(input)
      input.select()
      const ok=document.execCommand('copy')
      input.remove()
      if(!ok) {setError('复制失败，请在编辑窗口中选中完整路径复制。');return}
    }
    setNotice('路径已复制')
  }
  function selectField(f:Field) {setField(f);setOptionsText(f.options.join('\n'));setError('')}
  const feedback = error && <p role="alert" className="project-error">{error}</p>
  const sortMode = catalog.sortMode || 'edited'
  const manual = sortMode === 'manual'
  function sortProjects(mode:string, move?:{source:string; target:string}) {
    void run(async()=>{
      await change({action:'sortProjects',mode,...move})
      await loadCatalog()
    })
  }
  const matches = catalog.projects.filter(n=>n.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase()))
  return <><MasterDetailLayout title="项目管理" detailKey={scope+kind} createLabel="新建项目" busy={busy} searchLabel="搜索项目" search={search} onSearch={setSearch}
    onCreate={()=>{setError('');setNodeForm({action:'createProject',title:'新建项目',name:''})}}
    filters={<label className="project-sort">排序方式 <select value={sortMode} disabled={busy} onChange={e=>sortProjects(e.target.value)}>
      <option value="edited">最近编辑优先</option><option value="created">最近创建优先</option><option value="manual">手动拖动排序</option>
    </select></label>}
    list={<>{matches.map(n=><div key={n.id} className={manual?'project-sort-row':undefined}
      onDragOver={e=>{if(manual && e.dataTransfer.types.includes('application/x-project-id')) e.preventDefault()}}
      onDrop={e=>{const source=e.dataTransfer.getData('application/x-project-id');if(manual && source && source!==n.id){e.preventDefault();sortProjects('manual',{source,target:n.id})}}}>
      {manual && <span className="project-grip" title="拖动调整项目顺序" draggable={!busy} onDragStart={e=>{e.dataTransfer.effectAllowed='move';e.dataTransfer.setData('application/x-project-id',n.id)}}><GripVertical size={16}/></span>}
      <MasterItem name={n.name} selected={n.id===project} icon={<Folder size={16}/>} disabled={busy}
      select={()=>void run(()=>openProject(n.id))} rename={()=>{setError('');setNodeForm({action:'renameNode',title:'重命名项目',name:n.name,owner:n.id})}}
      remove={()=>deleteNode(n.id,n.id)}/></div>)}{!matches.length && <p>{ready?'暂无匹配项目':'正在读取项目…'}</p>}</>}>
    {!nodeForm && !fieldsOpen && feedback}
    {!ready && error && <button onClick={()=>void run(loadCatalog)}>重新加载</button>}
    {!detail || !selected ? <p>{busy?'正在加载…':'请选择或新建项目。'}</p> : <>
      <header className="project-scope"><h2>{nodes.find(n=>n.id===project)?.name}</h2>
        <label>工作范围　<select value={scope} aria-label="工作范围" onChange={e=>setScope(e.target.value)}>
          <option value={project}>整个项目</option>
          {nodes.filter(n=>n.parent).sort((a,b)=>scopeName(nodes,a.id).localeCompare(scopeName(nodes,b.id),'zh-CN')).map(n=>
            <option key={n.id} value={n.id}>{lineage(nodes,n.id).slice(1).map(v=>v.name).join(' / ')}</option>)}</select></label>
        <div className="project-tools">{selected.kind!=='station' && <button onClick={()=>{setError('');setNodeForm({action:'createNode',title:selected.kind==='project'?'新增产线':'新增工位',name:''})}}>
          <Plus size={14}/>{selected.kind==='project'?'新增产线':'新增工位'}</button>}
          <button onClick={()=>{setError('');setNodeForm({action:'renameNode',title:'重命名'+levels[selected.kind],name:selected.name})}}>重命名</button>
          <button disabled={busy} onClick={()=>deleteNode()}>删除{levels[selected.kind]}</button></div></header>
        <div className="project-tabs">{(Object.keys(tabs) as Kind[]).map(k=><button aria-pressed={kind===k} key={k} onClick={()=>{setKind(k);setNotice('')}}>{tabs[k]}</button>)}</div>
        <div className="project-tools"><button className="project-primary" onClick={newEntry}><Plus size={15}/>{kind==='assessment'?'新增评估':'新增路径'}</button>
          <button onClick={()=>{selectField({...blankField(),kind});setFieldsOpen(true)}}>自定义字段</button>
          {selected.parent && <label><input type="checkbox" checked={shared} onChange={e=>setShared(e.target.checked)}/>显示上级共享记录</label>}
          <small role="status">{notice || visible.length+' 条记录'}</small></div>
        <div className="project-entries">{visible.map(r=><article key={r.id}>
          <div><h4>{r.data.name || r.data.category}</h4><small>{scopeName(nodes,r.node)}</small></div>
          {kind==='assessment' ? <><p>{r.data.question}</p>{r.data.solution && <p>{r.data.solution}</p>}<small>风险 {r.data.risk} · {r.data.status}</small></> :
            <><p className="project-path">{r.data.path || '待填写路径'}</p><small>{[r.data.type,r.data.environment,r.data.host,r.data.note].filter(Boolean).join(' · ')}</small></>}
          <div className="project-actions">{kind==='asset' && <button disabled={!r.data.path} onClick={()=>void copy(r.data.path)}>复制路径</button>}
            <button disabled={busy} onClick={()=>setEditing(r)}>编辑</button><button disabled={busy} onClick={()=>removeRecord(r)}>删除</button></div>
        </article>)}</div>
        {!visible.length && <p>当前范围暂无{kind==='assessment'?'评估':'路径'}记录。</p>}
      </>}
    </MasterDetailLayout>
    {nodeForm && <ProjectDialog title={nodeForm.title} close={()=>{if(!busy){setNodeForm(null);setError('')}}}>
      <form className="project-form" onSubmit={e=>{e.preventDefault();void run(saveNode)}}>
        <label>名称<input autoFocus required value={nodeForm.name} onChange={e=>setNodeForm({...nodeForm,name:e.target.value})}/></label>
        {feedback}<button className="project-primary" disabled={busy || !nodeForm.name.trim()}>保存</button>
      </form></ProjectDialog>}
    {editing && detail && <RecordEditor initial={editing} project={project} detail={detail} catalog={catalog}
      onSaved={onSaved} reload={()=>load()} close={()=>setEditing(null)}/>}
    {fieldsOpen && detail && <ProjectDialog title={'自定义字段 · '+tabs[kind]} close={()=>{if(!busy){setFieldsOpen(false);setError('')}}}>
      <div className="project-form"><label>选择字段<select value={field.id} onChange={e=>selectField(detail.fields.find(f=>f.id===e.target.value) || {...blankField(),kind})}>
        <option value="">新增自定义字段</option>{detail.fields.filter(f=>f.kind===kind).map(f=><option key={f.id} value={f.id}>{f.name}</option>)}</select></label>
        <label>字段名称<input value={field.name} onChange={e=>setField({...field,name:e.target.value})}/></label>
        <label>字段类型<select disabled={!!field.id} value={field.format} onChange={e=>setField({...field,format:e.target.value as Field['format']})}>
          <option value="text">单行文本</option><option value="multiline">多行文本</option><option value="select">下拉选项</option></select></label>
        {field.format==='select' && <label>选项（每行一个）<textarea value={optionsText} onChange={e=>setOptionsText(e.target.value)}/></label>}
        {feedback}<div className="project-tools"><button className="project-primary" disabled={busy || !field.name.trim()} onClick={()=>void run(async()=>{
          await change({action:'saveField',project,...field,options:optionsText.split('\n').map(v=>v.trim()).filter(Boolean)})
          await load();await loadCatalog();setFieldsOpen(false)
        })}>保存字段</button>
        {field.id && <button disabled={busy} onClick={()=>{if(window.confirm('删除字段及本项目中已填写的该字段值？')) void run(async()=>{
          await change({action:'deleteField',project,id:field.id});await load();await loadCatalog();selectField({...blankField(),kind})
        })}}>删除字段</button>}</div>
      </div></ProjectDialog>}
  </>
}
