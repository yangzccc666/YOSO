import { forwardRef, useImperativeHandle, useEffect, useState } from 'react'
import { api, ApiError, lineage } from '../projects/model'
import { formatCalendarDateRange } from '../workClock'
import { draft, hours, validate, type Day, type Draft, type Options } from './model'

function EntryRow({row, index, options, update, remove, move, count}: {
  row: Draft; index: number; count:number; move:(direction:number)=>void; options: Options; update: (patch: Partial<Draft>) => void; remove: () => void
}) {
  const projects = options.nodes.filter(n => n.kind === 'project')
  const scopes = options.nodes.filter(n => n.parent && lineage(options.nodes,n.id)[0]?.id === row.project_id)
  return <fieldset className="worklog-entry"><legend>工作明细 {index + 1}</legend>
    <label>所属项目<select value={row.publicMode ? 'public' : row.project_id} onChange={event => {
      const value = event.target.value
      update({publicMode:value === 'public',project_id:value === 'public' ? '' : value, project_name:'',scope_id:'',scope_name:'',public_affair:''})
    }}>
      <option value="">请选择项目或公共事务</option><option value="public">其他 / 公共事务</option>
      {row.project_id && !projects.some(n=>n.id === row.project_id) &&
        <option value={row.project_id}>{row.project_name}（已移除）</option>}
      {projects.map(n=><option key={n.id} value={n.id}>{n.name}</option>)}</select></label>
    {row.publicMode ? <label>公共事务名称<input list="worklog-affairs" value={row.public_affair} onChange={e=>update({public_affair:e.target.value})}/></label> :
      <label>产线 / 工位<select disabled={!row.project_id} value={row.scope_id} onChange={e=>update({scope_id:e.target.value})}>
        <option value="">项目整体</option>
        {row.scope_id && !scopes.some(n=>n.id === row.scope_id) && <option value={row.scope_id}>{row.scope_name}（已移除）</option>}
        {scopes.map(n=><option key={n.id} value={n.id}>{lineage(options.nodes,n.id).slice(1).map(v=>v.name).join(' / ')}</option>)}
      </select></label>}
    <label>工作阶段<input list="worklog-categories" value={row.task_category} placeholder="选择或输入工作阶段" onChange={e=>update({task_category:e.target.value})}/></label>
    <label>耗时（小时）<input type="number" min="0.5" max="24" step="0.5" value={row.hours} onChange={e=>update({hours:e.target.value})}/></label>
    <label className="worklog-content">具体工作内容<textarea rows={2} value={row.work_content} onChange={e=>update({work_content:e.target.value})}/></label>
    <label>产出物 / 备注<input value={row.deliverable_note} onChange={e=>update({deliverable_note:e.target.value})}/></label>
    <div className="worklog-buttons"><button disabled={index===0} onClick={()=>move(-1)} aria-label={'上移第 '+(index+1)+' 条'}>上移</button>
      <button disabled={index===count-1} onClick={()=>move(1)} aria-label={'下移第 '+(index+1)+' 条'}>下移</button>
      <button type="button" onClick={remove}>删除明细</button></div>
  </fieldset>
}

export type DayHandle = {save:()=>Promise<Day|null>}
type Props = {
  date: string; options: Options; setDirty:(v:boolean)=>void; setSaving:(v:boolean)=>void; onSaved:()=>void
}
export const WorkDay = forwardRef<DayHandle,Props>(function WorkDay({date,options,setDirty,setSaving,onSaved},ref) {
  const [day,setDay] = useState<Day|null>(null)
  const [rows,setRows] = useState<Draft[]>([])
  const [error,setError] = useState('')
  const [notice,setNotice] = useState('')
  const [saving,setBusy] = useState(false)
  const [reload,setReload] = useState(0)
  const [conflict,setConflict] = useState(false)
  useEffect(()=>{
    let alive = true
    api<Day>('/api/worklog/day?date='+date).then(data=>{
      if (alive) {setDay(data);setRows(data.entries.map(r=>draft(r)));setError('');setConflict(false);setDirty(false)}
    }).catch(e=>{if (alive) setError(e.message)})
    return ()=>{alive = false}
  },[date,reload,setDirty])
  function edit(next: Draft[]) {setRows(next);setDirty(true);setNotice('尚未保存');setError('')}
  async function save() {
    if (!day || saving || conflict) return null
    const message = validate(rows)
    if (message) {setError(message);return null}
    setBusy(true);setSaving(true);setError('')
    try {
      const entries = rows.map(r=>({id:r.id,project_id:r.project_id,scope_id:r.scope_id,public_affair:r.public_affair,
        task_category:r.task_category,work_content:r.work_content,duration_minutes:Number(r.hours)*60,deliverable_note:r.deliverable_note}))
      const data = await api<Day>('/api/worklog/day',{work_date:date,revision:day.revision,entries})
      setDay(data);setRows(data.entries.map(r=>draft(r)));setDirty(false);setNotice('当天工时已保存');onSaved();return data
    } catch(e) {
      setError((e as Error).message);setConflict(e instanceof ApiError && e.status===409);return null
    } finally {setSaving(false);setBusy(false)}
  }
  useImperativeHandle(ref,()=>({save}))
  if (!day) return <p role="status">{error || '正在读取当天记录…'} {error && <button onClick={()=>setReload(n=>n+1)}>重试</button>}</p>
  const minutes = rows.reduce((n,r)=>n+Number(r.hours)*60,0)
  const previous = day.previous
  return <>
    <div className={'worklog-total' + (minutes>480 ? ' overtime' : '')}><strong>已填报 {hours(minutes)} / 8.0 小时</strong>
      <span>{minutes>480 ? '超出 '+hours(minutes-480)+' 小时' : minutes<480 ? '还差 '+hours(480-minutes)+' 小时' : '已满 8 小时'}</span></div>
    <div className="worklog-previous">{previous ? <>
      <span>最近一次有效填报　{formatCalendarDateRange(previous.work_date)} · {previous.entries.length} 条 · {hours(previous.entries.reduce((n,r)=>n+r.duration_minutes,0))} 小时</span>
      <button disabled={saving} onClick={()=>{
        edit([...rows,...previous.entries.map(r=>draft(r,true))])
      }}>复制到当天</button></> : <span>此日期之前暂无有效填报</span>}</div>
    <div className="worklog-buttons"><button disabled={saving} onClick={()=>edit([...rows,draft()])}>＋ 添加工作项</button>
      <button disabled={saving || conflict} className="project-primary" onClick={()=>void save()}>{saving?'正在保存…':'保存全部'}</button>
      {conflict && <button onClick={()=>{if(window.confirm('放弃未保存的修改并重新读取当天工时？')) setReload(n=>n+1)}}>重新载入</button>}
      <span role={error?'alert':'status'} className={error?'project-error':''}>{error || notice}</span></div>
    <datalist id="worklog-affairs">{options.affairs.map(v=><option key={v} value={v}/>)}</datalist>
    <datalist id="worklog-categories">{options.categories.map(v=><option key={v} value={v}/>)}</datalist>
    <fieldset className="worklog-rows" disabled={saving}>{rows.map((r,i)=><EntryRow key={r.key} index={i} row={r} options={options} count={rows.length}
      move={direction=>{
        const next = [...rows]
        next.splice(i+direction,0,...next.splice(i,1))
        edit(next)
      }}
      update={patch=>edit(rows.map(item=>item.key===r.key?{...item,...patch}:item))}
      remove={()=>{if(window.confirm('确定移除第 '+(i+1)+' 条工作明细？保存当天后生效。')) edit(rows.filter(item=>item.key!==r.key))}}/>)}</fieldset>
  </>
})
