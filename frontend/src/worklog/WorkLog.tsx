import {forwardRef, useEffect, useImperativeHandle, useRef, useState} from 'react'
import {CalendarDays} from 'lucide-react'
import {ProjectDialog} from '../projects/ProjectDialog'
import {api} from '../projects/model'
import {calendarDate, dateKey, WEEKDAYS, type WorkClockPayload} from '../workClock'
import {MasterDetailLayout, MasterItem} from '../workspaces/MasterDetailLayout'
import {WorkDay, type DayHandle} from './WorkDay'
import {ExportPanel, PeriodReport} from './PeriodReport'
import {hours, rangeFor, type Day, type Options} from './model'
import './worklog.css'

type IndexDay = {work_date:string;minutes:number;revision:number}
type Continuation = (saved?:Day)=>void
type DateForm = {source:string;value:string}
export type WorkLogHandle = {leave:(next:()=>void)=>void}

async function changeDate(source:string, method:'PUT'|'DELETE', revision:number, target?:string) {
  const response = await fetch('/api/worklog/dates/'+source+'?revision='+revision,{
    method, headers:{'Content-Type':'application/json'},body:method==='PUT'?JSON.stringify({target,revision}):undefined})
  if (!response.ok) throw new Error((await response.json()).error || '日期修改失败')
}

export const WorkLog = forwardRef<WorkLogHandle>(function WorkLog(_,ref) {
  const today = dateKey(new Date())
  const [date,setDate] = useState(today), [month,setMonth] = useState(today.slice(0,7))
  const [search,setSearch] = useState('')
  const [dates,setDates] = useState<IndexDay[]>([])
  const [options,setOptions] = useState<Options|null>(null)
  const [clock,setClock] = useState<WorkClockPayload|null>(null)
  const [error,setError] = useState(''), [indexError,setIndexError] = useState('')
  const [dirty,setDirty] = useState(false), [saving,setSaving] = useState(false)
  const [busy,setBusy] = useState(false)
  const [refresh,setRefresh] = useState(0), [editorKey,setEditorKey] = useState(0)
  const [form,setForm] = useState<DateForm|null>(null)
  const [pending,setPending] = useState<Continuation|null>(null)
  const [stats,setStats] = useState(false)
  const [range,setRange] = useState<[string,string]>(()=>rangeFor(today,'week'))
  const editor = useRef<DayHandle>(null)
  const refreshLists = ()=>setRefresh(n=>n+1)
  useEffect(()=>{
    let active = true
    Promise.all([api<Options>('/api/worklog/options'),api<WorkClockPayload>('/api/work-clock')]).then(([o,c])=>{
      if(active){setOptions(o);setClock(c)}
    }).catch(e=>{if(active) setError(e.message)})
    return ()=>{active=false}
  },[refresh])
  useEffect(()=>{
    let active=true
    setIndexError('')
    api<{dates:IndexDay[]}>('/api/worklog/dates?month='+month).then(result=>{
      if(!active) return
      setDates(result.dates)
      setDate(current=>current.startsWith(month)?current:today.startsWith(month)?today:result.dates[0]?.work_date || '')
    }).catch(e=>{if(active)setIndexError(e.message)})
    return ()=>{active=false}
  },[month,refresh,today])
  useEffect(()=>{
    if(!dirty) return
    const warn=(e:BeforeUnloadEvent)=>{e.preventDefault();e.returnValue=''}
    window.addEventListener('beforeunload',warn)
    return ()=>window.removeEventListener('beforeunload',warn)
  },[dirty])
  function guard(next:Continuation) {
    if (saving || busy) return
    if (dirty) setPending(()=>next)
    else next()
  }
  useImperativeHandle(ref,()=>({leave:guard}))
  function select(value:string) {
    setError('');setDirty(false);setSearch('');setDate(value);setMonth(value.slice(0,7))
  }
  async function mutate(action:()=>Promise<void>) {
    setBusy(true);setError('')
    try {await action();refreshLists()} catch(e) {setError((e as Error).message);refreshLists()}
    finally {setBusy(false)}
  }
  function submitDate() {
    if (!form?.value) return
    const {source,value} = form
    setForm(null)
    guard(saved=>void mutate(async()=>{
      if (source) {
        const original = dates.find(d=>d.work_date===source)
        if (!original) throw new Error('源日期已变化，请刷新。')
        const revision = saved?.work_date===source ? saved.revision : original.revision
        await changeDate(source,'PUT',revision,value)
      } else await api('/api/worklog/dates',{work_date:value})
      select(value);setEditorKey(n=>n+1)
    }))
  }
  function remove(row:IndexDay) {
    const clear = row.minutes>0
    const verb = clear?'清空全部已保存明细':'移除空白日期'
    if (!window.confirm('确定'+verb+'（'+row.work_date+'）？')) return
    guard(saved=>void mutate(async()=>{
      const revision = saved?.work_date===row.work_date ? saved.revision : row.revision
      if(clear) await api('/api/worklog/day',{work_date:row.work_date,revision,entries:[]})
      else await changeDate(row.work_date,'DELETE',revision)
      if(date===row.work_date) {
        if(!clear && date!==today) setDate('')
        setDirty(false)
      }
      setEditorKey(n=>n+1)
    }))
  }
  const list = dates.filter(d=>d.work_date.startsWith(month))
  for (const virtual of new Set([today,date])) {
    if(virtual.startsWith(month) && !list.some(d=>d.work_date===virtual)) list.push({work_date:virtual,minutes:0,revision:0})
  }
  list.sort((a,b)=>b.work_date.localeCompare(a.work_date))
  const matched=list.filter(r=>(r.work_date+WEEKDAYS[calendarDate(r.work_date).getDay()]).includes(search.trim()))
  const locked=saving || busy
  const showStats=()=>{setRange(rangeFor(date || today,'week'));setStats(true)}
  const feedback=error && <p className="project-error" role="alert">{error} <button onClick={refreshLists}>刷新</button></p>
  return <><MasterDetailLayout title="工时记录" detailKey={date} createLabel="补录日期" searchLabel="搜索日期或星期" search={search}
    onSearch={setSearch} busy={locked} onCreate={()=>{setError('');setForm({source:'',value:date||month+'-01'})}}
    filters={<input aria-label="筛选月份" type="month" value={month} disabled={locked} onChange={e=>{
      const value=e.target.value
      if (value && value!==month) guard(()=>{setMonth(value);setDate('');setDates([]);setSearch('')})
    }}/>}
    list={<>{indexError && <p role="alert">{indexError} <button onClick={refreshLists}>重试</button></p>}
      {matched.map(row=><MasterItem key={row.work_date} name={row.work_date} icon={<CalendarDays size={15}/>}
        disabled={locked} selected={date===row.work_date}
        badge={<b className={'work-badge '+(row.minutes<480?'short':row.minutes===480?'full':'over')}>
          {WEEKDAYS[calendarDate(row.work_date).getDay()]} · {hours(row.minutes)}h</b>}
        select={()=>{if(date!==row.work_date)guard(()=>select(row.work_date))}}
        rename={row.revision?()=>{setError('');setForm({source:row.work_date,value:row.work_date})}:undefined}
        deleteLabel={row.minutes?'清空':'移除'} remove={row.revision?()=>remove(row):undefined}/>)}
      {!matched.length && <p>没有匹配日期，可补录历史日期。</p>}</>}>
    <header className="section-title"><h2>{date || '工时记录'} {date===today && <small>今天</small>}</h2>
      <button onClick={showStats}>工时统计概览</button></header>
    {!form && feedback}
    {date && options ? <WorkDay key={date+':'+editorKey} ref={editor} date={date} options={options}
      setDirty={setDirty} setSaving={setSaving} onSaved={refreshLists}/> : <p>{date?'正在读取工时台账…':'选择或补录日期以填写明细。'}</p>}
  </MasterDetailLayout>
  {form && <ProjectDialog title={form.source?'更改日期':'补录日期'} close={()=>{if(!busy)setForm(null)}}>
    <form className="project-form" onSubmit={e=>{e.preventDefault();submitDate()}}>
      <label>{form.source?'目标日期':'补录日期'}<input type="date" required value={form.value} onChange={e=>setForm({...form,value:e.target.value})}/></label>
      {feedback}<button disabled={busy || !form.value || form.source===form.value}>确定</button>
    </form></ProjectDialog>}
  {stats && options && clock && <ProjectDialog title="工时统计概览" wide close={()=>setStats(false)}>
    <div className="worklog-body"><div className="worklog-filters">
      <button onClick={()=>setRange(rangeFor(date||today,'week'))}>所选周</button><button onClick={()=>setRange(rangeFor(date||today,'month'))}>所选月</button>
      <label>统计开始日期<input type="date" value={range[0]} onChange={e=>setRange([e.target.value,range[1]])}/></label>
      <label>统计结束日期<input type="date" value={range[1]} onChange={e=>setRange([range[0],e.target.value])}/></label></div>
      {range[0] && range[1] && range[1]>=range[0] ? <>
        <PeriodReport start={range[0]} end={range[1]} clock={clock} openDay={value=>{
          setStats(false);if(value!==date) guard(()=>select(value))
        }}/>
        <ExportPanel key={range.join('/')} initialStart={range[0]} initialEnd={range[1]} options={options}/>
      </> : <p>请填写有效的日期范围。</p>}</div></ProjectDialog>}
  {pending && <ProjectDialog title="当天工时尚未保存" close={()=>{if(!saving)setPending(null)}}>
    <div className="project-form"><p>是否保存 {date} 的修改后继续？</p><div className="worklog-buttons">
      <button disabled={saving} className="project-primary" onClick={async()=>{
        const ok=await editor.current?.save()
        setPending(null);if(ok)pending(ok)
      }}>保存并切换</button>
      <button disabled={saving} onClick={()=>{setDirty(false);setEditorKey(n=>n+1);setPending(null);pending()}}>放弃修改</button>
      <button disabled={saving} onClick={()=>setPending(null)}>继续编辑</button></div></div></ProjectDialog>}
  </>
})
