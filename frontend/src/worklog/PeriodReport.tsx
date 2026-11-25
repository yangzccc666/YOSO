import {useEffect, useState} from 'react'
import {api} from '../projects/model'
import {calendarDate, dateKey, isWorkingDay, WEEKDAYS, type WorkClockPayload} from '../workClock'
import {datesBetween, hours, type Group, type Options, type Report} from './model'

function Distribution({title, groups, total}: {title:string;groups:Group[];total:number}) {
  return <section className="worklog-breakdown"><h3>{title}</h3>
    {groups.length === 0 && <p>暂无已保存工时</p>}
    {groups.map(g=>{
      const share = total ? (100*g.minutes/total) : 0
      return <div key={g.id}><span>{g.name}</span><strong>{hours(g.minutes)} h · {share.toFixed(1)}%</strong>
        <progress value={g.minutes} max={total || 1} aria-label={g.name+'工时占比'}/></div>
    })}</section>
}

export function PeriodReport({start,end,clock,openDay}: {
  start:string;end:string;clock:WorkClockPayload;openDay:(date:string)=>void
}) {
  const [report,setReport] = useState<Report|null>(null)
  const [error,setError] = useState('')
  const [retry,setRetry] = useState(0)
  const [page,setPage] = useState(0)
  useEffect(()=>{
    let active = true
    setReport(null);setError('');setPage(0)
    api<Report>('/api/worklog/report?'+new URLSearchParams({start,end})).then(data=>{
      if(active) setReport(data)
    }).catch(e=>{if(active) setError(e.message)})
    return ()=>{active=false}
  },[start,end,retry])
  if(!report) return <p role="status">{error || '正在统计…'} {error && <button onClick={()=>setRetry(n=>n+1)}>重试</button>}</p>
  const dates = datesBetween(start,end)
  const today = dateKey(new Date())
  const missing = (d:string) => d<today && !report.days[d] && isWorkingDay(calendarDate(d),clock.settings,clock.calendar)
  return <>
    <h3>{start} 至 {end}</h3>
    <div className="worklog-metrics">
      <div>总工时<strong>{hours(report.total_minutes)} 小时</strong></div>
      <div>日均工时<strong>{hours(report.total_minutes/(report.filled_days||1))} 小时</strong><small>按已填 {report.filled_days} 天计算</small></div>
      <div>漏填工作日<strong>{dates.filter(missing).length} 天</strong></div>
    </div>
    {dates.length>31 && <div className="worklog-buttons">
      <button disabled={!page} onClick={()=>setPage(n=>n-1)}>上一页日期</button>
      <span>{page+1} / {Math.ceil(dates.length/31)}</span>
      <button disabled={(page+1)*31>=dates.length} onClick={()=>setPage(n=>n+1)}>下一页日期</button></div>}
    <div className="worklog-calendar">{dates.slice(page*31,page*31+31).map(d=>{
      const minutes = report.days[d] || 0
      const label = minutes ? hours(minutes)+' 小时'+(minutes<480?' · 未满':minutes>480?' · 加班':'') :
        missing(d)?'漏填':isWorkingDay(calendarDate(d),clock.settings,clock.calendar)?'待填报':'休息日'
      return <button key={d} className={missing(d)?'missing':''} onClick={()=>openDay(d)}>
        <small>{d.slice(5)} {WEEKDAYS[calendarDate(d).getDay()]}</small><strong>{label}</strong></button>
    })}</div>
    <div className="worklog-groups"><Distribution title="按项目 / 公共事务" groups={report.projects} total={report.total_minutes}/>
      <Distribution title="按任务类型" groups={report.tasks} total={report.total_minutes}/></div>
  </>
}

export function ExportPanel({initialStart, initialEnd, options}: {initialStart:string;initialEnd:string;options:Options}) {
  const [start,setStart] = useState(initialStart), [end,setEnd] = useState(initialEnd)
  const [owner,setOwner] = useState(''), [error,setError] = useState(''), [busy,setBusy] = useState(false)
  const projects = new Map(options.report_projects.map(p=>[p.id,p.name]))
  options.nodes.filter(n=>n.kind==='project').forEach(n=>projects.set(n.id,n.name))
  async function download(format:'xlsx'|'csv') {
    if(!start || !end || start>end) {setError('请核对导出日期范围。');return}
    setBusy(true);setError('')
    try {
      const res = await fetch('/api/worklog/export?'+new URLSearchParams({start,end,owner,format}))
      if(!res.ok) throw new Error((await res.json()).error || '导出失败')
      const url = URL.createObjectURL(await res.blob())
      const link = document.createElement('a')
      link.href=url;link.download='worklog-'+start+'-'+end+'.'+format
      document.body.append(link);link.click();link.remove()
      window.setTimeout(()=>URL.revokeObjectURL(url),60000)
    } catch(e) {setError((e as Error).message)} finally {setBusy(false)}
  }
  return <details className="worklog-export"><summary>导出工时报表</summary>
    <div className="worklog-filters">
      <label>起始日期<input type="date" value={start} onChange={e=>setStart(e.target.value)}/></label>
      <label>截止日期<input type="date" value={end} onChange={e=>setEnd(e.target.value)}/></label>
      <label>项目 / 公共事务<select aria-label="项目 / 公共事务" value={owner} onChange={e=>setOwner(e.target.value)}><option value="">全部</option>
        <optgroup label="项目">{[...projects].map(([id,name])=><option key={id} value={'project:'+id}>{name}</option>)}</optgroup>
        <optgroup label="公共事务">{options.affairs.map(a=><option key={a} value={'affair:'+a}>{a}</option>)}</optgroup></select></label>
      <button disabled={busy} onClick={()=>void download('xlsx')}>导出 Excel</button><button disabled={busy} onClick={()=>void download('csv')}>导出 CSV</button>
    </div>{error && <p role="alert" className="project-error">{error}</p>}
  </details>
}
