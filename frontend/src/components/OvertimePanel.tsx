import { useState } from 'react'
import { confirmStop } from '../confirmActions'
import { formatClockSeconds } from '../workClock'
import { useOvertimeTracker } from '../hooks/useOvertimeTracker'
import type { OvertimeRecord } from '../overtime'

type Tracker = ReturnType<typeof useOvertimeTracker>
function inputTime(value: string) {
  const date = new Date(value)
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth()+1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}
const displayTime = (value: string) => new Date(value).toLocaleString('zh-CN', { hour12: false })

function RecordCard({ record, tracker }: { record: OvertimeRecord; tracker: Tracker }) {
  const [editing, setEditing] = useState(false)
  const [start, setStart] = useState('')
  const [end, setEnd] = useState('')
  const [error, setError] = useState('')
  const pending = record.status === 'pending'
  return <div className={`overtime-record ${pending ? 'pending' : ''}`}>
    <div><strong>{pending ? '待确认' : '已确认'}</strong><span>{formatClockSeconds(record.seconds)}</span></div>
    <small>{displayTime(record.startAt)} → {record.endAt ? displayTime(record.endAt) : '正在计时'}</small>
    {pending ? <p>{record.reason === 'idle' ? '长时间无操作，结束时间已回退到最后操作时间。' : record.reason === 'work-start' ? '已进入正常上班时段，本段加班已暂时结束。' : '计时设置已改变，本段请核对实际结束时间。'}</p> : null}
    {!editing ? <div className="overtime-actions">
      {pending ? <button type="button" disabled={tracker.busy} onClick={() => void tracker.action('confirm', { id: record.id })}>确认已下班</button> : null}
      {pending ? <button type="button" disabled={tracker.busy || !record.canContinue} title={record.canContinue ? '补回刚才的无操作时段，并继续本次加班' : '跨正常班次、超过24小时或不是最近一段，请改用修正时间'} onClick={() => {
        if (record.endAt && Date.now() - Date.parse(record.endAt) > 12*3600000 && !window.confirm('该段距离现在超过12小时，确认这段时间一直在加班吗？')) return
        void tracker.action('continue', { id: record.id })
      }}>刚才一直在加班</button> : null}
      <button type="button" disabled={tracker.busy} onClick={() => { setStart(inputTime(record.startAt)); setEnd(inputTime(record.endAt!)); setEditing(true); setError('') }}>修正时间</button>
    </div> : <form className="overtime-edit" onSubmit={async event => {
      event.preventDefault()
      const begin = new Date(start), finish = new Date(end)
      if (!Number.isFinite(begin.getTime()) || !Number.isFinite(finish.getTime())) { setError('请填写有效时间'); return }
      if (await tracker.action('confirm', { id: record.id, startAt: begin.toISOString(), endAt: finish.toISOString() })) setEditing(false)
    }}>
      <label>开始时间<input aria-label="加班开始时间" type="datetime-local" step="1" required value={start} onChange={event => setStart(event.target.value)} /></label>
      <label>结束时间<input aria-label="加班结束时间" type="datetime-local" step="1" required value={end} onChange={event => setEnd(event.target.value)} /></label>
      {error ? <p role="alert">{error}</p> : null}
      <div className="overtime-actions"><button type="submit" disabled={tracker.busy}>保存并确认</button><button type="button" onClick={() => setEditing(false)}>取消修正</button></div>
    </form>}
  </div>
}

export function OvertimePanel({ tracker, now }: { tracker: Tracker; now: Date }) {
  const status = tracker.status
  if (!status) return tracker.error ? <p className="work-clock-error" role="alert">加班记录：{tracker.error}</p> : null
  const pending = status.records.filter(record => record.status === 'pending')
  const confirmed = status.records.filter(record => record.status === 'confirmed')
  const seconds = status.active ? status.active.seconds + Math.max(0, Math.floor((now.getTime() - Date.parse(status.serverTime))/1000)) : 0
  return <section className="overtime-panel" aria-label="加班记录">
    <div className="overtime-heading"><strong>加班记录</strong><small>{status.enabled ? `${status.idleMinutes}分钟无操作后暂时结束` : '自动记录已关闭'}</small></div>
    {status.active ? <div className="overtime-active"><span>正在加班（估算）</span><strong data-clock="overtime">{formatClockSeconds(seconds, seconds >= 86400)}</strong><small>开始于 {displayTime(status.active.startAt)}</small><button type="button" disabled={tracker.busy} onClick={() => { if (confirmStop('加班计时', '本段将保存为已确认；当天不再自动开始，可手动开始新一段。')) void tracker.action('finish', { id: status.active!.id }) }}>结束加班</button></div>
      : <div className="overtime-actions"><small>{status.manualStoppedToday ? '今天已手动结束，不会因后续点击立即重启。' : status.workingNow ? '正常上班时段不记录加班。' : '非工作时间操作平台时自动开始。'}</small>{status.enabled && !status.workingNow ? <button type="button" disabled={tracker.busy} onClick={() => void tracker.action('start')}>开始新一段加班</button> : null}</div>}
    <div className="overtime-totals"><span>今日已确认<strong>{formatClockSeconds(status.todaySeconds)}</strong></span><span>本月已确认<strong>{formatClockSeconds(status.monthSeconds)}</strong></span></div>
    {pending.length ? <><p className="overtime-pending-note">{pending.length} 段待确认，尚未计入汇总。离开平台仍在工作，可衔接最近一段；跨过正常班次时请修正结束时间。</p><div className="overtime-record-list">{pending.map(record => <RecordCard key={record.id} record={record} tracker={tracker} />)}</div></> : null}
    {confirmed.length ? <details className="overtime-history"><summary>已确认记录（{confirmed.length}段）</summary><div className="overtime-record-list">{confirmed.map(record => <RecordCard key={record.id} record={record} tracker={tracker} />)}</div></details> : null}
    {tracker.error ? <p className="work-clock-error" role="alert">{tracker.error}<button type="button" onClick={() => void tracker.refresh()}>刷新记录</button></p> : null}
    <p className="work-clock-note">每段加班不足1小时不计入汇总，确认后不保留在历史列表；满1小时按整段计入。短时待确认记录仍可修正或继续计时。仅记录平台内的前台操作，不监控其他软件。无操作等待时间不计入待确认记录；后台任务和自动刷新不会延长计时。异常退出后重开仍可确认。记录用于个人估算，不等于考勤认定。</p>
  </section>
}
