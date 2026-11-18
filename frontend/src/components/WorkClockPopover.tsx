import { useEffect, useMemo, useRef, useState } from 'react'
import { confirmClose } from '../confirmActions'
import { getWorkClock, saveWorkClock } from '../api'
import { icons } from '../icons'
import { calendarDate, dateKey, daysBetween, formatCalendarDateRange, formatClockSeconds, upcomingFestivals, upcomingHolidays, WEEKDAYS, weeklyRestState, workShiftState } from '../workClock'
import type { WorkClockPayload, WorkClockSettings } from '../workClock'
import { useOvertimeTracker } from '../hooks/useOvertimeTracker'
import { OvertimePanel } from './OvertimePanel'

export function WorkClockPopover() {
  const overtime = useOvertimeTracker()
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState(false)
  const [payload, setPayload] = useState<WorkClockPayload | null>(null)
  const [draft, setDraft] = useState<WorkClockSettings | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)
  const [now, setNow] = useState(() => new Date())
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const closeTimer = useRef<ReturnType<typeof setTimeout> | undefined>()

  useEffect(() => {
    let cancelled = false
    getWorkClock().then(value => { if (!cancelled) { setPayload(value); setDraft(value.settings) } })
      .catch(reason => { if (!cancelled) setError(reason instanceof Error ? reason.message : '时间设置加载失败') })
    return () => { cancelled = true; clearTimeout(closeTimer.current) }
  }, [])

  useEffect(() => {
    if (!open) { setEditing(false); return }
    const tick = () => setNow(new Date())
    tick()
    const timer = setInterval(tick, 1000)
    const outside = (event: PointerEvent) => { if (!root.current?.contains(event.target as Node)) setOpen(false) }
    const escape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return
      trigger.current?.focus()
      setOpen(false)
    }
    document.addEventListener('pointerdown', outside)
    document.addEventListener('keydown', escape)
    document.addEventListener('visibilitychange', tick)
    return () => {
      clearInterval(timer)
      document.removeEventListener('pointerdown', outside)
      document.removeEventListener('keydown', escape)
      document.removeEventListener('visibilitychange', tick)
    }
  }, [open])

  const today = dateKey(now)
  const festivals = useMemo(() => payload ? upcomingFestivals(calendarDate(today), payload.calendar) : [], [payload, today])
  const shift = payload ? workShiftState(now, payload.settings, payload.calendar) : null
  const weekend = payload ? weeklyRestState(now, payload.settings, payload.calendar) : null
  const holiday = payload ? upcomingHolidays(now, payload.calendar)[0] : undefined
  const holidayDayCount = holiday ? daysBetween(calendarDate(holiday.start), calendarDate(holiday.end)) + 1 : 0
  const mode = payload?.settings.displayMode
  const show = () => { clearTimeout(closeTimer.current); setOpen(true) }
  const leave = () => {
    if (editing || root.current?.contains(document.activeElement)) return
    closeTimer.current = setTimeout(() => setOpen(false), 180)
  }
  const update = <K extends keyof WorkClockSettings>(key: K, value: WorkClockSettings[K]) => {
    setDraft(current => current ? { ...current, [key]: value } : current)
    setError('')
    setSaved(false)
  }
  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!draft || saving) return
    setSaving(true)
    setError('')
    try {
      const value = await saveWorkClock(draft)
      setPayload(value)
      setDraft(value.settings)
      setEditing(false)
      setSaved(true)
      setNow(new Date())
      void overtime.refresh()
    } catch (reason) { setError(reason instanceof Error ? reason.message : '设置保存失败') }
    finally { setSaving(false) }
  }

  return <div className="work-clock" ref={root} onMouseEnter={show} onMouseLeave={leave} onFocus={show} onBlur={event => { if (event.relatedTarget && !event.currentTarget.contains(event.relatedTarget as Node)) setOpen(false) }}>
    <button ref={trigger} type="button" aria-label="工作时间与假期" aria-haspopup="dialog" aria-expanded={open} aria-controls={open ? 'work-clock-panel' : undefined} onClick={show}><icons.CircleHelp size={18} />{overtime.status?.pendingCount ? <span className="overtime-badge" title={`${overtime.status.pendingCount} 段加班待确认`}>{overtime.status.pendingCount}</span> : overtime.status?.active ? <span className="overtime-badge active" title="正在记录加班">●</span> : overtime.error ? <span className="overtime-badge" title="加班记录暂未保存，请查看详情">!</span> : null}</button>
    {overtime.notice && !open ? <div className="overtime-notice" role="status"><span>{overtime.notice}</span><button type="button" onClick={() => { overtime.dismissNotice(); show() }}>查看</button></div> : null}
    {open ? <section id="work-clock-panel" className="work-clock-popover" role="dialog" aria-label="工作时间与假期" aria-modal="false">
      <header><div><strong>{editing ? '计时设置' : '工作与假期'}</strong><small>{now.toLocaleDateString('zh-CN', { month: 'long', day: 'numeric' })} {WEEKDAYS[now.getDay()]} · 本机时间</small></div><div>
        {payload ? <button type="button" className="work-clock-settings" aria-label={editing ? '返回计时' : '计时设置'} onClick={() => { setEditing(value => !value); setDraft(payload.settings); setError(''); setSaved(false) }}>{editing ? '返回' : <icons.Settings size={16} />}</button> : null}
        <button type="button" aria-label="关闭时间面板" onClick={() => { if (confirmClose('工作时间与假期', editing ? '未保存的计时设置将不会保留。' : '计时仍会继续，不影响其他任务。')) { trigger.current?.focus(); setOpen(false) } }}><icons.X size={16} /></button>
      </div></header>
      <div className="work-clock-body">
        {!payload ? <p className="work-clock-note">{error || '正在加载时间设置…'}{error ? <button type="button" onClick={() => { setError(''); void getWorkClock().then(value => { setPayload(value); setDraft(value.settings) }).catch(reason => setError(String(reason))) }}>重试</button> : null}</p> : null}
        {payload && editing && draft ? <form className="work-clock-form" onSubmit={event => void submit(event)}>
          <div className="work-clock-times"><label>上班时间<input type="time" required value={draft.startTime} onChange={event => update('startTime', event.target.value)} /></label><label>下班时间<input type="time" required value={draft.endTime} onChange={event => update('endTime', event.target.value)} /></label></div>
          <p className="work-clock-note">下班早于上班时按跨午夜夜班计算。时长按起止时间计算，包含午休。</p>
          <label>计时显示<select value={draft.displayMode} onChange={event => update('displayMode', event.target.value as WorkClockSettings['displayMode'])}><option value="remaining">距离下班还有多久</option><option value="elapsed">已经上班多长时间</option><option value="both">同时显示两种计时</option></select></label>
          <label>每周休息<select value={draft.weekendMode} onChange={event => update('weekendMode', event.target.value as WorkClockSettings['weekendMode'])}><option value="double">双休（周六、周日）</option><option value="single">单休</option></select></label>
          {draft.weekendMode === 'single' ? <label>单休日期<select value={draft.singleRestDay} onChange={event => update('singleRestDay', Number(event.target.value))}>{WEEKDAYS.map((day, index) => <option key={day} value={index}>{day}</option>)}</select></label> : null}
          <label className="work-clock-check"><input type="checkbox" checked={draft.followChinaCalendar} onChange={event => update('followChinaCalendar', event.target.checked)} />按中国节假日及调休补班计算</label>
          <label className="work-clock-check"><input type="checkbox" checked={draft.showWeekend} onChange={event => update('showWeekend', event.target.checked)} />显示周末／休息日倒计时</label>
          <label className="work-clock-check"><input type="checkbox" checked={draft.showHolidays} onChange={event => update('showHolidays', event.target.checked)} />显示节假日倒计时与节日列表</label>
          <label className="work-clock-check"><input type="checkbox" checked={draft.overtimeEnabled ?? true} onChange={event => update('overtimeEnabled', event.target.checked)} />自动记录非工作时间的平台使用</label>
          <label>加班无操作等待<select aria-label="加班无操作等待" value={draft.overtimeIdleMinutes ?? 30} onChange={event => update('overtimeIdleMinutes', Number(event.target.value))}>{[10,20,30,60].map(minutes => <option key={minutes} value={minutes}>{minutes}分钟</option>)}</select></label>
          <p className="work-clock-note">超时后结束时间回退到最后操作，并标记待确认；待确认不计入汇总。可确认下班、修正时间或衔接刚才一直工作的时段。</p>
          {error ? <p className="work-clock-error" role="alert">{error}</p> : null}
          <footer><button type="button" onClick={() => { setEditing(false); setError('') }}>取消</button><button type="submit" className="work-clock-save" disabled={saving}>{saving ? '保存中…' : '保存设置'}</button></footer>
        </form> : null}
        {payload && !editing && shift ? <>
          <OvertimePanel tracker={overtime} now={now} />
          <div className="work-clock-status"><span className={shift.status === 'working' ? 'working' : ''}>{({ working: '上班中', before: '尚未上班', finished: '已下班', rest: '今天休息' })[shift.status]}</span><small>{payload.settings.startTime}—{payload.settings.endTime}{payload.settings.endTime < payload.settings.startTime ? '（次日）' : ''}{shift.makeup ? ' · 调休补班' : shift.holiday ? ` · ${shift.holiday}` : ''}</small></div>
          {shift.status === 'rest' ? <div className="work-clock-countdown"><span>距离下次上班</span><strong data-clock="next-start">{shift.nextStart ? formatClockSeconds(Math.max(0, Math.ceil((shift.nextStart.getTime() - now.getTime()) / 1000)), true) : '暂无可用工作日'}</strong></div> : <>
            {mode !== 'elapsed' ? <div className="work-clock-countdown"><span>{shift.status === 'before' ? '距离本次下班（尚未上班）' : '距离下班'}</span><strong data-clock="remaining">{formatClockSeconds(shift.remainingSeconds)}</strong></div> : null}
            {mode !== 'remaining' ? <div className="work-clock-countdown"><span>{shift.status === 'finished' ? '本次已上班' : '已经上班'}</span><strong data-clock="elapsed">{formatClockSeconds(shift.elapsedSeconds)}</strong></div> : null}
          </>}
          {payload.settings.showWeekend ? <div className="work-clock-summary"><span>{weekend?.ongoing ? '正在周休中' : payload.settings.weekendMode === 'single' && payload.settings.singleRestDay !== 0 && payload.settings.singleRestDay !== 6 ? '距离休息日' : '距离周末'}</span><strong data-clock="weekend">{weekend?.ongoing ? '好好休息' : weekend ? formatClockSeconds(weekend.seconds, true) : '暂无可用安排'}</strong><small>{payload.settings.weekendMode === 'double' ? '双休' : `单休 · ${WEEKDAYS[payload.settings.singleRestDay]}`} · 从休息前最后一个班次下班时开始计算</small></div> : null}
          {payload.settings.showHolidays ? <>
            <div className="work-clock-summary holiday"><span>{holiday && holiday.start <= today ? `正在${holiday.name}假期中` : '距离下一个放假节日'}</span><strong data-clock="holiday">{holiday ? `${holiday.name} · 放假${holidayDayCount}天 · ${holiday.start <= today ? '假期中' : `还有${daysBetween(now, calendarDate(holiday.start))}天`}` : '暂无后续放假安排'}</strong><small>{holiday ? formatCalendarDateRange(holiday.start, holiday.end) : '请更新平台中的年度放假安排'}</small></div>
            <div className="work-clock-festivals"><div><strong>近期节日</strong><small><i className="off" />全民放假 <i />其他节日</small></div><ul>{festivals.map(festival => <li key={`${festival.start}:${festival.name}`} className={festival.dayOff ? 'day-off' : 'observance'}><span><b>{festival.name}{festival.dayOff ? ` · 放假${daysBetween(calendarDate(festival.start), calendarDate(festival.end)) + 1}天` : ''}</b><small>{formatCalendarDateRange(festival.start, festival.end)} · {festival.note}</small></span><em>{festival.start <= today ? '进行中' : `${daysBetween(now, calendarDate(festival.start))}天后`}</em></li>)}</ul></div>
          </> : null}
          {saved ? <p className="work-clock-saved" role="status">计时设置已保存，下次打开自动沿用。</p> : null}
          <p className="work-clock-note">设置保存在当前电脑。{payload.settings.followChinaCalendar && !payload.calendar.years.includes(now.getFullYear()) ? '当前年份尚未内置放假安排，暂按每周休息设置计算。' : ''}</p>
          {payload.settings.showHolidays ? <p className="work-clock-source">中国大陆放假安排：{payload.calendar.sources.map(source => <a key={source.year} href={source.url} target="_blank" rel="noreferrer" title={source.title}>{source.year}</a>)}；灰色节日不属于全民放假，部分人群可能放假。</p> : null}
        </> : null}
      </div>
    </section> : null}
  </div>
}
