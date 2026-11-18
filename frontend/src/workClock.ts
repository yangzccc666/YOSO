export type WorkClockSettings = {
  startTime: string
  endTime: string
  displayMode: 'remaining' | 'elapsed' | 'both'
  weekendMode: 'single' | 'double'
  singleRestDay: number
  followChinaCalendar: boolean
  showWeekend: boolean
  showHolidays: boolean
  overtimeEnabled: boolean
  overtimeIdleMinutes: number
}

export type ChinaHoliday = { name: string; start: string; end: string }
export type ChinaCalendar = {
  years: number[]
  holidays: ChinaHoliday[]
  workdays: string[]
  sources: Array<{ year: number; title: string; url: string }>
}
export type WorkClockPayload = { settings: WorkClockSettings; calendar: ChinaCalendar }

export const WEEKDAYS = ['周日', '周一', '周二', '周三', '周四', '周五', '周六']
export const DEFAULT_WORK_CLOCK: WorkClockSettings = {
  startTime: '09:00', endTime: '18:00', displayMode: 'remaining',
  weekendMode: 'double', singleRestDay: 0, followChinaCalendar: true,
  showWeekend: true, showHolidays: true,
  overtimeEnabled: true, overtimeIdleMinutes: 30,
}

export function dateKey(date: Date): string {
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}-${String(date.getDate()).padStart(2, '0')}`
}

export function calendarDate(key: string): Date {
  const [year, month, day] = key.split('-').map(Number)
  return new Date(year, month - 1, day)
}

/** Parse calendar components locally, never as UTC midnight (which can shift weekdays). */
export function formatCalendarDateRange(start: string, end = start): string {
  const format = (key: string) => `${key}（${WEEKDAYS[calendarDate(key).getDay()]}）`
  return start === end ? format(start) : `${format(start)} 至 ${format(end)}`
}

function addDays(date: Date, count: number): Date {
  const next = new Date(date)
  next.setDate(next.getDate() + count)
  return next
}

function atTime(date: Date, time: string): Date {
  const [hours, minutes] = time.split(':').map(Number)
  return new Date(date.getFullYear(), date.getMonth(), date.getDate(), hours, minutes)
}

export function daysBetween(from: Date, to: Date): number {
  // Calendar days stay correct across DST and never depend on parsing UTC dates.
  return Math.round((Date.UTC(to.getFullYear(), to.getMonth(), to.getDate()) - Date.UTC(from.getFullYear(), from.getMonth(), from.getDate())) / 86400000)
}

export function holidayOn(date: Date, calendar: ChinaCalendar): ChinaHoliday | undefined {
  const key = dateKey(date)
  return calendar.holidays.find(holiday => key >= holiday.start && key <= holiday.end)
}

function weeklyRestDay(date: Date, settings: WorkClockSettings): boolean {
  return settings.weekendMode === 'double' ? [0, 6].includes(date.getDay()) : date.getDay() === settings.singleRestDay
}

export function isWorkingDay(date: Date, settings: WorkClockSettings, calendar: ChinaCalendar): boolean {
  if (settings.followChinaCalendar) {
    if (calendar.workdays.includes(dateKey(date))) return true
    if (holidayOn(date, calendar)) return false
  }
  return !weeklyRestDay(date, settings)
}

function shiftForDay(day: Date, settings: WorkClockSettings) {
  const start = atTime(day, settings.startTime)
  let end = atTime(day, settings.endTime)
  if (end <= start) end = addDays(end, 1)
  return { start, end }
}

function nextShiftStart(now: Date, settings: WorkClockSettings, calendar: ChinaCalendar): Date | null {
  for (let offset = 0; offset < 370; offset++) {
    const day = addDays(now, offset)
    if (!isWorkingDay(day, settings, calendar)) continue
    const start = atTime(day, settings.startTime)
    if (start > now) return start
  }
  return null
}

export function workShiftState(now: Date, settings: WorkClockSettings, calendar: ChinaCalendar) {
  let anchor = now
  const previousDay = addDays(now, -1)
  const previous = shiftForDay(previousDay, settings)
  // The early hours of a night shift belong to its starting workday, even if
  // the calendar date has advanced to Saturday or a public holiday.
  if (isWorkingDay(previousDay, settings, calendar) && now >= previous.start && now < previous.end) anchor = previousDay
  const shift = shiftForDay(anchor, settings)
  const workingDay = isWorkingDay(anchor, settings, calendar)
  const status = !workingDay ? 'rest' : now < shift.start ? 'before' : now >= shift.end ? 'finished' : 'working'
  return {
    status, start: shift.start, end: shift.end,
    elapsedSeconds: workingDay ? Math.max(0, Math.floor((Math.min(now.getTime(), shift.end.getTime()) - shift.start.getTime()) / 1000)) : 0,
    remainingSeconds: workingDay ? Math.max(0, Math.ceil((shift.end.getTime() - now.getTime()) / 1000)) : 0,
    nextStart: nextShiftStart(now, settings, calendar),
    makeup: settings.followChinaCalendar && calendar.workdays.includes(dateKey(anchor)),
    holiday: settings.followChinaCalendar ? holidayOn(anchor, calendar)?.name : undefined,
  }
}

export function weeklyRestState(now: Date, settings: WorkClockSettings, calendar: ChinaCalendar) {
  let upcoming: { start: Date; end: Date; restDay: Date } | null = null
  // Include the previous week to recognize a weekend already in progress.
  for (let offset = -8; offset <= 21; offset++) {
    const day = addDays(now, offset)
    if (!weeklyRestDay(day, settings) || isWorkingDay(day, settings, calendar)) continue
    let before = addDays(day, -1)
    let after = addDays(day, 1)
    for (let i = 0; i < 20 && !isWorkingDay(before, settings, calendar); i++) before = addDays(before, -1)
    for (let i = 0; i < 20 && !isWorkingDay(after, settings, calendar); i++) after = addDays(after, 1)
    const start = shiftForDay(before, settings).end
    const end = shiftForDay(after, settings).start
    if (end <= now) continue
    if (!upcoming || start < upcoming.start) upcoming = { start, end, restDay: day }
  }
  if (!upcoming) return null
  return { ...upcoming, ongoing: now >= upcoming.start, seconds: Math.max(0, Math.ceil((upcoming.start.getTime() - now.getTime()) / 1000)) }
}

export function formatClockSeconds(seconds: number, withDays = false): string {
  const total = Math.max(0, Math.floor(seconds))
  const days = Math.floor(total / 86400)
  const hours = withDays ? Math.floor(total / 3600) % 24 : Math.floor(total / 3600)
  const minutes = Math.floor(total / 60) % 60
  return `${withDays ? `${days}天 ` : ''}${String(hours).padStart(2, '0')}小时 ${String(minutes).padStart(2, '0')}分 ${String(total % 60).padStart(2, '0')}秒`
}

export function upcomingHolidays(now: Date, calendar: ChinaCalendar): ChinaHoliday[] {
  const key = dateKey(now)
  return calendar.holidays.filter(holiday => holiday.end >= key).sort((left, right) => left.start.localeCompare(right.start))
}

// These are observances, not additional whole-population public holidays.
const OBSERVANCES = [
  ['02-14', '情人节'], ['03-08', '妇女节'], ['03-12', '植树节'],
  ['05-04', '青年节'], ['06-01', '儿童节'], ['07-01', '建党节'],
  ['08-01', '建军节'], ['09-10', '教师节'], ['12-04', '国家宪法日'],
  ['12-13', '国家公祭日'], ['12-25', '圣诞节'],
] as const

export function upcomingFestivals(now: Date, calendar: ChinaCalendar) {
  const today = dateKey(now)
  const holidays = upcomingHolidays(now, calendar).map(holiday => ({ ...holiday, dayOff: true, note: '全民放假' }))
  const observances = [now.getFullYear(), now.getFullYear() + 1].flatMap(year => OBSERVANCES.map(([date, name]) => {
    const key = `${year}-${date}`
    const coinciding = holidayOn(calendarDate(key), calendar)
    return { name, start: key, end: key, dayOff: false,
      note: coinciding ? `不单独放假 · ${coinciding.name}假期内` : ['妇女节', '青年节', '儿童节', '建军节'].includes(name) ? '部分人群放假' : '不放假' }
  })).filter(item => item.end >= today)
  return [...holidays, ...observances].sort((left, right) => left.start.localeCompare(right.start) || Number(right.dayOff) - Number(left.dayOff)).slice(0, 8)
}
