import type { Node } from '../projects/model'
import { calendarDate, dateKey } from '../workClock'

export type View = 'day' | 'week' | 'month'
export type Ownership = { id: string; name: string }
export type Options = {
  nodes: Node[]; report_projects: Ownership[]; categories: string[]; affairs: string[]
}
export type Entry = {
  id: string; project_id: string; project_name: string; scope_id: string; scope_name: string
  public_affair: string; task_category: string; work_content: string; duration_minutes: number; deliverable_note: string
}
export type Draft = Entry & { key: string; hours: string; publicMode: boolean }
export type Day = { work_date: string; revision: number; entries: Entry[]; previous: {work_date: string; entries: Entry[]} | null }
export type Group = {id: string; name: string; minutes: number}
export type Report = {total_minutes: number; filled_days: number; days: Record<string, number>; projects: Group[]; tasks: Group[]}

export function draft(entry?: Entry, copy = false): Draft {
  const key = crypto.randomUUID()
  if (entry) return {...entry, id: copy ? '' : entry.id, key, hours: String(entry.duration_minutes / 60), publicMode: !entry.project_id}
  return {id:'', key, project_id:'', project_name:'', scope_id:'', scope_name:'', public_affair:'',
    task_category:'', work_content:'', duration_minutes:0, deliverable_note:'', hours:'', publicMode:false}
}

export function hours(minutes: number) { return (minutes / 60).toFixed(1) }

export function rangeFor(anchor: string, view: View): [string, string] {
  const start = calendarDate(anchor), end = calendarDate(anchor)
  if (view === 'week') {
    const offset = (start.getDay() + 6) % 7
    start.setDate(start.getDate() - offset)
    end.setDate(end.getDate() + 6 - offset)
  } else if (view === 'month') {
    start.setDate(1)
    end.setMonth(end.getMonth() + 1, 0)
  }
  return [dateKey(start), dateKey(end)]
}

export function datesBetween(start: string, end: string) {
  const days: string[] = []
  const last = calendarDate(end)
  for (const d = calendarDate(start); d <= last; d.setDate(d.getDate() + 1)) days.push(dateKey(d))
  return days
}

export function validate(rows: Draft[]) {
  for (const [i, row] of rows.entries()) {
    const message = row.publicMode ? !row.public_affair.trim() && '请填写公共事务名称' : !row.project_id && '请选择项目或公共事务'
    const h = Number(row.hours)
    const problem = message || (!row.task_category.trim() && '请填写工作阶段') || (!row.work_content.trim() && '请填写具体工作内容')
      || (!(h >= 0.5 && h <= 24 && Number.isInteger(h * 2)) && '耗时应为 0.5～24 小时，步长 0.5')
    if (problem) return '第 ' + (i + 1) + ' 条，' + problem
  }
  return rows.reduce((n, r) => n + Number(r.hours), 0) > 24 ? '当日总工时超过 24 小时，请核对明细。' : ''
}
