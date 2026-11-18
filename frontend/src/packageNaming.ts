export type PackageVersion = [number, number, number]
export type PackageNamingRecord = { id: string; title: string; plan: string; output: string; labels: string[]; note: string; createdAt: string }
export type PackageNamingSeries = { key: string; line: string; station: number; version: PackageVersion; latest: PackageNamingRecord; count: number }
export type PackageNamingHistory = { series: PackageNamingSeries[]; records: PackageNamingRecord[]; count: number }
export const localDate = (date = new Date()) => `${date.getFullYear()}-${String(date.getMonth()+1).padStart(2,'0')}-${String(date.getDate()).padStart(2,'0')}`

export function parsePackageTitle(title: string) {
  const match = /^([A-Za-z][A-Za-z0-9_]{0,31})-([1-9]\d{0,3})#([1-9]\d{0,3})\.(0|[1-9]\d{0,3})\.(0|[1-9]\d{0,3})\.(\d{4})$/.exec(title)
  if (!match || match[0] !== title) return null
  const day = match[6]
  const test = new Date(2000, Number(day.slice(0,2))-1, Number(day.slice(2)), 12)
  if (test.getMonth()+1 !== Number(day.slice(0,2)) || test.getDate() !== Number(day.slice(2))) return null
  return { line: match[1].toUpperCase(), station: Number(match[2]), version: match.slice(3,6).map(Number) as PackageVersion, day }
}

export function buildPackageTitle(line: string, station: string, version: PackageVersion, date: string) {
  const value = `${line.trim().toUpperCase()}-${station}#${version.join('.')}.${date.slice(5,7)}${date.slice(8,10)}`
  if (!/^\d{4}-\d{2}-\d{2}$/.test(date) || localDate(new Date(`${date}T12:00:00`)) !== date || !parsePackageTitle(value)) throw new Error('产线请填英文首拼（字母开头，可含数字或下划线）；工位填1–9999；主版号1–9999，其余0–9999；日期需有效。')
  return value
}

export function bumpPackageVersion(version: PackageVersion, kind: 'major'|'minor'|'patch'): PackageVersion {
  const result: PackageVersion = kind === 'major' ? [version[0]+1,0,0] : kind === 'minor' ? [version[0],version[1]+1,0] : [version[0],version[1],version[2]+1]
  if (result.some(value => value > 9999)) throw new Error('版本号超过9999，请手动调整版本方案。')
  return result
}

export function comparePackageVersions(a: PackageVersion, b: PackageVersion) {
  for (let i=0;i<3;i++) if (a[i] !== b[i]) return a[i] - b[i]
  return 0
}

export function packageTitleWarning(title: string, labels: string[], history: PackageNamingSeries[]) {
  const parsed = parsePackageTitle(title)
  if (!parsed) return ''
  const series = history.find(item => item.line === parsed.line && item.station === parsed.station)
  if (!series) return ''
  const warnings = []
  if (comparePackageVersions(parsed.version, series.version) <= 0) warnings.push(`该工位已成功使用 ${series.version.join('.')}，当前版本未递增或发生回退。`)
  if (labels.length && JSON.stringify(labels) !== JSON.stringify(series.latest.labels) && parsed.version[0] <= series.version[0]) warnings.push('类别名称或顺序与最高版本不同，建议增加主版号。')
  return warnings.join(' ')
}
