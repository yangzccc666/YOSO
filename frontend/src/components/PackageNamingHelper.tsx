import { useEffect, useState } from 'react'
import { getPackageNaming } from '../api'
import { HelpTip } from './HelpTip'
import { buildPackageTitle, bumpPackageVersion, localDate, packageTitleWarning, parsePackageTitle } from '../packageNaming'
import type { PackageNamingHistory, PackageNamingSeries, PackageVersion } from '../packageNaming'

type Props = { title: string; labels: string[]; note: string; running: boolean; onTitle: (title: string) => void; onNote: (note: string) => void; onHistory: (series: PackageNamingSeries[]) => void }

export function PackageNamingHelper(props: Props) {
  const initial = parsePackageTitle(props.title)
  const [line, setLine] = useState(initial?.line || '')
  const [station, setStation] = useState(initial ? String(initial.station) : '')
  const [version, setVersion] = useState<PackageVersion>(initial?.version || [1,0,0])
  const [date, setDate] = useState(localDate)
  const [history, setHistory] = useState<PackageNamingHistory>({ series: [], records: [], count: 0 })
  const [loading, setLoading] = useState(false)
  const [loadError, setLoadError] = useState('')
  const [refresh, setRefresh] = useState(0)
  const [message, setMessage] = useState('')
  const { onHistory } = props
  useEffect(() => {
    if (props.running) return
    let cancelled = false
    const timer = window.setTimeout(() => {
      setLoading(true)
      getPackageNaming(line.trim(), station).then(value => {
        if (cancelled) return
        setHistory(value); onHistory(value.series); setLoadError('')
      }).catch(reason => { if (!cancelled) setLoadError(reason instanceof Error ? reason.message : '读取版本记录失败') })
        .finally(() => { if (!cancelled) setLoading(false) })
    }, 200)
    return () => { cancelled = true; window.clearTimeout(timer) }
  }, [line, station, props.running, refresh, onHistory])
  const latest = history.series.find(item => item.line === line.trim().toUpperCase() && String(item.station) === station)
  let preview = '', validation = ''
  try { preview = buildPackageTitle(line, station, version, date) }
  catch (reason) { validation = reason instanceof Error ? reason.message : '请检查命名参数' }
  const warning = packageTitleWarning(props.title, props.labels, history.series)
  const increment = (kind: 'major'|'minor'|'patch') => {
    try {
      const base = latest?.version || version
      setVersion(bumpPackageVersion(base, kind))
      setDate(localDate()); setMessage(latest ? `已按最高成功版本 ${base.join('.')} 计算建议；还需点击“填入打包名称”。` : '尚无成功记录，已按输入版本递增。')
    } catch (reason) { setMessage(String(reason)) }
  }
  return <div className="wide package-naming">
    <details>
      <summary>命名助手与版本记录 <small>产线-工位#主版号.次版号.修订号.mmdd</small></summary>
      <div className="package-naming-body">
        <div className="help-heading"><strong>示例：CT2-2#1.1.0.1117</strong><HelpTip label="模型命名规则">产线填写英文首拼或标识；工位使用自然数。新增标签：主版号加1，其余归零；增加数据：次版号加1，修订归零；重标或重训：修订号加1。日期为实际发布日期mmdd，可修改。这里的版本用于打包title，不会修改高级配置中的model.version。</HelpTip></div>
        <label>已记录工位<select aria-label="已记录工位" value={latest?.key || ''} disabled={props.running} onChange={event => {
          const item = history.series.find(item => item.key === event.target.value)
          if (item) { setLine(item.line); setStation(String(item.station)); setVersion(item.version); setDate(localDate()); setMessage('已载入该工位的最高成功版本，请选择本次变更或手动调整。') }
        }}><option value="">新工位／手动填写</option>{history.series.map(item => <option key={item.key} value={item.key}>{item.key} · 最高 {item.version.join('.')}</option>)}</select></label>
        <div className="package-naming-fields">
          <label>产线标识<input aria-label="命名产线标识" value={line} maxLength={32} disabled={props.running} placeholder="CT2" onChange={event => { setLine(event.target.value.toUpperCase()); setMessage('') }} /></label>
          <label>工位号<input aria-label="命名工位号" type="number" min="1" max="9999" step="1" value={station} disabled={props.running} placeholder="2" onChange={event => { setStation(event.target.value); setMessage('') }} /></label>
          {(['主版号','次版号','修订号'] as const).map((name,index) => <label key={name}>{name}<input aria-label={`命名${name}`} type="number" min={index===0?1:0} max="9999" step="1" value={Number.isNaN(version[index]) ? '' : version[index]} disabled={props.running} onChange={event => setVersion(current => current.map((value,i) => i===index ? (event.target.value==='' ? NaN : Number(event.target.value)) : value) as PackageVersion)} /></label>)}
          <label>发布日期<input aria-label="命名发布日期" type="date" value={date} disabled={props.running} onChange={event => setDate(event.target.value)} /></label>
        </div>
        <div className="package-naming-actions">
          <button type="button" disabled={props.running || !parsePackageTitle(props.title)} onClick={() => {
            const parsed = parsePackageTitle(props.title)
            if (parsed) { setLine(parsed.line); setStation(String(parsed.station)); setVersion(parsed.version); setDate(localDate()); setMessage('已读取当前title中的工位和版本；新发布日期默认今天。') }
          }}>读取当前名称</button>
          <button type="button" disabled={props.running} onClick={() => increment('major')}>新增标签 · 主版+1</button>
          <button type="button" disabled={props.running} onClick={() => increment('minor')}>增加数据 · 次版+1</button>
          <button type="button" disabled={props.running} onClick={() => increment('patch')}>重标重训 · 修订+1</button>
          {latest ? <button type="button" disabled={props.running} onClick={() => { setVersion(latest.version); setMessage('已载入最高成功版本，可手动调整。') }}>载入最高版本</button> : null}
        </div>
        <p className="package-naming-latest">{loading ? '正在读取版本台账…' : latest ? `本工位最高成功版本：${latest.latest.title}（共${latest.count}次成功打包）` : '本工位暂无成功版本记录，可手动填写已有版本作为起点。'}</p>
        <label>本次变更说明（可选）<input aria-label="打包变更说明" value={props.note} maxLength={500} disabled={props.running} placeholder="例如：增加夜班样本；修正MARK标签" onChange={event => props.onNote(event.target.value)} /></label>
        <div className="package-naming-preview"><span>命名预览</span><strong>{preview || '请填写有效的产线、工位、版本和日期'}</strong><button type="button" disabled={!preview || props.running} title={validation} onClick={() => { props.onTitle(preview); setMessage('已填入title；仅成功打包后记录版本。') }}>填入打包名称</button></div>
        {!preview && (line || station) ? <p className="package-naming-warning">{validation}</p> : null}
        {message ? <p className="package-naming-message" role="status">{message}</p> : null}
        <div className="help-heading package-naming-note"><small>仅成功打包记录版本；预览不会自动覆盖名称。</small><HelpTip label="版本记录说明"><p>默认起点1.0.0，可手动校准。相同工位的递增按钮均基于最高成功版本，不会因重复点击或打包失败空耗版本。手填名称仍可使用。</p><p>台账保存在当前电脑，独立于最近5条运行历史；其他电脑及尚未登记的历史版本需手动校准。日期不包含年份，请持续递增版本号。</p></HelpTip></div>
        <details className="package-version-history"><summary>本工位成功记录（{history.count}次，最多展示最近30条）</summary><div className="package-version-list">{history.records.length ? history.records.map(record => <article key={record.id}><strong>{record.title}</strong><small>{new Date(record.createdAt).toLocaleString('zh-CN')} · {record.labels.length}类</small>{record.note ? <p>{record.note}</p> : null}<small title={record.output}>{record.output}</small></article>) : <p>暂无记录。生成名称或打包失败不会写入。</p>}</div></details>
      </div>
    </details>
    {loadError ? <p className="package-naming-warning" role="alert">版本台账：{loadError}<button type="button" onClick={() => setRefresh(value=>value+1)}>重试</button></p> : null}
    {warning ? <p className="package-naming-warning" role="status">命名提醒：{warning}</p> : null}
  </div>
}
