import { useEffect, useRef, useState } from 'react'
import { deleteInferenceVideo, getInferenceVideos, INFERENCE_VIDEOS_UPDATED_EVENT, saveInferenceVideo } from '../api'
import { filterInferenceVideos, videoRecordDate, videoRecordLabel } from '../inferenceVideos'
import type { InferenceVideoRecord } from '../inferenceVideos'
import { HelpTip } from './HelpTip'

type Props = { path: string; scene: string; station: string; onPath: (value: string) => void; onScene: (value: string) => void; onStation: (value: string) => void }

export function InferenceVideoHistory(props: Props) {
  const [records, setRecords] = useState<InferenceVideoRecord[]>([])
  const [query, setQuery] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const [refresh, setRefresh] = useState(0)
  const generation = useRef(0)
  useEffect(() => {
    let cancelled = false
    const load = () => {
      const request = ++generation.current
      getInferenceVideos().then(data => { if (!cancelled && request === generation.current) { setRecords(data.videos); setError('') } })
        .catch(reason => { if (!cancelled && request === generation.current) setError(String(reason instanceof Error ? reason.message : reason)) })
    }
    const updated = (event: Event) => {
      load()
      setMessage(event instanceof CustomEvent && event.detail ? String(event.detail) : '')
    }
    load()
    window.addEventListener(INFERENCE_VIDEOS_UPDATED_EVENT, updated)
    window.addEventListener('focus', load)
    return () => { cancelled = true; generation.current += 1; window.removeEventListener(INFERENCE_VIDEOS_UPDATED_EVENT, updated); window.removeEventListener('focus', load) }
  }, [refresh])
  const selected = records.find(row => row.path === props.path.trim())
  const filtered = filterInferenceVideos(records, query)
  const options = selected && !filtered.some(row => row.id === selected.id) ? [selected, ...filtered] : filtered
  const save = async () => {
    setBusy(true); setError(''); setMessage('')
    generation.current += 1
    try {
      const data = await saveInferenceVideo(props.path, props.scene, props.station)
      generation.current += 1
      setRecords(data.videos); setMessage('已保存测试视频路径与场景、工位；下次可直接选择。')
    } catch (reason) { setError(reason instanceof Error ? reason.message : '保存失败') }
    finally { setBusy(false) }
  }
  const remove = async () => {
    if (!selected || !window.confirm(`仅删除视频记录“${selected.scene || '未分类场景'} · ${selected.station || '未分类工位'}”？\n不会删除视频文件，也不会停止正在运行的检测。`)) return
    setBusy(true); setError(''); setMessage(''); generation.current += 1
    try {
      await deleteInferenceVideo(selected.id)
      generation.current += 1
      setRecords(current => current.filter(row => row.id !== selected.id)); setMessage('已删除记录；当前输入路径和原视频文件均保留。')
    } catch (reason) { setError(reason instanceof Error ? reason.message : '删除失败') }
    finally { setBusy(false) }
  }
  return <section className="inference-video-history" aria-label="测试视频记录">
    <div className="help-heading"><strong>测试视频记录</strong><HelpTip label="测试视频记录说明">本地与远程推理共用，只保存本机视频路径，不复制视频、不保存模型或SSH信息。填写场景、工位后可手动保存；启动请求被接受后也自动记录，未填写时归为未分类。按最近使用或保存时间倒序，不受运行历史五条限制。相同路径只保留一条，重新使用更新排序。选择记录会填入视频路径及场景、工位，不会启动检测或更改模型。删除只移除记录，不删除视频。</HelpTip></div>
    <div className="inference-video-selection">
      <input type="search" aria-label="筛选测试视频" value={query} onChange={event => setQuery(event.target.value)} placeholder="筛选场景、工位或文件名" />
      <select aria-label="历史测试视频" title={selected?.path} disabled={busy} value={selected?.id || ''} onChange={event => {
        const record = records.find(row => row.id === event.target.value)
        if (record) { props.onPath(record.path); props.onScene(record.scene); props.onStation(record.station); setMessage('已填入所选视频，点击开始检测即可。'); setError('') }
      }}><option value="">{records.length ? `选择已记录视频（共${records.length}条，最近使用／保存优先）` : '暂无记录：填入视频路径后保存或开始检测'}</option>{options.map(row => <option key={row.id} value={row.id}>{videoRecordLabel(row)}</option>)}</select>
    </div>
    <div className="inference-video-metadata">
      <label>场景<input aria-label="测试视频场景" maxLength={100} placeholder="例如：磁探3号线" value={props.scene} onChange={event => props.onScene(event.target.value)} /></label>
      <label>工位<input aria-label="测试视频工位" maxLength={100} placeholder="例如：工位2" value={props.station} onChange={event => props.onStation(event.target.value)} /></label>
      <button type="button" disabled={busy || !props.path.trim()} onClick={() => void save()}>{busy ? '处理中…' : selected ? '更新视频记录' : '保存视频记录'}</button>
      <button type="button" disabled={busy || !selected} onClick={() => void remove()}>删除记录</button>
    </div>
    {selected ? <small className={selected.available ? '' : 'inference-video-warning'}>{selected.available ? `最近使用：${selected.lastUsedAt ? videoRecordDate(selected.lastUsedAt) : '尚未运行'} · 使用${selected.useCount}次` : '视频路径已失效，请在上方修改路径后重新保存。旧记录不会自动删除。'}</small> : null}
    {error ? <div className="inference-video-warning" role="alert">{error}<button type="button" onClick={() => setRefresh(current => current+1)}>重新读取</button></div> : null}
    {message ? <small role="status">{message}</small> : null}
  </section>
}
