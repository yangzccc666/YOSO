import { useEffect, useRef, useState } from 'react'
import { detectExportModelLabels } from '../api'
import type { ExportModelLabels } from '../api'
import type { WorkingValues } from '../types'
import { parseExportModelPaths } from '../jetsonModelLabels'
import { HelpTip } from './HelpTip'

type Record = { path: string; metadata?: ExportModelLabels; error?: string }
type Props = { values: WorkingValues; running: boolean }

export function JetsonModelLabels({ values, running }: Props) {
  const raw = String(values.paths.weights_file || '')
  const key = JSON.stringify([raw.trim(), values.parameters.model_family || 'YOLO26', values.parameters.python_path || '', values.parameters.yolov5_repo || ''])
  const latestKey = useRef(key)
  latestKey.current = key
  const requestId = useRef(0)
  const [batch, setBatch] = useState<{ key: string; records: Record[]; reading: boolean; current: string; error: string } | null>(null)
  useEffect(() => () => { requestId.current += 1 }, [])
  const current = batch?.key === key ? batch : null
  const reading = Boolean(current?.reading)
  const read = async () => {
    let paths: string[]
    try { paths = parseExportModelPaths(raw); if (!paths.length) throw new Error('请先选择或粘贴 PT 权重路径。') }
    catch (cause) { setBatch({ key, records: [], reading: false, current: '', error: cause instanceof Error ? cause.message : 'PT 路径格式错误' }); return }
    const id = ++requestId.current
    const valid = () => requestId.current === id && latestKey.current === key
    const records: Record[] = []
    setBatch({ key, records, reading: true, current: paths[0], error: '' })
    try {
      for (const path of paths) {
        if (!valid()) return
        setBatch({ key, records: [...records], reading: true, current: path, error: '' })
        try { records.push({ path, metadata: await detectExportModelLabels(path, values.parameters) }) }
        catch (cause) { records.push({ path, error: cause instanceof Error ? cause.message : '读取类别失败' }) }
        if (!valid()) return
        setBatch({ key, records: [...records], reading: true, current: path, error: '' })
      }
    } finally {
      if (id === requestId.current) setBatch((previous) => previous?.key === key ? { ...previous, reading: false, current: '' } : previous)
    }
  }

  return <section className="jetson-model-labels" aria-label="PT 模型类别识别">
    <header><strong className="help-heading">PT 模型类别<HelpTip label="PT 类别识别说明">直接读取 PT 内嵌 names，使用本页配置的模型版本和 YOLO Python。批量模型分别显示，不混用标签；每行对应类别 ID 0、1、2…，可选中文本复制。不导出 ONNX，也不修改 PT。仅加载可信来源的模型。</HelpTip></strong><button type="button" onClick={() => void read()} disabled={running || reading || !raw.trim()}>{reading ? '正在识别类别…' : '识别 PT 类别'}</button></header>
    {reading ? <p role="status">正在读取：{current?.current}</p> : null}
    {current?.error ? <p className="jetson-label-error" role="alert">{current.error}</p> : null}
    {current?.records.map((record) => <article key={record.path}>
      <strong title={record.path}>{record.path}</strong>
      {record.error ? <p className="jetson-label-error" role="alert">{record.error}</p> : record.metadata ? <>
        <span>已识别 {record.metadata.count} 个类别 · {record.metadata.modelFamily} · {record.metadata.task}</span>
        <textarea aria-label={`${record.path} 的类别名称`} readOnly rows={Math.min(8, Math.max(3, record.metadata.count))} value={record.metadata.labels.join('\n')} />
        <small>{record.metadata.source}；保持类别 ID 顺序，不排序、不去重。</small>
      </> : null}
    </article>)}
    {!reading && current?.records.length ? <p role="status">识别完成：成功 {current.records.filter((record) => record.metadata).length} 个，失败 {current.records.filter((record) => record.error).length} 个。更换 PT 或运行环境后请重新识别。</p> : null}
  </section>
}
