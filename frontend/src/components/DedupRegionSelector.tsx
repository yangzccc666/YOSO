import { useEffect, useRef, useState } from 'react'
import { previewDedupRegion } from '../api'
import { readRegions, rectangle, validRegion } from '../processRegions'
import type { Region, RegionPreview } from '../processRegions'
import { HelpTip } from './HelpTip'

type Props = { folder: string; recursive: boolean; value: string; disabled: boolean; onChange: (value: string) => void }

export function DedupRegionSelector({ folder, recursive, value, disabled, onChange }: Props) {
  const [preview, setPreview] = useState<RegionPreview | null>(null)
  const [selected, setSelected] = useState('')
  const [sample, setSample] = useState('')
  const [draft, setDraft] = useState<Region | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [message, setMessage] = useState('')
  const request = useRef(0)
  const start = useRef<[number, number] | null>(null)
  useEffect(() => () => { request.current += 1 }, [])
  const stored = readRegions(value)
  const regions = preview && stored?.root === preview.root ? stored.regions : {}
  const ready = preview ? preview.folders.filter(key => validRegion(regions[key])).length : 0

  const load = async (key?: string, name?: string) => {
    const token = ++request.current
    setLoading(true); setError(''); setMessage('')
    try {
      const result = await previewDedupRegion(folder, recursive, key, name)
      if (request.current !== token) return
      setPreview(result); setSelected(result.folderKey); setSample(result.sampleName)
      setDraft(stored?.root === result.root ? stored.regions[result.folderKey] || null : null)
    } catch (reason) {
      if (request.current === token) setError(reason instanceof Error ? reason.message : '读取图片失败')
    } finally { if (request.current === token) setLoading(false) }
  }
  const save = () => {
    if (!preview || !validRegion(draft)) return
    if ((draft[2] - draft[0]) * preview.width < 16 || (draft[3] - draft[1]) * preview.height < 16) {
      setError('区域太小，请覆盖完整作业区，至少 16×16 像素。'); return
    }
    onChange(JSON.stringify({ version: 1, root: preview.root, regions: { ...regions, [preview.folderKey]: draft } }))
    setError(''); setMessage(`已保存工位区域：${preview.folderKey}`)
  }
  const point = (event: React.PointerEvent<HTMLDivElement>): [number, number] => {
    const box = event.currentTarget.getBoundingClientRect()
    return [(event.clientX - box.left) / box.width, (event.clientY - box.top) / box.height]
  }
  return <div className="dedup-region-panel" aria-label="工位关注区域">
    <div className="dedup-region-heading"><strong>目标工位区域</strong><HelpTip label="工位框选说明">在真实图片上拖动框选完整作业区，包括手、工具、工件及其运动范围，尽量排除旁边工位。每个目录分别保存；图片按相对区域比较，不裁剪或改写原图。只读预览不会启动去重。不同视角不要共用区域；若同目录混有不同机位，请先分开。此模式允许有限姿态/压缩差异，不要求像素一致，不保证微小缺陷或工序语义相同。先分析检查对照样本，再执行；执行会移入备份，非永久删除。</HelpTip>
      <button type="button" className="outline-button" disabled={disabled || loading || !folder.trim()} onClick={() => void load()}>{loading ? '读取中…' : '读取真实图片目录'}</button>
    </div>
    {preview ? <>
      <div className="dedup-region-controls"><label>工位图片目录<select aria-label="工位图片目录" value={selected} disabled={disabled || loading} onChange={event => void load(event.target.value)}>{preview.folders.map(key => <option key={key} value={key}>{key === '.' ? '当前目录' : key}</option>)}</select></label><span>已配置 {ready}/{preview.folders.length} 个目录</span></div>
      <div className="dedup-region-controls"><label>示例图片文件名<input aria-label="示例图片文件名" value={sample} disabled={disabled || loading} onChange={event => setSample(event.target.value)} /></label><button type="button" className="outline-button" disabled={disabled || loading || selected !== preview.folderKey} onClick={() => void load(selected, sample)}>更换示例图片</button></div>
      <div className="dedup-region-canvas" aria-label="在图片上框选目标工位" style={{ pointerEvents: disabled || loading ? 'none' : undefined }}
        onPointerDown={event => { if (event.button !== 0) return; event.preventDefault(); start.current = point(event); event.currentTarget.setPointerCapture(event.pointerId); setMessage(''); setDraft(null) }}
        onPointerMove={event => { if (start.current) setDraft(rectangle(start.current, point(event))) }}
        onPointerUp={event => { if (!start.current) return; setDraft(rectangle(start.current, point(event))); start.current = null; event.currentTarget.releasePointerCapture(event.pointerId) }}
        onPointerCancel={() => { start.current = null }}>
        <img src={preview.image} alt={`工位 ${preview.folderKey} 示例：${preview.sampleName}`} draggable={false} />
        {draft ? <span className="dedup-region-box" style={{ left: `${draft[0] * 100}%`, top: `${draft[1] * 100}%`, width: `${(draft[2] - draft[0]) * 100}%`, height: `${(draft[3] - draft[1]) * 100}%` }} /> : null}
      </div>
      <div className="dedup-region-controls"><span>{preview.width}×{preview.height} · {preview.sampleName}</span><span>{draft ? `框选：${draft.map(n => (n * 100).toFixed(1)).join('%, ')}%` : '请在图片上拖动框选完整作业区'}</span></div>
      <div className="dedup-region-actions"><button type="button" className="outline-button" disabled={disabled || loading} onClick={() => setDraft([0, 0, 1, 1])}>使用整张图</button><button type="button" className="outline-button save-button" disabled={disabled || loading || !validRegion(draft)} onClick={save}>保存本工位区域</button>
        {preview.folders.length > 1 ? <button type="button" className="outline-button" disabled={disabled || loading || !validRegion(draft)} onClick={() => {
          if (!draft || !window.confirm('仅适用于相同机位、相同构图的图片！将此区域填充到尚未配置的目录，已配置目录不覆盖。是否继续？')) return
          const next = { ...regions }
          preview.folders.forEach(key => { if (!next[key]) next[key] = draft })
          onChange(JSON.stringify({ version: 1, root: preview.root, regions: next })); setMessage('已填充未配置目录，请确认各工位视角一致。')
        }}>填充未配置工位（相同视角）</button> : null}
      </div>
    </> : null}
    {message ? <p role="status" className="dedup-region-message">{message}</p> : null}
    {error ? <p role="alert" className="dedup-region-error">{error}</p> : null}
  </div>
}
