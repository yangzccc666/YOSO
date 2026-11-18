import { useRef, useState } from 'react'
import { detectLocalStarLabels } from '../api'
import { icons } from '../icons'
import { HelpTip } from './HelpTip'
import { CollapsibleSection } from './CollapsibleSection'
import { PackageNamingHelper } from './PackageNamingHelper'
import { packageTitleWarning } from '../packageNaming'
import type { PackageNamingSeries } from '../packageNaming'
import { formatStarLabelText } from '../starLabelText'
import type { PlatformTaskStatus, WorkingValues } from '../types'

type Props = {
  values: WorkingValues
  output: string
  running: boolean
  platformTask: PlatformTaskStatus | null
  onBrowse: (fieldId: string, mode: string) => void
  onPathChange: (fieldId: string, value: string) => void
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  onRun: (parameterOverrides?: Record<string, string | number | boolean>) => void
}

const LABEL_SEPARATOR = /[,，\r\n]+/

function parseLabels(raw: string): string[] {
  return [...new Set(raw.split(LABEL_SEPARATOR).map((value) => value.trim()).filter(Boolean))]
}

function parentFolder(path: string): string {
  const trimmed = path.trim().replace(/[\\/]+$/, '')
  const separator = Math.max(trimmed.lastIndexOf('/'), trimmed.lastIndexOf('\\'))
  return separator > 0 ? trimmed.slice(0, separator) : ''
}

export function LocalStarPackagePanel(props: Props) {
  const [error, setError] = useState('')
  const [namingSeries, setNamingSeries] = useState<PackageNamingSeries[]>([])
  const [detectingLabels, setDetectingLabels] = useState(false)
  const [labelDetection, setLabelDetection] = useState<{ modelFile: string; message: string } | null>(null)
  const parameter = (id: string) => props.values.parameters[id]
  const set = props.onParameterChange
  const modelFile = String(props.values.paths.model_file || '')
  const title = String(parameter('title') || '')
  const labelsText = String(parameter('labels') || '')
  const labels = parseLabels(labelsText)
  const outputFolder = parentFolder(modelFile)
  const labelDetectionMessage = labelDetection?.modelFile === modelFile ? labelDetection.message : ''
  const latestModelFile = useRef(modelFile)
  const detectionRequest = useRef(0)
  latestModelFile.current = modelFile

  const detectLabels = async () => {
    setError('')
    setLabelDetection(null)
    if (labelsText.trim()) {
      try {
        const formatted = formatStarLabelText(labelsText)
        detectionRequest.current += 1
        set('labels', formatted)
        setLabelDetection({ modelFile, message: `已整理 ${parseLabels(formatted).length} 个类别：去除开头的数字序号和冒号，按原顺序每行一个填充。` })
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : '类别文本整理失败，原内容已保留。')
      }
      return
    }
    if (!modelFile.trim().toLowerCase().endsWith('.plan')) {
      setError('请先在 labels 中粘贴类别文本；或选择一个本机 TensorRT .plan 模型文件后识别配套类别。')
      return
    }
    const requestedModel = modelFile
    const requestId = ++detectionRequest.current
    setDetectingLabels(true)
    try {
      const result = await detectLocalStarLabels(requestedModel)
      if (requestId !== detectionRequest.current || latestModelFile.current !== requestedModel) return
      set('labels', result.labels.join('\n'))
      setLabelDetection({ modelFile: requestedModel, message: `${result.message} 来源：${result.source}` })
    } catch (reason) {
      if (requestId === detectionRequest.current && latestModelFile.current === requestedModel) {
        setError(reason instanceof Error ? reason.message : '自动识别类别失败。')
      }
    } finally {
      setDetectingLabels(false)
    }
  }

  const run = () => {
    setError('')
    if (!modelFile.trim().toLowerCase().endsWith('.plan')) {
      setError('请选择一个本机 TensorRT .plan 模型文件。')
      return
    }
    if (!title.trim()) {
      setError('请填写打包名称 title。')
      return
    }
    if (!labels.length) {
      setError('请至少填写一个类别名称。')
      return
    }
    const namingWarning = packageTitleWarning(title, labels, namingSeries)
    if (namingWarning && !window.confirm(`${namingWarning}\n仍按当前名称打包吗？`)) return
    props.onRun()
  }

  return <>
    <CollapsibleSection className="workspace-section calibration-section star-package-section" heading={<div className="section-title"><icons.Box size={21} /><h2>模型打包</h2></div>} count={4}>
      <div className="calibration-form">
        <label className="wide"><span>TensorRT PLAN 模型 <em>必填</em></span><div className="package-file-picker"><input value={modelFile} onChange={(event) => props.onPathChange('model_file', event.target.value)} placeholder="请选择或粘贴 .plan 文件路径" /><button type="button" onClick={() => props.onBrowse('model_file', 'file')}>浏览</button></div></label>
        <label><span>打包名称 title <em>必填；可手填或用命名助手生成</em></span><input value={title} onChange={(event) => set('title', event.target.value)} placeholder="例如：CT2-2#1.1.0.1117" /></label>
        <label><span className="help-heading">引擎精度声明<HelpTip label="STAR 精度声明说明">这是写入 STAR 的元数据声明，不会改变 PLAN 内部的计算精度，也不能把 INT8 恢复成 FP16/FP32。请与构建引擎时的实际精度对应。需要更高精度时，请从未量化 FP32 ONNX 重新构建 PLAN，再打包。</HelpTip></span><select aria-label="STAR 引擎精度声明" value={String(parameter('precision') || 'INT8 (1)')} onChange={event => set('precision', event.target.value)}><option>INT4 (0)</option><option>INT8 (1)</option><option>FP8 (2)</option><option>FP16 (3)</option><option>FP32 (4)</option></select></label>
        <PackageNamingHelper title={title} labels={labels} note={String(parameter('package_change_note') || '')} running={props.running} onTitle={value=>set('title',value)} onNote={value=>set('package_change_note',value)} onHistory={setNamingSeries} />
        <label className="wide"><span className="package-label-heading"><span>类别名称 labels <em>必填；可粘贴带序号类别，再点击右侧按钮整理</em></span><button type="button" aria-label="自动识别类别" onClick={detectLabels} disabled={detectingLabels || props.running}>{detectingLabels ? '正在识别…' : '自动识别类别'}</button></span><textarea rows={5} value={labelsText} onChange={(event) => { detectionRequest.current += 1; set('labels', event.target.value); setLabelDetection(null) }} placeholder={'0: MARK\n1: start_okp\n2: hold_inspect'} /></label>
        <div className="wide help-heading"><span>已识别 {labels.length} 个类别</span><HelpTip label="STAR 打包类别与输出说明"><p>粘贴类别后点击“自动识别类别”，会将 0: MARK 转为 MARK，保留原顺序，每行一个；支持英文或中文冒号。labels 留空时，按所选 PLAN 查找配套类别文件。</p><p>原始 trt.toml 不会被修改；生成的 STAR 将保存到 PLAN 同级目录。</p></HelpTip></div>
        {labelDetectionMessage ? <p className="wide package-label-detection"><icons.Check size={16} /><span>{labelDetectionMessage}</span></p> : null}
        {outputFolder ? <p className="wide package-output-location">输出目录：<strong>{outputFolder}</strong></p> : null}
      </div>
    </CollapsibleSection>

    <section className="workspace-section star-package-section">
      <details className="package-advanced">
        <summary><span><icons.Settings size={20} /><strong>高级打包配置</strong></span><em>已按当前 trt.toml 设置常用默认值</em><icons.ChevronRight size={18} /></summary>
        <div className="calibration-form">
          <label className="wide"><span>package_tool 路径</span><input value={String(parameter('package_tool_path') || '')} onChange={(event) => set('package_tool_path', event.target.value)} /></label>
          <label className="wide"><span>trt.toml 模板路径（可选，留空使用平台内置模板）</span><input value={String(parameter('template_file') || '')} onChange={(event) => set('template_file', event.target.value)} placeholder="无需准备外部 trt.toml；也可指定自定义模板" /></label>
          <label><span>硬件平台</span><input value={String(parameter('hardware_name') || '')} onChange={(event) => set('hardware_name', event.target.value)} /></label>
          <label><span>硬件架构</span><input value={String(parameter('architecture') || '')} onChange={(event) => set('architecture', event.target.value)} /></label>
          <label><span>驱动信息</span><input value={String(parameter('driver') || '')} onChange={(event) => set('driver', event.target.value)} /></label>
          <label><span>模型架构名称</span><input value={String(parameter('model_name') || '')} onChange={(event) => set('model_name', event.target.value)} /></label>
          <label><span>模型类别</span><select value={String(parameter('category') || 'generic')} onChange={(event) => set('category', event.target.value)}><option value="generic">generic</option><option value="seg">seg</option><option value="classify">classify</option></select></label>
          <label><span>版本信息</span><input value={String(parameter('version') || '')} onChange={(event) => set('version', event.target.value)} /></label>
          <label><span>颜色格式</span><select value={String(parameter('color_format') || 'RGB (2)')} onChange={(event) => set('color_format', event.target.value)}><option>NV12 (0)</option><option>NV21 (1)</option><option>RGB (2)</option><option>BGR (3)</option><option>GRAY (4)</option></select></label>
          <div className="wide package-tensor-fields"><span>输入张量 N × C × H × W</span><div>{(['input_n', 'input_c', 'input_h', 'input_w'] as const).map((id) => <input key={id} aria-label={id.replace('input_', '').toUpperCase()} title={id.replace('input_', '').toUpperCase()} type="number" min="1" value={String(parameter(id) ?? '')} onChange={(event) => set(id, event.target.value === '' ? '' : Number(event.target.value))} />)}</div></div>
        </div>
      </details>
    </section>

    <section className="remote-inference-section calibration-run-section">
      <div className="run-actions"><button className="run-button" onClick={run} disabled={props.running}><icons.Play size={18} fill="currentColor" />{props.platformTask ? '正在打包' : props.running ? '正在启动' : '开始打包 STAR'}</button><span className="ready"><icons.Info size={17} />完成后保存到 PLAN 同级目录，并加入实时推理待办</span></div>
      {error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{error}</span></div> : null}
      <details className="run-output" open><summary><icons.ChevronRight size={18} />打包日志</summary><pre>{props.output || '开始后将在这里显示临时配置、package_tool 输出和最终 STAR 文件路径'}</pre></details>
    </section>
  </>
}
