import { useRef, useState } from 'react'
import { detectLocalStarLabels } from '../api'
import { icons } from '../icons'
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
    if (!modelFile.trim().toLowerCase().endsWith('.plan')) {
      setError('请先选择一个本机 TensorRT .plan 模型文件。')
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
      setError(reason instanceof Error ? reason.message : '自动识别类别失败。')
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
    props.onRun()
  }

  return <>
    <section className="workspace-section calibration-section star-package-section">
      <div className="section-title"><icons.Box size={21} /><h2>模型打包</h2></div>
      <div className="calibration-form">
        <label className="wide"><span>TensorRT PLAN 模型 <em>必填</em></span><div className="package-file-picker"><input value={modelFile} onChange={(event) => props.onPathChange('model_file', event.target.value)} placeholder="请选择或粘贴 .plan 文件路径" /><button type="button" onClick={() => props.onBrowse('model_file', 'file')}>浏览</button></div></label>
        <label><span>打包名称 title <em>必填</em></span><input value={title} onChange={(event) => set('title', event.target.value)} placeholder="例如：tcl-191-0920" /></label>
        <label className="wide"><span className="package-label-heading"><span>类别名称 labels <em>必填；每行一个或用逗号分隔</em></span><button type="button" onClick={detectLabels} disabled={detectingLabels || props.running}>{detectingLabels ? '正在识别…' : '自动识别类别'}</button></span><textarea rows={5} value={labelsText} onChange={(event) => { detectionRequest.current += 1; set('labels', event.target.value); setLabelDetection(null) }} placeholder={'elec_screw\nboard\nconnector'} /></label>
        {labelDetectionMessage ? <p className="wide package-label-detection"><icons.Check size={16} /><span>{labelDetectionMessage}</span></p> : null}
        <p className="wide calibration-hint">已识别 <strong>{labels.length}</strong> 个类别。原始 trt.toml 不会被修改；生成的 STAR 将保存到 PLAN 同级目录{outputFolder ? <>：<strong>{outputFolder}</strong></> : null}。</p>
      </div>
    </section>

    <section className="workspace-section star-package-section">
      <details className="package-advanced">
        <summary><span><icons.Settings size={20} /><strong>高级打包配置</strong></span><em>已按当前 trt.toml 设置常用默认值</em><icons.ChevronRight size={18} /></summary>
        <div className="calibration-form">
          <label className="wide"><span>package_tool 路径</span><input value={String(parameter('package_tool_path') || '')} onChange={(event) => set('package_tool_path', event.target.value)} /></label>
          <label className="wide"><span>trt.toml 模板路径</span><input value={String(parameter('template_file') || '')} onChange={(event) => set('template_file', event.target.value)} /></label>
          <label><span>硬件平台</span><input value={String(parameter('hardware_name') || '')} onChange={(event) => set('hardware_name', event.target.value)} /></label>
          <label><span>硬件架构</span><input value={String(parameter('architecture') || '')} onChange={(event) => set('architecture', event.target.value)} /></label>
          <label><span>驱动信息</span><input value={String(parameter('driver') || '')} onChange={(event) => set('driver', event.target.value)} /></label>
          <label><span>模型架构名称</span><input value={String(parameter('model_name') || '')} onChange={(event) => set('model_name', event.target.value)} /></label>
          <label><span>模型类别</span><select value={String(parameter('category') || 'generic')} onChange={(event) => set('category', event.target.value)}><option value="generic">generic</option><option value="seg">seg</option><option value="classify">classify</option></select></label>
          <label><span>版本信息</span><input value={String(parameter('version') || '')} onChange={(event) => set('version', event.target.value)} /></label>
          <label><span>模型精度</span><select value={String(parameter('precision') || 'INT8 (1)')} onChange={(event) => set('precision', event.target.value)}><option>INT4 (0)</option><option>INT8 (1)</option><option>FP8 (2)</option><option>FP16 (3)</option><option>FP32 (4)</option></select></label>
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
