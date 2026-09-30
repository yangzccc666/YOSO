import { useState } from 'react'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import { RemoteInferencePanel } from './RemoteInferencePanel'
import { LocalPTInferencePanel } from './LocalPTInferencePanel'
import { RemoteYoloDatasetPanel } from './RemoteYoloDatasetPanel'
import { RemoteCalibrationPanel } from './RemoteCalibrationPanel'
import { YoloTrainingPanel } from './YoloTrainingPanel'
import { RemoteTensorRTBuildPanel } from './RemoteTensorRTBuildPanel'
import { LocalStarPackagePanel } from './LocalStarPackagePanel'
import { FunctionRunHistory } from './FunctionRunHistory'
import { WorkflowTodoPanel } from './WorkflowTodoPanel'
import type { FunctionDefinition, FunctionParameter, PlatformTaskStatus, WorkingValues, YoloTrainingRecommendation } from '../types'
import type { WorkflowTodo } from '../workflowTodos'

type Props = {
  item: FunctionDefinition | null
  values: WorkingValues
  output: string
  running: boolean
  activeTasks: PlatformTaskStatus[]
  onEdit: () => void
  onBrowse: (fieldId: string, mode: string) => void
  onPathChange: (fieldId: string, value: string) => void
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  onAddPath: () => void
  onAddParameter: () => void
  onRun: (parameterOverrides?: Record<string, string | number | boolean>) => void
  onStop: (taskId: string) => void
  yoloTrainingRecommendation: YoloTrainingRecommendation | null
  workflowTodos: WorkflowTodo[]
  onApplyWorkflowTodo: (item: WorkflowTodo) => void
  onDeleteWorkflowTodo: (id: string) => void
  onWorkflowComplete: (handlerId: string) => void
}

function ParameterValue({ field, value, onChange }: { field: FunctionParameter; value: string | number | boolean; onChange: (value: string | number | boolean) => void }) {
  if (field.type === 'boolean') return <label className="switch-control"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} /><span /><em>{value ? '开启' : '关闭'}</em></label>
  if (field.type === 'select') return <span className="workspace-select"><select value={String(value)} onChange={(event) => onChange(event.target.value)}><option value="">请选择</option>{field.options.map((option) => <option value={option} key={option}>{option}</option>)}</select><icons.ChevronDown size={16} /></span>
  const passwordRemembered = field.type === 'password' && value === REMEMBERED_PASSWORD_VALUE
  const displayedValue = passwordRemembered ? '已保存密码' : String(value)
  const placeholder = field.id === 'labels'
    ? '必填，例如：person,car'
    : field.id === 'remote_password'
      ? '请输入本次 SSH 连接密码'
    : field.type === 'password'
      ? '首次连接或未保存时输入'
      : '请输入参数值'
  return <input type={field.type === 'number' ? 'number' : field.type === 'password' ? 'password' : 'text'} autoComplete={field.type === 'password' ? 'current-password' : undefined} value={displayedValue} placeholder={placeholder} onFocus={() => { if (passwordRemembered) onChange('') }} onChange={(event) => onChange(field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
}

export function FunctionWorkspace(props: Props) {
  const [yoloMode, setYoloMode] = useState<'dataset' | 'training'>('dataset')
  if (!props.item) return <main className="empty-workspace"><span><icons.Box size={34} /></span><h2>暂无可用功能</h2><p>后续接入新的处理脚本后，功能会显示在左侧分组中。</p></main>
  const item = props.item
  const isYoloWorkspace = item.handlerId === 'yolo.split_dataset'
  const isCalibrationWorkspace = item.handlerId === 'calib.select_dataset'
  const isRemoteStarWorkspace = item.handlerId === 'remote.star_inference'
  const isRemoteTrtWorkspace = item.handlerId === 'remote.build_tensorrt'
  const isLocalStarPackageWorkspace = item.handlerId === 'model.package_star'
  const currentTask = isYoloWorkspace && yoloMode === 'training' ? undefined : props.activeTasks.find((task) => task.functionId === item.id && !task.name.startsWith('YOLO 训练 ·'))
  const visibleParameters = item.parameters.filter((field) =>
    (!field.visibleWhen
      || (field.visibleWhen.equals !== undefined && props.values.parameters[field.visibleWhen.fieldId] === field.visibleWhen.equals)
      || (field.visibleWhen.notEquals !== undefined && props.values.parameters[field.visibleWhen.fieldId] !== field.visibleWhen.notEquals))
    && (!isRemoteStarWorkspace || !['labels', 'conf', 'additional_models', 'star_models'].includes(field.id)))
  const visiblePathFields = item.pathFields.filter((field) => !isRemoteStarWorkspace || field.id !== 'local_model_file')
  const remoteYoloPaths = item.handlerId === 'yolo.split_dataset' && Boolean(
    String(props.values.parameters.remote_host || '').trim()
    || String(props.values.parameters.remote_username || '').trim()
    || String(props.values.parameters.remote_password || ''),
  )
  return (
    <main className="function-workspace">
      <header className="workspace-header"><div><h1>{item.name}<button aria-label="编辑功能" onClick={props.onEdit}><icons.Pencil size={19} /></button></h1><p>{item.description || '在这里配置数据路径和运行参数'}</p></div><button className="outline-button" onClick={props.onEdit}><icons.Pencil size={17} />编辑功能</button></header>
      {currentTask ? <section className={`active-task-banner ${currentTask.status}`} role="status">
        <span><i /><strong>{currentTask.status === 'stopping' ? '正在终止任务' : '任务正在运行'}</strong><em>“{currentTask.name}”正在运行；其他功能仍可独立使用。</em></span>
        <button className="stop-run-button" onClick={() => props.onStop(currentTask.id)} disabled={currentTask.status === 'stopping'}><icons.X size={17} />{currentTask.status === 'stopping' ? '正在终止…' : '终止运行'}</button>
      </section> : null}
      <WorkflowTodoPanel items={props.workflowTodos} onApply={props.onApplyWorkflowTodo} onDelete={props.onDeleteWorkflowTodo} />
      {isYoloWorkspace ? <nav className="yolo-mode-tabs" aria-label="YOLO 数据集操作">
        <button className={yoloMode === 'dataset' ? 'active' : ''} onClick={() => setYoloMode('dataset')}><icons.Folder size={18} /><span>数据集划分<small>整理数据并生成 data.yaml</small></span></button>
        <button className={yoloMode === 'training' ? 'active' : ''} onClick={() => setYoloMode('training')}><icons.Play size={18} /><span>模型训练<small>配置参数、保存场景并启动训练</small></span></button>
      </nav> : null}
      {(!isYoloWorkspace || yoloMode === 'dataset') && !isCalibrationWorkspace && !isRemoteTrtWorkspace && !isLocalStarPackageWorkspace ? <>
      <section className="workspace-section">
        <div className="section-title"><icons.Folder size={21} /><h2>数据路径</h2></div>
        <div className="path-fields">
          {visiblePathFields.map((field) => <label className="path-field" key={field.id}><span>{field.label}</span><div><input value={props.values.paths[field.id] || ''} onChange={(event) => props.onPathChange(field.id, event.target.value)} placeholder={remoteYoloPaths ? '请输入远程服务器上的绝对路径' : `请选择或粘贴${field.mode === 'file' ? '文件' : '文件夹'}路径`} /><button onClick={() => props.onBrowse(field.id, field.mode)} disabled={remoteYoloPaths} title={remoteYoloPaths ? '远程模式请手动填写服务器路径' : undefined}>{remoteYoloPaths ? '远程路径' : '浏览'}</button></div></label>)}
          {!visiblePathFields.length ? <p className="section-empty">尚未定义数据路径，接入功能时再添加。</p> : null}
        </div>
        {!isRemoteStarWorkspace ? <button className="text-action" onClick={props.onAddPath}><icons.Plus size={17} />添加路径</button> : null}
      </section>
      <section className="workspace-section parameter-section">
        <div className="section-title"><icons.SlidersHorizontal size={21} /><h2>运行参数</h2></div>
        {visibleParameters.length ? <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{visibleParameters.map((field) => <label className="parameter-row" key={field.id}><span>{field.label}</span><ParameterValue field={field} value={props.values.parameters[field.id] ?? field.default} onChange={(value) => props.onParameterChange(field.id, value)} /></label>)}</div> : <p className="section-empty parameter-empty">当前没有运行参数。具体功能确定后，可通过“添加参数”配置。</p>}
        {!isRemoteStarWorkspace ? <button className="text-action" onClick={props.onAddParameter}><icons.Plus size={17} />添加参数</button> : null}
      </section>
      </> : null}
        {item.handlerId === 'remote.star_inference' ? <RemoteInferencePanel itemId={item.id} values={props.values} onParameterChange={props.onParameterChange} platformTask={currentTask || null} onCompleted={() => props.onWorkflowComplete('remote.star_inference')} /> : item.handlerId === 'local.pt_inference' ? <LocalPTInferencePanel itemId={item.id} values={props.values} platformTask={currentTask || null} /> : isRemoteTrtWorkspace ? <RemoteTensorRTBuildPanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} /> : isLocalStarPackageWorkspace ? <LocalStarPackagePanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onBrowse={props.onBrowse} onPathChange={props.onPathChange} onParameterChange={props.onParameterChange} onRun={props.onRun} /> : isCalibrationWorkspace ? <RemoteCalibrationPanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} /> : isYoloWorkspace ? (yoloMode === 'dataset' ? <RemoteYoloDatasetPanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} /> : <YoloTrainingPanel values={props.values} onParameterChange={props.onParameterChange} recommendation={props.yoloTrainingRecommendation} onCompleted={() => props.onWorkflowComplete('yolo.split_dataset')} />) : <section className="run-section">
        <div className="run-actions"><button className="run-button" onClick={() => props.onRun()} disabled={props.running || Boolean(currentTask)}><icons.Play size={18} fill="currentColor" />{currentTask ? '正在运行' : props.running ? '正在启动' : '一键运行'}</button><span className={item.handlerReady ? 'ready' : ''}><icons.Info size={17} />{item.handlerReady ? '处理逻辑已接入' : '处理逻辑尚未接入'}</span></div>
        <details className="run-output" open><summary><icons.ChevronRight size={18} />运行输出</summary><pre>{props.output || '运行后将在这里显示处理进度和结果'}</pre></details>
      </section>}
      <FunctionRunHistory key={item.id} functionId={item.id} />
    </main>
  )
}
