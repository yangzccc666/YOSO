import { icons } from '../icons'
import { RemoteInferencePanel } from './RemoteInferencePanel'
import type { FunctionDefinition, FunctionParameter, WorkingValues } from '../types'

type Props = {
  item: FunctionDefinition | null
  values: WorkingValues
  output: string
  running: boolean
  onEdit: () => void
  onBrowse: (fieldId: string, mode: string) => void
  onPathChange: (fieldId: string, value: string) => void
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  onAddPath: () => void
  onAddParameter: () => void
  onRun: () => void
}

function ParameterValue({ field, value, onChange }: { field: FunctionParameter; value: string | number | boolean; onChange: (value: string | number | boolean) => void }) {
  if (field.type === 'boolean') return <label className="switch-control"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} /><span /><em>{value ? '开启' : '关闭'}</em></label>
  if (field.type === 'select') return <span className="workspace-select"><select value={String(value)} onChange={(event) => onChange(event.target.value)}><option value="">请选择</option>{field.options.map((option) => <option value={option} key={option}>{option}</option>)}</select><icons.ChevronDown size={16} /></span>
  return <input type={field.type === 'number' ? 'number' : field.type === 'password' ? 'password' : 'text'} autoComplete={field.type === 'password' ? 'new-password' : undefined} value={String(value)} placeholder={field.type === 'password' ? '首次连接或未记住时输入' : '请输入参数值'} onChange={(event) => onChange(field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
}

export function FunctionWorkspace(props: Props) {
  if (!props.item) return <main className="empty-workspace"><span><icons.Box size={34} /></span><h2>暂无可用功能</h2><p>后续接入新的处理脚本后，功能会显示在左侧分组中。</p></main>
  const item = props.item
  const visibleParameters = item.parameters.filter((field) => !field.visibleWhen || props.values.parameters[field.visibleWhen.fieldId] === field.visibleWhen.equals)
  return (
    <main className="function-workspace">
      <header className="workspace-header"><div><h1>{item.name}<button aria-label="编辑功能" onClick={props.onEdit}><icons.Pencil size={19} /></button></h1><p>{item.description || '在这里配置数据路径和运行参数'}</p></div><button className="outline-button" onClick={props.onEdit}><icons.Pencil size={17} />编辑功能</button></header>
      <section className="workspace-section">
        <div className="section-title"><icons.Folder size={21} /><h2>数据路径</h2></div>
        <div className="path-fields">
          {item.pathFields.map((field) => <label className="path-field" key={field.id}><span>{field.label}</span><div><input value={props.values.paths[field.id] || ''} onChange={(event) => props.onPathChange(field.id, event.target.value)} placeholder={`请选择或粘贴${field.mode === 'file' ? '文件' : '文件夹'}路径`} /><button onClick={() => props.onBrowse(field.id, field.mode)}>浏览</button></div></label>)}
          {!item.pathFields.length ? <p className="section-empty">尚未定义数据路径，接入功能时再添加。</p> : null}
        </div>
        <button className="text-action" onClick={props.onAddPath}><icons.Plus size={17} />添加路径</button>
      </section>
      <section className="workspace-section parameter-section">
        <div className="section-title"><icons.SlidersHorizontal size={21} /><h2>运行参数</h2></div>
        {visibleParameters.length ? <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{visibleParameters.map((field) => <label className="parameter-row" key={field.id}><span>{field.label}</span><ParameterValue field={field} value={props.values.parameters[field.id] ?? field.default} onChange={(value) => props.onParameterChange(field.id, value)} /></label>)}</div> : <p className="section-empty parameter-empty">当前没有运行参数。具体功能确定后，可通过“添加参数”配置。</p>}
        <button className="text-action" onClick={props.onAddParameter}><icons.Plus size={17} />添加参数</button>
      </section>
      {item.handlerId === 'remote.star_inference' ? <RemoteInferencePanel itemId={item.id} values={props.values} onParameterChange={props.onParameterChange} /> : <section className="run-section">
        <div className="run-actions"><button className="run-button" onClick={props.onRun} disabled={props.running}><icons.Play size={18} fill="currentColor" />{props.running ? '正在运行' : '一键运行'}</button><span className={item.handlerReady ? 'ready' : ''}><icons.Info size={17} />{item.handlerReady ? '处理逻辑已接入' : '处理逻辑尚未接入'}</span></div>
        <details className="run-output" open><summary><icons.ChevronRight size={18} />运行输出</summary><pre>{props.output || '运行后将在这里显示处理进度和结果'}</pre></details>
      </section>}
    </main>
  )
}
