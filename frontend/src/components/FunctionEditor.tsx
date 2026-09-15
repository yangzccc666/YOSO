import { useEffect, useState } from 'react'
import { icons } from '../icons'
import type { FunctionDefinition, FunctionParameter, PathField } from '../types'

type Props = {
  item: FunctionDefinition | null
  onClose: () => void
  onSave: (item: FunctionDefinition) => Promise<void>
  onDelete: (item: FunctionDefinition) => Promise<void>
}

const makePath = (): PathField => ({ id: `path_${crypto.randomUUID().slice(0, 8)}`, label: '新路径', mode: 'directory' })
const makeParameter = (): FunctionParameter => ({ id: `param_${crypto.randomUUID().slice(0, 8)}`, label: '新参数', type: 'text', default: '', options: [] })

export function FunctionEditor({ item, onClose, onSave, onDelete }: Props) {
  const [draft, setDraft] = useState<FunctionDefinition | null>(item)
  const [saving, setSaving] = useState(false)
  useEffect(() => setDraft(item ? structuredClone(item) : null), [item])
  if (!draft) return null

  const updatePath = (index: number, changes: Partial<PathField>) => setDraft((current) => current ? ({ ...current, pathFields: current.pathFields.map((field, fieldIndex) => fieldIndex === index ? { ...field, ...changes } : field) }) : current)
  const updateParam = (index: number, changes: Partial<FunctionParameter>) => setDraft((current) => current ? ({ ...current, parameters: current.parameters.map((field, fieldIndex) => fieldIndex === index ? { ...field, ...changes } : field) }) : current)
  const save = async () => { setSaving(true); try { await onSave(draft) } finally { setSaving(false) } }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <section className="editor-modal" role="dialog" aria-modal="true" aria-labelledby="editor-title">
        <header><div><h2 id="editor-title">编辑功能</h2><p>设置名称，以及运行时需要填写的路径和参数。</p></div><button className="icon-button" onClick={onClose} aria-label="关闭"><icons.X size={20} /></button></header>
        <div className="editor-body">
          <label className="editor-field"><span>功能名称</span><input autoFocus value={draft.name} onChange={(event) => setDraft({ ...draft, name: event.target.value })} /></label>
          <label className="editor-field"><span>功能说明</span><input value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} placeholder="简要说明这个功能做什么" /></label>

          <div className="editor-section-title"><div><h3>数据路径</h3><p>定义运行时需要选择的文件或文件夹。</p></div><button onClick={() => setDraft({ ...draft, pathFields: [...draft.pathFields, makePath()] })}><icons.Plus size={15} />添加</button></div>
          <div className="definition-list">
            {draft.pathFields.map((field, index) => (
              <div className="definition-row" key={field.id}>
                <input value={field.label} onChange={(event) => updatePath(index, { label: event.target.value })} aria-label="路径名称" />
                <select value={field.mode} onChange={(event) => updatePath(index, { mode: event.target.value as PathField['mode'] })} aria-label="路径类型"><option value="directory">文件夹</option><option value="file">文件</option></select>
                <button className="remove-button" onClick={() => setDraft({ ...draft, pathFields: draft.pathFields.filter((_, fieldIndex) => fieldIndex !== index) })} aria-label="删除路径"><icons.Trash2 size={16} /></button>
              </div>
            ))}
            {!draft.pathFields.length ? <p className="definition-empty">暂未定义路径</p> : null}
          </div>

          <div className="editor-section-title"><div><h3>运行参数</h3><p>参数会自动显示在功能运行区。</p></div><button onClick={() => setDraft({ ...draft, parameters: [...draft.parameters, makeParameter()] })}><icons.Plus size={15} />添加</button></div>
          <div className="definition-list">
            {draft.parameters.map((field, index) => (
              <div className="definition-row parameter-definition" key={field.id}>
                <input value={field.label} onChange={(event) => updateParam(index, { label: event.target.value })} aria-label="参数名称" />
                <select value={field.type} onChange={(event) => updateParam(index, { type: event.target.value as FunctionParameter['type'], default: event.target.value === 'boolean' ? false : '' })} aria-label="参数类型"><option value="text">文本</option><option value="password">密码</option><option value="number">数字</option><option value="boolean">开关</option><option value="select">选项</option></select>
                <button className="remove-button" onClick={() => setDraft({ ...draft, parameters: draft.parameters.filter((_, fieldIndex) => fieldIndex !== index) })} aria-label="删除参数"><icons.Trash2 size={16} /></button>
                {field.type === 'select' ? <input className="option-input" value={field.options.join('，')} onChange={(event) => updateParam(index, { options: event.target.value.split(/[，,]/).map((value) => value.trim()).filter(Boolean) })} placeholder="选项一，选项二" aria-label="选项内容" /> : null}
              </div>
            ))}
            {!draft.parameters.length ? <p className="definition-empty">暂未定义参数，接入功能时再添加即可</p> : null}
          </div>
        </div>
        <footer><button className="delete-button" onClick={() => onDelete(draft)}><icons.Trash2 size={16} />删除功能</button><div><button className="cancel-button" onClick={onClose}>取消</button><button className="save-button" disabled={saving || !draft.name.trim()} onClick={save}><icons.Check size={16} />保存修改</button></div></footer>
      </section>
    </div>
  )
}
