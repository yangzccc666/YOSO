import { icons } from '../icons'
import type { WorkflowTodo } from '../workflowTodos'

type Props = {
  items: WorkflowTodo[]
  onApply: (item: WorkflowTodo) => void
  onDelete: (id: string) => void
}

function formattedTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

export function WorkflowTodoPanel({ items, onApply, onDelete }: Props) {
  if (!items.length) return null
  const hasApplied = items.some((item) => item.status === 'applied')
  return <section className="workflow-todos" aria-label="工作流待办">
    <div className="workflow-todos-heading"><span><icons.Clock3 size={18} /><strong>工作流待办</strong><em>{items.length}</em></span><small>前置步骤的结果保存在这里，不会自动覆盖当前配置。</small></div>
    <div className="workflow-todo-list">{items.map((item) => <article className={`workflow-todo ${item.status}`} key={item.id}>
      <div><span>{item.status === 'applied' ? '已应用，等待运行' : item.sourceName}</span><strong>{item.title}</strong><p>{item.description}</p><small>{formattedTime(item.createdAt)}</small></div>
      <div className="workflow-todo-actions">
        <button type="button" className="workflow-apply" disabled={item.status === 'applied' || hasApplied} onClick={() => onApply(item)}><icons.Check size={15} />{item.status === 'applied' ? '已应用' : '应用到本功能'}</button>
        <button type="button" className="workflow-delete" onClick={() => onDelete(item.id)} aria-label={`删除待办 ${item.title}`}><icons.Trash2 size={15} />删除</button>
      </div>
    </article>)}</div>
    {hasApplied ? <p className="workflow-todo-hint">当前有一条待办已应用。运行本功能成功后会自动删除它，之后即可应用下一条。</p> : null}
  </section>
}
