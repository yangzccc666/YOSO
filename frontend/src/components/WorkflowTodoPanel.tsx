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
  const appliedTargets = new Set(
    items.filter((item) => item.status === 'applied').map((item) => item.targetHandlerId),
  )
  return <section className="workflow-todos" aria-label="工作流待办">
    <div className="workflow-todos-heading"><span><icons.Clock3 size={18} /><strong>工作流待办</strong><em>{items.length}</em></span><small>前置步骤的结果保存在这里，不会自动覆盖当前配置。</small></div>
    <div className="workflow-todo-list">{items.map((item) => {
      const sameTargetIsApplied = item.status !== 'applied' && appliedTargets.has(item.targetHandlerId)
      const buttonTitle = sameTargetIsApplied ? '同一目标功能已有待办等待运行；运行成功后即可应用此待办' : undefined
      return <article className={`workflow-todo ${item.status}`} key={item.id}>
        <div><span>{item.status === 'applied' ? '已应用配置，等待目标功能运行' : item.sourceName}</span><strong>{item.title}</strong><p>{item.description}</p><small>{formattedTime(item.createdAt)}</small></div>
        <div className="workflow-todo-actions">
          <button type="button" className="workflow-apply" disabled={item.status === 'applied' || sameTargetIsApplied} title={buttonTitle} onClick={() => onApply(item)}><icons.Check size={15} />{item.status === 'applied' ? '等待运行' : sameTargetIsApplied ? '同功能待运行' : '应用到本功能'}</button>
          <button type="button" className="workflow-delete" onClick={() => onDelete(item.id)} aria-label={`删除待办 ${item.title}`}><icons.Trash2 size={15} />删除</button>
        </div>
      </article>
    })}</div>
    {appliedTargets.size ? <p className="workflow-todo-hint">已应用的待办只会暂时锁定同一个目标功能；其他功能的待办可以立即应用。目标功能成功运行后，该待办会自动删除并解除锁定。</p> : null}
  </section>
}
