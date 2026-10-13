import { icons } from '../icons'
import type { WorkflowTodo } from '../workflowTodos'
import { WorkflowTodoPanel } from './WorkflowTodoPanel'

type Props = {
  items: WorkflowTodo[]
  onApply: (item: WorkflowTodo) => void
  onDelete: (id: string) => void
  onClose: () => void
}

export function WorkflowTodoDialog({ items, onApply, onDelete, onClose }: Props) {
  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="history-modal workflow-todo-modal" role="dialog" aria-modal="true" aria-labelledby="workflow-todo-title">
      <header><div><h2 id="workflow-todo-title">工作流待办</h2><p>前置步骤产生的路径和参数不会直接覆盖目标功能。</p></div><button className="icon-button" onClick={onClose} aria-label="关闭工作流待办"><icons.X size={19} /></button></header>
      <div className="history-content">
        {items.length ? <WorkflowTodoPanel items={items} onApply={onApply} onDelete={onDelete} /> : <div className="workflow-todo-empty"><icons.Check size={28} /><strong>没有待处理的工作流结果</strong><span>完成前置步骤后，候选路径和参数会出现在这里。</span></div>}
      </div>
      <footer><span>不同目标功能的待办可以连续应用；同一功能会保留当前配置，运行成功后自动解锁下一条。</span><button onClick={onClose}>关闭</button></footer>
    </section>
  </div>
}
