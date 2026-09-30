export type WorkflowTodoPayload = {
  paths?: Record<string, string>
  parameters?: Record<string, string | number | boolean>
  training?: { data: string; project: string; runName?: string }
  starModel?: {
    path: string
    labels: string
    conf?: number
  }
}

export type WorkflowTodo = {
  id: string
  sourceFunctionId: string
  sourceName: string
  targetHandlerId: string
  title: string
  description: string
  createdAt: string
  status: 'pending' | 'applied'
  payload: WorkflowTodoPayload
}

const storageKey = 'yolo-data-platform:workflow-todos:v1'
const limit = 100

export function loadWorkflowTodos(): WorkflowTodo[] {
  try {
    const parsed: unknown = JSON.parse(localStorage.getItem(storageKey) || '[]')
    if (!Array.isArray(parsed)) return []
    return parsed.flatMap((item): WorkflowTodo[] => {
      if (item === null || typeof item !== 'object') return []
      const candidate = item as Partial<WorkflowTodo>
      if (!candidate.id || !candidate.targetHandlerId || !candidate.payload) return []
      return [{
        id: String(candidate.id),
        sourceFunctionId: String(candidate.sourceFunctionId || ''),
        sourceName: String(candidate.sourceName || '上一步功能'),
        targetHandlerId: String(candidate.targetHandlerId),
        title: String(candidate.title || '待应用的工作流结果'),
        description: String(candidate.description || ''),
        createdAt: String(candidate.createdAt || new Date().toISOString()),
        status: candidate.status === 'applied' ? 'applied' : 'pending',
        payload: candidate.payload,
      }]
    }).slice(0, limit)
  } catch {
    return []
  }
}

export function saveWorkflowTodos(items: WorkflowTodo[]): void {
  try { localStorage.setItem(storageKey, JSON.stringify(items.slice(0, limit))) } catch { /* keep current session state */ }
}

export function createWorkflowTodo(input: Omit<WorkflowTodo, 'id' | 'createdAt' | 'status'>): WorkflowTodo {
  return {
    ...input,
    id: crypto.randomUUID(),
    createdAt: new Date().toISOString(),
    status: 'pending',
  }
}
