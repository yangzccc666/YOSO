/** Shared guards: callers must check before changing state or sending requests. */
export function confirmStop(name: string, detail = '已完成的输出会保留；继续运行需要重新启动任务。'): boolean {
  return window.confirm(`确定停止“${name}”吗？\n${detail}\n\n点击“取消”将保持当前任务运行。`)
}

export function confirmClose(name: string, detail = '关闭后不会停止正在运行的任务。'): boolean {
  return window.confirm(`确定关闭“${name}”吗？\n${detail}`)
}
