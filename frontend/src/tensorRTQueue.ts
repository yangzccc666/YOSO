import { REMEMBERED_PASSWORD_VALUE } from './types'
import type { TensorRTQueueItem, TensorRTQueueStatus, WorkingValues } from './types'

/** Never send another device's typed password while reattaching a saved task. */
export function tensorRTReconnectParameters(current: WorkingValues['parameters'], item: TensorRTQueueItem): WorkingValues['parameters'] {
  const sameDevice = String(current.host || '').trim().toLowerCase() === item.host.toLowerCase()
    && Number(current.port || 22) === item.port && String(current.username || '').trim() === item.username
  return { host: item.host, port: item.port, username: item.username,
    password: sameDevice ? current.password || '' : REMEMBERED_PASSWORD_VALUE,
    remember_password: sameDevice ? current.remember_password !== false : true }
}

/** A slow poll must not undo a newly submitted job or a pause/resume action. */
export function retainTensorRTQueue(current: TensorRTQueueStatus | null, next: TensorRTQueueStatus): TensorRTQueueStatus {
  return current && current.instanceId === next.instanceId && current.version >= next.version ? current : next
}

/** Do not present successful enqueueing as successful execution. */
export function tensorRTDispatchStatus(queue: TensorRTQueueStatus | null): { title: string; detail: string; blocked: boolean } {
  if (!queue) return { title: '正在读取构建队列', detail: '读取实际调度状态后再显示执行信息。', blocked: false }
  if (queue.closed) return { title: '队列已关闭，未执行等待任务', detail: '请重新打开平台。', blocked: true }
  if (queue.disconnectedCount) return { title: '后续尚未执行：需先重连确认', detail: queue.pauseReason || '确认原远端任务状态后将自动继续，避免重复构建。', blocked: true }
  if (queue.paused) return { title: queue.runningCount ? '当前任务运行中，后续已暂停' : '队列已暂停：等待任务不会执行', detail: queue.pauseReason || '点击继续队列，或加入新任务以恢复自动执行。', blocked: true }
  if (queue.runningCount) return { title: '自动执行中', detail: '当前任务完成后，等待任务将按加入顺序自动开始，无需再次点击。', blocked: false }
  if (queue.queuedCount) return { title: '自动执行已开启，等待调度', detail: '即将按顺序开始；当前没有任务正在构建。', blocked: false }
  return { title: '队列空闲', detail: '加入构建队列后会自动执行。', blocked: false }
}
