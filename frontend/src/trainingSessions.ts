import { REMEMBERED_PASSWORD_VALUE } from './types'
import type { WorkingValues, YoloTrainingSummary, YoloTrainingValues } from './types'

/** Saved sessions can belong to a different device than the current form. */
export function trainingSessionConnection(current: WorkingValues['parameters'], session: Pick<YoloTrainingSummary, 'host' | 'port' | 'username'>): YoloTrainingValues {
  const same = String(current.remote_host || '').trim().toLowerCase() === session.host.toLowerCase()
    && Number(current.remote_port || 22) === session.port
    && String(current.remote_username || '').trim() === session.username
  return { remote_host: session.host, remote_port: session.port, remote_username: session.username,
    remote_password: same ? current.remote_password || '' : REMEMBERED_PASSWORD_VALUE,
    remember_password: same ? current.remember_password !== false : true }
}

export function trainingSessionFailedWithOOM(session: Pick<YoloTrainingSummary, 'status' | 'error' | 'message'>): boolean {
  const text = `${session.error || ''}\n${session.message}`
  return session.status === 'failed' && !/不能仅凭退出码确认 OOM/.test(text)
    && /\boom\b|out\s+of\s+memory|outofmemoryerror|内存.*不足|显存.*不足/i.test(text)
}
