import type { ConnectionTestResult, FunctionDefinition, PlatformTaskStatus, RemoteCredentialStatus, RemoteInferenceStatus, RunHistoryRecord, WorkingValues, WorkspaceData, YoloConnectionTestResult, YoloTrainingProfile, YoloTrainingProfilesPayload, YoloTrainingSession, YoloTrainingSummary, YoloTrainingValues } from './types'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  const data = await response.json()
  if (!response.ok) throw new Error(data.error || '请求失败')
  return data as T
}

export async function getFunctions(): Promise<FunctionDefinition[]> {
  const data = await request<{ functions: FunctionDefinition[] }>('/api/functions')
  return data.functions
}

export async function getWorkspace(): Promise<WorkspaceData> {
  return request<WorkspaceData>('/api/workspace')
}

export async function createGroup(name: string): Promise<WorkspaceData> {
  return request<WorkspaceData>('/api/groups', { method: 'POST', body: JSON.stringify({ name }) })
}

export async function renameGroup(groupId: string, name: string): Promise<WorkspaceData> {
  return request<WorkspaceData>(`/api/groups/${groupId}`, { method: 'PUT', body: JSON.stringify({ name }) })
}

export async function deleteGroup(groupId: string): Promise<WorkspaceData> {
  return request<WorkspaceData>(`/api/groups/${groupId}`, { method: 'DELETE' })
}

export async function reorderGroups(groupIds: string[]): Promise<WorkspaceData> {
  return request<WorkspaceData>('/api/groups/reorder', { method: 'POST', body: JSON.stringify({ groupIds }) })
}

export async function moveFunction(itemId: string, groupId: string | null, position?: number): Promise<WorkspaceData> {
  return request<WorkspaceData>(`/api/functions/${itemId}/move`, {
    method: 'POST',
    body: JSON.stringify({ groupId, position }),
  })
}

export async function createFunction(): Promise<FunctionDefinition> {
  return request<FunctionDefinition>('/api/functions', { method: 'POST', body: '{}' })
}

export async function updateFunction(item: FunctionDefinition): Promise<FunctionDefinition> {
  return request<FunctionDefinition>(`/api/functions/${item.id}`, { method: 'PUT', body: JSON.stringify(item) })
}

export async function deleteFunction(itemId: string): Promise<void> {
  await request(`/api/functions/${itemId}`, { method: 'DELETE' })
}

export type RunResponse = {
  ok: boolean
  result: { message?: string; outputFolders?: string[]; [key: string]: unknown }
  messages: string[]
}

export async function runFunction(itemId: string, values: WorkingValues): Promise<RunResponse> {
  return request(`/api/functions/${itemId}/run`, { method: 'POST', body: JSON.stringify(values) })
}

export async function getActiveTask(): Promise<PlatformTaskStatus[]> {
  const data = await request<{ active: PlatformTaskStatus[] }>('/api/tasks/active')
  return data.active
}

export async function getRunHistory(): Promise<RunHistoryRecord[]> {
  const data = await request<{ history: RunHistoryRecord[] }>('/api/tasks/history')
  return data.history
}

export async function stopActiveTask(taskId: string): Promise<PlatformTaskStatus> {
  const data = await request<{ active: PlatformTaskStatus }>(`/api/tasks/${taskId}/stop`, {
    method: 'POST',
    body: '{}',
  })
  return data.active
}

export async function choosePaths(mode: string): Promise<string[]> {
  const data = await request<{ paths: string[] }>('/api/dialog', { method: 'POST', body: JSON.stringify({ mode }) })
  return data.paths
}

export async function startRemoteInference(values: WorkingValues): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>('/api/remote-inference/start', {
    method: 'POST',
    body: JSON.stringify(values),
  })
}

export async function startLocalPTInference(values: WorkingValues): Promise<RemoteInferenceStatus> {
  return request('/api/local-pt-inference/start', { method: 'POST', body: JSON.stringify(values) })
}

export async function getLocalPTInferenceStatus(id: string): Promise<RemoteInferenceStatus> {
  return request(`/api/local-pt-inference/${id}/status`)
}

export async function stopLocalPTInference(id: string): Promise<RemoteInferenceStatus> {
  return request(`/api/local-pt-inference/${id}/stop`, { method: 'POST', body: '{}' })
}

export async function seekLocalPTInference(id: string, seconds: number): Promise<void> {
  await request(`/api/local-pt-inference/${id}/seek`, { method: 'POST', body: JSON.stringify({ seconds }) })
}

export async function setLocalPTInferencePaused(id: string, paused: boolean): Promise<RemoteInferenceStatus> {
  return request(`/api/local-pt-inference/${id}/playback`, { method: 'POST', body: JSON.stringify({ paused }) })
}

export async function testRemoteConnection(parameters: WorkingValues['parameters']): Promise<ConnectionTestResult> {
  return request<ConnectionTestResult>('/api/remote-inference/test-connection', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function getRemoteCredentialStatus(parameters: WorkingValues['parameters']): Promise<RemoteCredentialStatus> {
  return request<RemoteCredentialStatus>('/api/remote-inference/credential-status', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function forgetRemotePassword(parameters: WorkingValues['parameters']): Promise<{ ok: true; removed: boolean; message: string }> {
  return request('/api/remote-inference/forget-password', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function testYoloRemoteConnection(parameters: WorkingValues['parameters']): Promise<YoloConnectionTestResult> {
  return request<YoloConnectionTestResult>('/api/yolo-dataset/test-connection', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function getYoloRemoteCredentialStatus(parameters: WorkingValues['parameters']): Promise<RemoteCredentialStatus> {
  return request<RemoteCredentialStatus>('/api/yolo-dataset/credential-status', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function forgetYoloRemotePassword(parameters: WorkingValues['parameters']): Promise<{ ok: true; removed: boolean; message: string }> {
  return request('/api/yolo-dataset/forget-password', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function getYoloTrainingProfiles(): Promise<YoloTrainingProfilesPayload> {
  return request<YoloTrainingProfilesPayload>('/api/yolo-training/profiles')
}

export async function createYoloTrainingProfile(payload: { name: string; description: string; values: YoloTrainingValues }): Promise<YoloTrainingProfile> {
  return request<YoloTrainingProfile>('/api/yolo-training/profiles', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export async function updateYoloTrainingProfile(profileId: string, payload: { name: string; description: string; values: YoloTrainingValues }): Promise<YoloTrainingProfile> {
  return request<YoloTrainingProfile>(`/api/yolo-training/profiles/${profileId}`, {
    method: 'PUT',
    body: JSON.stringify(payload),
  })
}

export async function deleteYoloTrainingProfile(profileId: string): Promise<void> {
  await request(`/api/yolo-training/profiles/${profileId}`, { method: 'DELETE' })
}

export async function startYoloTraining(parameters: YoloTrainingValues): Promise<YoloTrainingSession> {
  return request<YoloTrainingSession>('/api/yolo-training/start', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
}

export async function getYoloTrainingStatus(sessionId: string): Promise<YoloTrainingSession> {
  return request<YoloTrainingSession>(`/api/yolo-training/${sessionId}/status`)
}

export async function getYoloTrainingSessions(): Promise<YoloTrainingSummary[]> {
  const data = await request<{ sessions: YoloTrainingSummary[] }>('/api/yolo-training/sessions')
  return data.sessions
}

export async function deleteYoloTrainingSession(sessionId: string): Promise<void> {
  await request(`/api/yolo-training/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function stopYoloTraining(sessionId: string): Promise<YoloTrainingSession> {
  return request<YoloTrainingSession>(`/api/yolo-training/${sessionId}/stop`, {
    method: 'POST',
    body: '{}',
  })
}

export async function reconnectYoloTraining(sessionId: string, parameters: YoloTrainingValues): Promise<YoloTrainingSession> {
  return request<YoloTrainingSession>(`/api/yolo-training/${sessionId}/reconnect`, {
    method: 'POST', body: JSON.stringify({ parameters }),
  })
}

export async function getRemoteInferenceStatus(sessionId: string): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>(`/api/remote-inference/${sessionId}/status`)
}

export async function stopRemoteInference(sessionId: string): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>(`/api/remote-inference/${sessionId}/stop`, {
    method: 'POST',
    body: '{}',
  })
}

export async function seekRemoteInference(sessionId: string, seconds: number): Promise<{ ok: true; requestedSeconds: number }> {
  return request<{ ok: true; requestedSeconds: number }>(`/api/remote-inference/${sessionId}/seek`, {
    method: 'POST',
    body: JSON.stringify({ seconds }),
  })
}

export async function setRemoteInferencePaused(sessionId: string, paused: boolean): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>(`/api/remote-inference/${sessionId}/playback`, {
    method: 'POST',
    body: JSON.stringify({ paused }),
  })
}
