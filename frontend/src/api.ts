import type { ConnectionTestResult, FunctionDefinition, RemoteInferenceStatus, WorkingValues, WorkspaceData } from './types'

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

export async function testRemoteConnection(parameters: WorkingValues['parameters']): Promise<ConnectionTestResult> {
  return request<ConnectionTestResult>('/api/remote-inference/test-connection', {
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

export async function getRemoteInferenceStatus(sessionId: string): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>(`/api/remote-inference/${sessionId}/status`)
}

export async function stopRemoteInference(sessionId: string): Promise<RemoteInferenceStatus> {
  return request<RemoteInferenceStatus>(`/api/remote-inference/${sessionId}/stop`, {
    method: 'POST',
    body: '{}',
  })
}
