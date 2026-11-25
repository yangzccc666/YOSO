import type { ConnectionTestResult, FunctionDefinition, PlatformSettings, PlatformTaskStatus, RemoteCredentialStatus, RemoteInferenceStatus, RunHistoryRecord, SshConnectionProfile, StarLabelDetectionResult, WorkingValues, WorkspaceData, YoloConnectionTestResult, YoloTrainingProfile, YoloTrainingProfilesPayload, YoloTrainingSession, YoloTrainingSummary, YoloTrainingValues } from './types'
import type { StarModel, StarModelHistoryRecord } from './remoteStarModels'
import { createSetupGuidanceMonitor, SETUP_GUIDANCE_EVENT } from './setupGuidance'

export const getPlatformDependencies = () => request<import('./types').PlatformDependencyReport>('/api/dependencies')
export const getTensorRTQueue = () => request<import('./types').TensorRTQueueStatus>('/api/tensorrt-build/queue')
export const enqueueTensorRTBuild = (functionId: string, values: WorkingValues) => request<import('./types').TensorRTQueueSubmission>('/api/tensorrt-build/queue', {
  method: 'POST', body: JSON.stringify({ functionId, ...values }),
})
export const pauseTensorRTQueue = (paused: boolean, parameters?: WorkingValues['parameters']) => request<import('./types').TensorRTQueueStatus>('/api/tensorrt-build/queue/pause', { method: 'POST', body: JSON.stringify({ paused, parameters }) })
export const clearTensorRTWaiting = () => request<import('./types').TensorRTQueueStatus>('/api/tensorrt-build/queue/clear', { method: 'POST', body: '{}' })
export const removeTensorRTQueueItem = (id: string, forget = false) => request<import('./types').TensorRTQueueStatus>(`/api/tensorrt-build/queue/${encodeURIComponent(id)}${forget ? '?forget=true' : ''}`, { method: 'DELETE' })
export const stopTensorRTQueueItem = (id: string) => request<import('./types').TensorRTQueueStatus>(`/api/tensorrt-build/queue/${encodeURIComponent(id)}/stop`, { method: 'POST', body: '{}' })
export const reconnectTensorRTQueueItem = (id: string, parameters: WorkingValues['parameters']) => request<import('./types').TensorRTQueueStatus>(`/api/tensorrt-build/queue/${encodeURIComponent(id)}/reconnect`, { method: 'POST', body: JSON.stringify({ parameters }) })
export const discoverTensorRTBuilds = (functionId: string, parameters: WorkingValues['parameters']) => request<import('./types').TensorRTQueueStatus & { message: string }>('/api/tensorrt-build/queue/discover', { method: 'POST', body: JSON.stringify({ functionId, parameters }) })
export const getStarModelHistory = () => request<{ models: StarModelHistoryRecord[] }>('/api/star-model-history')
export const saveStarModelHistory = (models: StarModel[]) => request<{ models: StarModelHistoryRecord[] }>('/api/star-model-history', {
  method: 'POST', body: JSON.stringify({ models }),
})
export const deleteStarModelHistory = (id: string) => request<{ ok: boolean }>(`/api/star-model-history/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const INFERENCE_VIDEOS_UPDATED_EVENT = 'yolo-inference-videos-updated'
export const getInferenceVideos = () => request<{ videos: import('./inferenceVideos').InferenceVideoRecord[] }>('/api/inference-video-history')
export const saveInferenceVideo = (path: string, scene: string, station: string) => request<{ videos: import('./inferenceVideos').InferenceVideoRecord[] }>('/api/inference-video-history', { method: 'POST', body: JSON.stringify({ path, scene, station }) })
export const deleteInferenceVideo = (id: string) => request<{ ok: boolean }>(`/api/inference-video-history/${encodeURIComponent(id)}`, { method: 'DELETE' })

export const SSH_HISTORY_UPDATED_EVENT = 'yolo-ssh-history-updated'

const observeSetup = createSetupGuidanceMonitor(guide => {
  window.dispatchEvent(new CustomEvent(SETUP_GUIDANCE_EVENT, { detail: guide }))
})

export class ApiRequestError extends Error {
  constructor(message: string, readonly messages: string[]) { super(message) }
}

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  const data = await response.json()
  if (data.code === 'activation_required') window.dispatchEvent(new Event('yolo-activation-required'))
  observeSetup(url, options?.method || 'GET', response.ok, data)
  if (!response.ok) throw new ApiRequestError(data.error || '请求失败', Array.isArray(data.messages) ? data.messages : [])
  return data as T
}

export async function getFunctions(): Promise<FunctionDefinition[]> {
  const data = await request<{ functions: FunctionDefinition[] }>('/api/functions')
  return data.functions
}

export const previewDedupRegion = (imagesFolder: string, recursive: boolean, folderKey?: string, sampleName?: string) => request<import('./processRegions').RegionPreview>('/api/image-dedup/region-preview', {
  method: 'POST', body: JSON.stringify({ imagesFolder, recursive, ...(folderKey === undefined ? {} : { folderKey }), ...(sampleName ? { sampleName } : {}) }),
})

export async function launchXAnyLabeling(): Promise<{ message: string }> {
  return request('/api/apps/xanylabeling/launch', { method: 'POST', body: '{}' })
}

export async function launchAntigravity(): Promise<{ message: string }> {
  return request('/api/apps/antigravity/launch', { method: 'POST', body: '{}' })
}

export async function launchExternalApp(appId: string): Promise<{ message: string }> {
  return request(`/api/apps/${encodeURIComponent(appId)}/launch`, { method: 'POST', body: '{}' })
}

export async function getPlatformSettings(): Promise<PlatformSettings> {
  return request('/api/settings')
}
export const getPackageNaming = (line = '', station = '') => request<import('./packageNaming').PackageNamingHistory>(`/api/local-star-package/naming?line=${encodeURIComponent(line)}&station=${encodeURIComponent(station)}`)

export const getWorkClock = () => request<import('./workClock').WorkClockPayload>('/api/work-clock')
export type Countdown = { id: string; note: string; deadlineMs: number }
export type CountdownPayload = { timers: Countdown[]; nativeNotifications: boolean }
export const getCountdowns = (signal: AbortSignal) => request<CountdownPayload>('/api/countdowns', { signal })
export const startCountdown = (note: string, durationSeconds: number) => request<CountdownPayload>('/api/countdowns', {
  method: 'POST', body: JSON.stringify({ note, durationSeconds }),
})
export const removeCountdown = (id: string) => request<CountdownPayload>(`/api/countdowns/${encodeURIComponent(id)}`, { method: 'DELETE' })
export const getOvertime = () => request<import('./overtime').OvertimeStatus>('/api/overtime')
export const updateOvertime = (action: string, values: Record<string, unknown>) => request<import('./overtime').OvertimeStatus>(`/api/overtime/${action}`, { method: 'POST', body: JSON.stringify(values) })
export const saveWorkClock = (settings: import('./workClock').WorkClockSettings) => request<import('./workClock').WorkClockPayload>('/api/work-clock', { method: 'PUT', body: JSON.stringify(settings) })
export const saveFunctionDefaults = (functionId: string, values: WorkingValues) => request<FunctionDefinition>(`/api/functions/${encodeURIComponent(functionId)}/defaults`, { method: 'PUT', body: JSON.stringify(values) })

export async function savePlatformSettings(settings: PlatformSettings): Promise<PlatformSettings> {
  return request('/api/settings', { method: 'PUT', body: JSON.stringify(settings) })
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
  task?: PlatformTaskStatus
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

export async function getRunHistory(functionId?: string): Promise<RunHistoryRecord[]> {
  const endpoint = functionId
    ? `/api/tasks/history?${new URLSearchParams({ functionId })}`
    : '/api/tasks/history'
  const data = await request<{ history: RunHistoryRecord[] }>(endpoint)
  return data.history
}

export async function deleteRunHistory(recordId: string): Promise<void> {
  await request(`/api/tasks/history/${encodeURIComponent(recordId)}`, { method: 'DELETE' })
}

export async function stopActiveTask(taskId: string): Promise<PlatformTaskStatus> {
  const data = await request<{ active: PlatformTaskStatus }>(`/api/tasks/${taskId}/stop`, {
    method: 'POST',
    body: '{}',
  })
  return data.active
}

let choosingPath = false

export async function choosePaths(mode: string): Promise<string[]> {
  if (choosingPath) return []
  choosingPath = true
  const trigger = document.activeElement
  if (trigger instanceof HTMLButtonElement) trigger.setAttribute('aria-busy', 'true')
  try {
    const data = await request<{ paths: string[] }>('/api/dialog', { method: 'POST', body: JSON.stringify({ mode }) })
    return data.paths
  } finally {
    choosingPath = false
    if (trigger instanceof HTMLButtonElement) trigger.removeAttribute('aria-busy')
  }
}

export async function detectLocalStarLabels(modelFile: string): Promise<StarLabelDetectionResult> {
  return request<StarLabelDetectionResult>('/api/local-star-package/detect-labels', {
    method: 'POST',
    body: JSON.stringify({ modelFile }),
  })
}

export async function startRemoteInference(values: WorkingValues): Promise<RemoteInferenceStatus> {
  const result = await request<RemoteInferenceStatus>('/api/remote-inference/start', {
    method: 'POST',
    body: JSON.stringify(values),
  })
  window.dispatchEvent(new CustomEvent(INFERENCE_VIDEOS_UPDATED_EVENT, { detail: result.videoHistoryWarning || '' }))
  return result
}

export const detectRemoteStarLabels = (modelFile: string) => request<StarLabelDetectionResult>('/api/remote-inference/detect-labels', {
  method: 'POST', body: JSON.stringify({ modelFile }),
})

export async function startLocalPTInference(values: WorkingValues): Promise<RemoteInferenceStatus> {
  const result = await request<RemoteInferenceStatus>('/api/local-pt-inference/start', { method: 'POST', body: JSON.stringify(values) })
  if (values.parameters.input_mode !== '图片文件夹检测') window.dispatchEvent(new CustomEvent(INFERENCE_VIDEOS_UPDATED_EVENT, { detail: result.videoHistoryWarning || '' }))
  return result
}

export type LocalModelLabels = { labels: string[]; count: number; found: boolean; modelType: string; source: string; message: string }
export type ExportModelLabels = { labels: string[]; count: number; found: boolean; modelFamily: string; task: string; source: string; message: string }
export const detectExportModelLabels = (modelFile: string, parameters: WorkingValues['parameters']) => request<ExportModelLabels>('/api/jetson-onnx-export/detect-labels', {
  method: 'POST', body: JSON.stringify({ modelFile, parameters: { model_family: parameters.model_family, python_path: parameters.python_path, yolov5_repo: parameters.yolov5_repo } }),
})
export const detectLocalModelLabels = (modelFile: string) => request<LocalModelLabels>('/api/local-pt-inference/detect-labels', {
  method: 'POST', body: JSON.stringify({ modelFile }),
})

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

export async function stepLocalPTInference(id: string, direction: -1 | 1): Promise<RemoteInferenceStatus> {
  return request(`/api/local-pt-inference/${id}/step`, { method: 'POST', body: JSON.stringify({ direction }) })
}

export async function testRemoteConnection(parameters: WorkingValues['parameters']): Promise<ConnectionTestResult> {
  const result = await request<ConnectionTestResult>('/api/remote-inference/test-connection', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
  window.dispatchEvent(new Event(SSH_HISTORY_UPDATED_EVENT))
  return result
}

export async function getSshConnectionHistory(): Promise<SshConnectionProfile[]> {
  const data = await request<{ connections: SshConnectionProfile[] }>('/api/ssh/connections')
  return data.connections
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
  const result = await request<YoloConnectionTestResult>('/api/yolo-dataset/test-connection', {
    method: 'POST',
    body: JSON.stringify({ parameters }),
  })
  window.dispatchEvent(new Event(SSH_HISTORY_UPDATED_EVENT))
  return result
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

export async function deleteYoloTrainingSession(sessionId: string, forget = false): Promise<void> {
  await request(`/api/yolo-training/sessions/${encodeURIComponent(sessionId)}${forget ? '?forget=true' : ''}`, { method: 'DELETE' })
}

export async function stopYoloTraining(sessionId: string, parameters?: YoloTrainingValues): Promise<YoloTrainingSession> {
  return request<YoloTrainingSession>(`/api/yolo-training/${sessionId}/stop`, {
    method: 'POST',
    body: JSON.stringify({ parameters }),
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

export async function stepRemoteInference(sessionId: string, direction: -1 | 1): Promise<RemoteInferenceStatus> {
  return request(`/api/remote-inference/${sessionId}/step`, { method: 'POST', body: JSON.stringify({ direction }) })
}
