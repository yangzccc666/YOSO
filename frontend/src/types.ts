export type PathField = {
  id: string
  label: string
  mode: 'file' | 'directory'
}

export const REMEMBERED_PASSWORD_VALUE = '__YOLO_REMEMBERED_SSH_PASSWORD__'

export type FunctionParameter = {
  id: string
  label: string
  type: 'text' | 'password' | 'number' | 'boolean' | 'select'
  default: string | number | boolean
  options: string[]
  visibleWhen?: { fieldId: string; equals?: string | number | boolean; notEquals?: string | number | boolean } | null
}

export type RemoteInferenceStatus = {
  id: string
  host: string
  username: string
  status: 'connecting' | 'uploading' | 'starting' | 'running' | 'paused' | 'stopping' | 'completed' | 'stopped' | 'failed'
  message: string
  startedAt: string
  finishedAt: string | null
  frameCount: number
  durationSeconds: number
  positionSeconds: number
  sourceFps: number
  atEnd: boolean
  replayRemainingSeconds: number
  logs: string[]
  error: string | null
  transferStage: string | null
  uploadProgress: number
  streamUrl: string
}

export type ConnectionTestResult = {
  ok: true
  message: string
  host: string
  port: number
  username: string
  fingerprint: string
  passwordRemembered: boolean
  usedSavedPassword: boolean
  testedAt: string
}

export type YoloConnectionTestResult = ConnectionTestResult & {
  connectionToken: string
}

export type YoloTrainingValues = Record<string, string | number | boolean>

export type YoloTrainingRecommendation = {
  data: string
  project: string
  runName?: string
  updatedAt: string
}

export type YoloTrainingProfile = {
  id: string
  name: string
  description: string
  values: YoloTrainingValues
  updatedAt: string
}

export type YoloTrainingProfilesPayload = {
  profiles: YoloTrainingProfile[]
  defaults: YoloTrainingValues
}

export type YoloTrainingSession = {
  id: string
  remote: boolean
  model: string
  device: string
  output: string
  host: string
  port: number
  username: string
  status: 'starting' | 'running' | 'disconnected' | 'stopping' | 'completed' | 'stopped' | 'failed'
  message: string
  startedAt: string
  finishedAt: string | null
  logs: string[]
  error: string | null
  result: Record<string, unknown> | null
}

export type YoloTrainingSummary = Omit<YoloTrainingSession, 'logs'>

export type RemoteCredentialStatus = {
  ok: true
  host: string
  port: number
  username: string
  remembered: boolean
}

export type StarLabelDetectionResult = {
  ok: true
  labels: string[]
  source: string
  count: number
  message: string
}

export type PlatformTaskStatus = {
  id: string
  functionId: string
  name: string
  kind: 'local' | 'remote' | 'remote-build' | 'remote-training'
  status: 'running' | 'stopping'
  startedAt: string
  logs?: string[]
}

export type RunHistoryRecord = {
  id: string
  functionId: string
  name: string
  kind: 'local' | 'remote' | 'remote-build' | 'remote-training'
  status: 'running' | 'completed' | 'failed' | 'stopped'
  startedAt: string
  finishedAt: string
  message: string
  details: Record<string, string>
  outputs: string[]
  logs: string[]
}

export type FunctionDefinition = {
  id: string
  name: string
  description: string
  handlerId: string | null
  handlerReady: boolean
  pathFields: PathField[]
  parameters: FunctionParameter[]
  updatedAt: string
  groupId: string | null
  order: number
}

export type GroupDefinition = {
  id: string
  name: string
  order: number
  updatedAt: string
}

export type WorkspaceData = {
  groups: GroupDefinition[]
  functions: FunctionDefinition[]
}

export type WorkingValues = {
  paths: Record<string, string>
  parameters: Record<string, string | number | boolean>
}
