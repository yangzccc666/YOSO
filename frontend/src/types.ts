export type PathField = {
  id: string
  label: string
  mode: 'file' | 'directory'
}

export type FunctionParameter = {
  id: string
  label: string
  type: 'text' | 'password' | 'number' | 'boolean' | 'select'
  default: string | number | boolean
  options: string[]
  visibleWhen?: { fieldId: string; equals: string | number | boolean } | null
}

export type RemoteInferenceStatus = {
  id: string
  host: string
  username: string
  status: 'connecting' | 'uploading' | 'starting' | 'running' | 'stopping' | 'completed' | 'stopped' | 'failed'
  message: string
  startedAt: string
  finishedAt: string | null
  frameCount: number
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
