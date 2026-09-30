import { useCallback, useEffect, useMemo, useState } from 'react'
import { choosePaths, createGroup, deleteFunction, deleteGroup, getActiveTask, getWorkspace, moveFunction, renameGroup, reorderGroups, runFunction, stopActiveTask, updateFunction } from './api'
import { FunctionEditor } from './components/FunctionEditor'
import { GroupDialog } from './components/GroupDialog'
import { FunctionSidebar } from './components/FunctionSidebar'
import { FunctionWorkspace } from './components/FunctionWorkspace'
import { RunHistoryDialog } from './components/RunHistoryDialog'
import { WorkflowTodoDialog } from './components/WorkflowTodoDialog'
import { icons } from './icons'
import { parseStarModels, upsertStarModel } from './remoteStarModels'
import { createWorkflowTodo, loadWorkflowTodos, saveWorkflowTodos } from './workflowTodos'
import type { FunctionDefinition, GroupDefinition, PlatformTaskStatus, WorkingValues, WorkspaceData, YoloTrainingRecommendation } from './types'
import type { WorkflowTodo, WorkflowTodoPayload } from './workflowTodos'

const storageKey = (id: string) => `processing-view:values:${id}`
const sidebarStorageKey = 'yolo-data-platform:sidebar-collapsed:v1'
const remoteDefaultsMigrationKey = 'yolo-data-platform:remote-defaults:v2'
const remotePreviewMigrationKey = 'yolo-data-platform:remote-preview:v3'
const yoloConnectionMigrationKey = 'yolo-data-platform:yolo-connection:v1'
const videoClipCopyMigrationKey = 'yolo-data-platform:video-clip-copy:v1'
const datasetFolderBlankMigrationKey = 'yolo-data-platform:dataset-folder-blank:v1'
const automaticDatasetFolderPattern = /^yolo_train_\d{8}_\d{6}$/

function sameActiveTaskState(current: PlatformTaskStatus[], next: PlatformTaskStatus[]): boolean {
  if (current.length !== next.length) return false
  return current.every((task, index) => {
    const candidate = next[index]
    return candidate !== undefined
      && task.id === candidate.id
      && task.functionId === candidate.functionId
      && task.name === candidate.name
      && task.kind === candidate.kind
      && task.status === candidate.status
      && task.startedAt === candidate.startedAt
  })
}

function loadSidebarCollapsed(): boolean {
  try {
    return localStorage.getItem(sidebarStorageKey) === 'true'
  } catch {
    return false
  }
}

function migrateRemoteDefaults(items: FunctionDefinition[]) {
  try {
    if (localStorage.getItem(remoteDefaultsMigrationKey) === 'done') return
    const remoteItem = items.find((item) => item.handlerId === 'remote.star_inference')
    if (remoteItem) {
      const key = storageKey(remoteItem.id)
      const stored = localStorage.getItem(key)
      if (stored) {
        const values = JSON.parse(stored) as WorkingValues
        values.parameters = {
          ...values.parameters,
          remember_password: true,
          labels: values.parameters.labels === 'class0' ? '' : values.parameters.labels,
          password: '',
        }
        localStorage.setItem(key, JSON.stringify(values))
      }
    }
    localStorage.setItem(remoteDefaultsMigrationKey, 'done')
  } catch {
    // Defaults still work when browser storage is unavailable.
  }
}

function migrateRemotePreviewDefaults(items: FunctionDefinition[]) {
  try {
    if (localStorage.getItem(remotePreviewMigrationKey) === 'done') return
    const remoteItem = items.find((item) => item.handlerId === 'remote.star_inference')
    if (remoteItem) {
      const key = storageKey(remoteItem.id)
      const stored = localStorage.getItem(key)
      if (stored) {
        const values = JSON.parse(stored) as WorkingValues
        const settings = values.parameters
        if (Number(settings.preview_fps) === 12 && Number(settings.stream_width) === 1280 && Number(settings.jpeg_quality) === 80) {
          localStorage.setItem(key, JSON.stringify({ ...values, parameters: { ...settings, preview_fps: 20, stream_width: 960, jpeg_quality: 70 } }))
        }
      }
    }
    localStorage.setItem(remotePreviewMigrationKey, 'done')
  } catch {
    // A stored custom setting still takes precedence if migration is unavailable.
  }
}

function migrateYoloConnectionDefaults(items: FunctionDefinition[]) {
  try {
    if (localStorage.getItem(yoloConnectionMigrationKey) === 'done') return
    const yoloItem = items.find((item) => item.handlerId === 'yolo.split_dataset')
    if (yoloItem) {
      const key = storageKey(yoloItem.id)
      const stored = localStorage.getItem(key)
      if (stored) {
        const values = JSON.parse(stored) as WorkingValues
        const usedLegacyRemoteMode = values.parameters.execution_location === 'SSH 远程服务器'
        const parameters = { ...values.parameters }
        delete parameters.execution_location
        parameters.remote_password = ''
        parameters.remember_password = parameters.remember_password ?? true
        if (!usedLegacyRemoteMode) {
          parameters.remote_host = ''
          parameters.remote_username = ''
        }
        localStorage.setItem(key, JSON.stringify({ ...values, parameters }))
      }
    }
    localStorage.setItem(yoloConnectionMigrationKey, 'done')
  } catch {
    // New catalog defaults remain available when browser storage cannot be migrated.
  }
}

function migrateDatasetFolderToBlank(items: FunctionDefinition[]) {
  try {
    if (localStorage.getItem(datasetFolderBlankMigrationKey) === 'done') return
    const yoloItem = items.find((item) => item.handlerId === 'yolo.split_dataset')
    if (yoloItem) {
      const key = storageKey(yoloItem.id)
      const stored = localStorage.getItem(key)
      if (stored) {
        const values = JSON.parse(stored) as WorkingValues
        const currentName = String(values.parameters.dataset_folder_name ?? '').trim()
        if (currentName === 'yolo_train' || automaticDatasetFolderPattern.test(currentName)) {
          localStorage.setItem(key, JSON.stringify(withoutSecrets(yoloItem, {
            ...values,
            parameters: { ...values.parameters, dataset_folder_name: '' },
          })))
        }
      }
    }
    localStorage.setItem(datasetFolderBlankMigrationKey, 'done')
  } catch {
    // The field remains manually editable if browser storage is unavailable.
  }
}

function migrateVideoClipDefault(items: FunctionDefinition[]) {
  try {
    if (localStorage.getItem(videoClipCopyMigrationKey) === 'done') return
    const clipItem = items.find((item) => item.handlerId === 'video.clip')
    if (clipItem) {
      const key = storageKey(clipItem.id)
      const stored = localStorage.getItem(key)
      if (stored) {
        const values = JSON.parse(stored) as WorkingValues
        const currentMode = values.parameters.encoding_mode
        if (currentMode === '精确裁剪（推荐）' || currentMode === '快速裁剪（不重新编码）') {
          localStorage.setItem(key, JSON.stringify({
            ...values,
            parameters: { ...values.parameters, encoding_mode: '原画质裁剪（推荐，不重新编码）' },
          }))
        }
      }
    }
    localStorage.setItem(videoClipCopyMigrationKey, 'done')
  } catch {
    // The new catalog default remains available when browser storage cannot be migrated.
  }
}

function withoutSecrets(item: FunctionDefinition, values: WorkingValues): WorkingValues {
  const secretIds = new Set(item.parameters.filter((field) => field.type === 'password').map((field) => field.id))
  return {
    paths: values.paths,
    parameters: Object.fromEntries(Object.entries(values.parameters).map(([key, value]) => [key, secretIds.has(key) ? '' : value])),
  }
}

function defaultsFor(item: FunctionDefinition): WorkingValues {
  const stored = localStorage.getItem(storageKey(item.id))
  if (stored) {
    try { return withoutSecrets(item, JSON.parse(stored) as WorkingValues) } catch { /* use defaults */ }
  }
  return {
    paths: Object.fromEntries(item.pathFields.map((field) => [field.id, ''])),
    parameters: Object.fromEntries(item.parameters.map((field) => [field.id, field.default])),
  }
}

export default function App() {
  const [functions, setFunctions] = useState<FunctionDefinition[]>([])
  const [groups, setGroups] = useState<GroupDefinition[]>([])
  const [selectedId, setSelectedId] = useState('')
  const [search, setSearch] = useState('')
  const [editing, setEditing] = useState<FunctionDefinition | null>(null)
  const [groupDraft, setGroupDraft] = useState<GroupDefinition | null | undefined>(undefined)
  const [working, setWorking] = useState<Record<string, WorkingValues>>({})
  const [outputs, setOutputs] = useState<Record<string, string>>({})
  const [runningIds, setRunningIds] = useState<string[]>([])
  const [activeTasks, setActiveTasks] = useState<PlatformTaskStatus[]>([])
  const [notice, setNotice] = useState('')
  const [historyOpen, setHistoryOpen] = useState(false)
  const [workflowOpen, setWorkflowOpen] = useState(false)
  const [sidebarCollapsed, setSidebarCollapsed] = useState(loadSidebarCollapsed)
  const [yoloTrainingRecommendation, setYoloTrainingRecommendation] = useState<YoloTrainingRecommendation | null>(null)
  const [workflowTodos, setWorkflowTodos] = useState<WorkflowTodo[]>(loadWorkflowTodos)
  const closeHistory = useCallback(() => setHistoryOpen(false), [])

  const selected = useMemo(() => functions.find((item) => item.id === selectedId) || null, [functions, selectedId])
  const values = selected ? (working[selected.id] || defaultsFor(selected)) : { paths: {}, parameters: {} }

  useEffect(() => {
    const url = new URL(window.location.href)
    if (url.searchParams.has('hotReload')) {
      setNotice('平台功能已热更新，可以直接使用。')
      window.history.replaceState({}, '', `${url.pathname}${url.hash}`)
    }
  }, [])

  useEffect(() => {
    getWorkspace().then((workspace) => {
      migrateRemoteDefaults(workspace.functions)
      migrateRemotePreviewDefaults(workspace.functions)
      migrateYoloConnectionDefaults(workspace.functions)
      migrateVideoClipDefault(workspace.functions)
      migrateDatasetFolderToBlank(workspace.functions)
      setFunctions(workspace.functions)
      setGroups(workspace.groups)
      if (workspace.functions[0]) setSelectedId(workspace.functions[0].id)
    }).catch((error) => setNotice(error.message))
  }, [])

  useEffect(() => {
    let disposed = false
    let timer: number | undefined
    const refresh = async () => {
      try {
        const tasks = await getActiveTask()
        if (disposed) return
        setActiveTasks((current) => sameActiveTaskState(current, tasks) ? current : tasks)
        setOutputs((current) => {
          let changed = false
          const next = { ...current }
          for (const task of tasks) {
            if (task.logs?.length) {
              const output = task.logs.join('\n')
              if (current[task.functionId] !== output) {
                next[task.functionId] = output
                changed = true
              }
            }
          }
          return changed ? next : current
        })
      } catch {
        // The main request will surface server errors.
      } finally {
        if (!disposed) timer = window.setTimeout(refresh, 800)
      }
    }
    void refresh()
    return () => {
      disposed = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  const applyWorkspace = (workspace: WorkspaceData) => {
    setFunctions(workspace.functions)
    setGroups(workspace.groups)
    setSelectedId((current) => workspace.functions.some((item) => item.id === current) ? current : (workspace.functions[0]?.id || ''))
  }

  const updatePathValue = (fieldId: string, value: string) => {
    if (!selected) return
    setWorking((current) => {
      const currentValues = current[selected.id] || defaultsFor(selected)
      const next = { ...currentValues, paths: { ...currentValues.paths, [fieldId]: value } }
      localStorage.setItem(storageKey(selected.id), JSON.stringify(withoutSecrets(selected, next)))
      return { ...current, [selected.id]: next }
    })
  }

  const updateParameterValue = (fieldId: string, value: string | number | boolean) => {
    if (!selected) return
    setWorking((current) => {
      const currentValues = current[selected.id] || defaultsFor(selected)
      const next = { ...currentValues, parameters: { ...currentValues.parameters, [fieldId]: value } }
      localStorage.setItem(storageKey(selected.id), JSON.stringify(withoutSecrets(selected, next)))
      return { ...current, [selected.id]: next }
    })
  }

  const save = async (draft: FunctionDefinition) => {
    try {
      const item = await updateFunction(draft)
      setFunctions((current) => current.map((entry) => entry.id === item.id ? item : entry))
      setEditing(null)
    } catch (error) { setNotice(error instanceof Error ? error.message : '保存失败') }
  }

  const saveGroup = async (name: string) => {
    try {
      const workspace = groupDraft ? await renameGroup(groupDraft.id, name) : await createGroup(name)
      applyWorkspace(workspace)
      setGroupDraft(undefined)
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '保存分组失败')
      throw error
    }
  }

  const removeGroup = async (group: GroupDefinition) => {
    const count = functions.filter((item) => item.groupId === group.id).length
    const detail = count ? `其中 ${count} 个功能会移到“未分组”，功能本身不会被删除。` : '该分组目前为空。'
    if (!window.confirm(`确定删除分组“${group.name}”吗？\n${detail}`)) return
    try {
      applyWorkspace(await deleteGroup(group.id))
    } catch (error) { setNotice(error instanceof Error ? error.message : '删除分组失败') }
  }

  const changeGroupOrder = async (groupIds: string[]) => {
    try {
      applyWorkspace(await reorderGroups(groupIds))
    } catch (error) { setNotice(error instanceof Error ? error.message : '调整分组顺序失败') }
  }

  const moveItem = async (itemId: string, groupId: string | null, position?: number) => {
    try {
      applyWorkspace(await moveFunction(itemId, groupId, position))
    } catch (error) { setNotice(error instanceof Error ? error.message : '移动功能失败') }
  }

  const remove = async (item: FunctionDefinition) => {
    if (!window.confirm(`确定删除“${item.name}”吗？`)) return
    try {
      await deleteFunction(item.id)
      const remaining = functions.filter((entry) => entry.id !== item.id)
      setFunctions(remaining)
      setSelectedId(remaining[0]?.id || '')
      setEditing(null)
      localStorage.removeItem(storageKey(item.id))
    } catch (error) { setNotice(error instanceof Error ? error.message : '删除失败') }
  }

  const browse = async (fieldId: string, mode: string) => {
    try {
      const paths = await choosePaths(mode)
      if (paths[0]) updatePathValue(fieldId, paths[0])
    } catch (error) { setNotice(error instanceof Error ? error.message : '无法打开选择窗口') }
  }

  const appendPath = async () => {
    if (!selected) return
    const next = { ...selected, pathFields: [...selected.pathFields, { id: `path_${crypto.randomUUID().slice(0, 8)}`, label: '新路径', mode: 'directory' as const }] }
    await save(next)
  }

  const appendParameter = async () => {
    if (!selected) return
    const next = { ...selected, parameters: [...selected.parameters, { id: `param_${crypto.randomUUID().slice(0, 8)}`, label: '新参数', type: 'text' as const, default: '', options: [] }] }
    await save(next)
  }

  const updateWorkflowTodos = (updater: (current: WorkflowTodo[]) => WorkflowTodo[]) => {
    setWorkflowTodos((current) => {
      const next = updater(current).slice(0, 100)
      saveWorkflowTodos(next)
      return next
    })
  }

  const enqueueWorkflowTodo = (
    source: FunctionDefinition,
    targetHandlerId: string,
    title: string,
    description: string,
    payload: WorkflowTodoPayload,
  ) => {
    const todo = createWorkflowTodo({
      sourceFunctionId: source.id,
      sourceName: source.name,
      targetHandlerId,
      title,
      description,
      payload,
    })
    updateWorkflowTodos((current) => [todo, ...current])
    return todo
  }

  const deleteWorkflowTodo = (id: string) => updateWorkflowTodos((current) => current.filter((item) => item.id !== id))

  const completeWorkflowTarget = (handlerId: string) => {
    if (handlerId === 'yolo.split_dataset') setYoloTrainingRecommendation(null)
    updateWorkflowTodos((current) => current.filter((todo) => !(todo.targetHandlerId === handlerId && todo.status === 'applied')))
  }

  const applyWorkflowTodo = (todo: WorkflowTodo) => {
    if (workflowTodos.some((item) => item.targetHandlerId === todo.targetHandlerId && item.status === 'applied' && item.id !== todo.id)) {
      setNotice('请先运行当前已应用的工作流待办，或将它删除后再应用下一条。')
      return
    }
    const target = functions.find((item) => item.handlerId === todo.targetHandlerId)
    if (!target) {
      setNotice('待办对应的目标功能不存在，可能已被删除。')
      return
    }
    if (todo.payload.training) {
      const recommendation: YoloTrainingRecommendation = {
        ...todo.payload.training,
        updatedAt: new Date().toISOString(),
      }
      setYoloTrainingRecommendation(recommendation)
    } else {
      const currentValues = working[target.id] || defaultsFor(target)
      let next: WorkingValues = {
        paths: { ...currentValues.paths, ...todo.payload.paths },
        parameters: { ...currentValues.parameters, ...todo.payload.parameters },
      }
      if (todo.payload.starModel) {
        const configuredConfidence = Number(currentValues.parameters.conf ?? 0.5)
        const confidence = Number.isFinite(configuredConfidence) ? configuredConfidence : 0.5
        const models = parseStarModels(
          currentValues.parameters.star_models,
          String(currentValues.paths.local_model_file || ''),
          String(currentValues.parameters.labels || ''),
          confidence,
        )
        const model = todo.payload.starModel
        const merged = upsertStarModel(models, {
          path: model.path,
          labels: model.labels,
          conf: model.conf ?? confidence,
        })
        if (merged.action === 'full') {
          setNotice('远程实时 AI 推理已有 8 个模型，请先删除一个模型后再应用此待办。')
          return
        }
        next = {
          ...next,
          paths: { ...next.paths, local_model_file: model.path },
          parameters: {
            ...next.parameters,
            labels: model.labels,
            star_models: JSON.stringify(merged.models),
          },
        }
      }
      setWorking((current) => ({ ...current, [target.id]: next }))
      try { localStorage.setItem(storageKey(target.id), JSON.stringify(withoutSecrets(target, next))) } catch { /* keep current session state */ }
    }
    updateWorkflowTodos((current) => current.map((item) => item.id === todo.id ? { ...item, status: 'applied' } : item))
    setNotice(`已应用待办“${todo.title}”。本功能成功运行后会自动清理该待办。`)
  }

  const applyWorkflowTodoFromDialog = (todo: WorkflowTodo) => {
    const target = functions.find((item) => item.handlerId === todo.targetHandlerId)
    if (target) setSelectedId(target.id)
    applyWorkflowTodo(todo)
    setWorkflowOpen(false)
  }

  const run = async (parameterOverrides: Record<string, string | number | boolean> = {}) => {
    if (!selected) return
    const item = selected
    const runValues = values
    setRunningIds((current) => [...current, item.id])
    const started = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    setOutputs((current) => ({ ...current, [item.id]: `[${started}] 正在准备运行……` }))
    try {
      const result = await runFunction(item.id, {
        ...runValues,
        parameters: { ...runValues.parameters, ...parameterOverrides },
      })
      const lines = [...result.messages]
      if (result.result.message) lines.push('', result.result.message)
      if (result.result.outputFolders?.length) lines.push('', '输出目录：', ...result.result.outputFolders)
      const trainingDefaults = result.result.trainingDefaults
      if (item.handlerId === 'yolo.split_dataset' && trainingDefaults && typeof trainingDefaults === 'object') {
        const candidate = trainingDefaults as Record<string, unknown>
        if (typeof candidate.data === 'string' && typeof candidate.project === 'string') {
          enqueueWorkflowTodo(item, 'yolo.split_dataset', '使用本次数据集开始模型训练', `data=${candidate.data}；project=${candidate.project}；name=运行时自动生成`, {
            training: { data: candidate.data, project: candidate.project },
          })
          lines.push('', '已加入工作流待办：模型训练参数。当前训练配置没有被覆盖。')
        }
      }
      const onnxQuantDefaults = result.result.onnxQuantDefaults
      if (item.handlerId === 'model.export_jetson_onnx' && onnxQuantDefaults && typeof onnxQuantDefaults === 'object') {
        const candidate = onnxQuantDefaults as Record<string, unknown>
        if (typeof candidate.inputOnnx === 'string') {
          enqueueWorkflowTodo(item, 'docker.quantize_onnx', '量化本次导出的 ONNX', candidate.inputOnnx, {
            paths: { input_onnx: candidate.inputOnnx },
          })
          lines.push('', '已加入工作流待办：Docker ONNX 量化。当前量化配置没有被覆盖。')
        }
      }
      const remoteInferenceDefaults = result.result.remoteInferenceDefaults
      if (item.handlerId === 'model.package_star' && remoteInferenceDefaults && typeof remoteInferenceDefaults === 'object') {
        const candidate = remoteInferenceDefaults as Record<string, unknown>
        const labels = Array.isArray(candidate.labels)
          ? candidate.labels.filter((label): label is string => typeof label === 'string' && Boolean(label.trim()))
          : []
        if (typeof candidate.modelFile === 'string' && candidate.modelFile.trim() && labels.length) {
          const joinedLabels = labels.join(',')
          enqueueWorkflowTodo(item, 'remote.star_inference', '测试新打包的 STAR 模型', `${candidate.modelFile}；类别：${joinedLabels}`, {
            starModel: {
              path: candidate.modelFile,
              labels: joinedLabels,
            },
          })
          lines.push('', '已加入工作流待办：远程实时 AI 推理模型。当前推理模型配置没有被覆盖。')
        }
      }
      const operation = parameterOverrides.operation ?? runValues.parameters.operation
      const consumedWorkflowTodo = item.handlerId !== 'yolo.split_dataset'
        && !(item.handlerId === 'docker.quantize_onnx' && operation === '测试 Docker 环境')
        && !(item.handlerId === 'remote.build_tensorrt' && operation === '检查远程环境')
      if (consumedWorkflowTodo) completeWorkflowTarget(String(item.handlerId || ''))
      setOutputs((current) => ({ ...current, [item.id]: lines.join('\n') }))
    } catch (error) {
      const message = error instanceof Error ? error.message : '运行失败'
      setOutputs((current) => ({ ...current, [item.id]: `[${started}] ${message}` }))
    } finally {
      setRunningIds((current) => current.filter((id) => id !== item.id))
      setActiveTasks((current) => current.filter((task) => task.functionId !== item.id))
    }
  }

  const stopRunningTask = async (taskId: string) => {
    try {
      const task = await stopActiveTask(taskId)
      setActiveTasks((current) => current.map((entry) => entry.id === task.id ? task : entry))
      setOutputs((current) => ({
        ...current,
        [task.functionId]: `${current[task.functionId] || ''}\n正在终止运行，请稍候……`.trim(),
      }))
    } catch (error) {
      setNotice(error instanceof Error ? error.message : '终止运行失败')
    }
  }

  const toggleSidebar = () => {
    setSidebarCollapsed((current) => {
      const next = !current
      try { localStorage.setItem(sidebarStorageKey, String(next)) } catch { /* keep session state */ }
      return next
    })
  }

  return (
    <div className={`app-shell ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><icons.Code2 size={20} /></span><strong>YOLO数据处理平台</strong></div>
        <div className="top-actions">
          <button className="history-trigger workflow-trigger" onClick={() => setWorkflowOpen(true)} aria-label={`工作流待办 ${workflowTodos.length} 条`}><icons.FolderInput size={17} /><span>待办 {workflowTodos.length}</span></button>
          <button className="history-trigger" onClick={() => setHistoryOpen(true)} aria-label="全局运行历史"><icons.Clock3 size={17} /><span>运行历史</span></button>
          {activeTasks.length ? <details className="task-overview"><summary>运行中 {activeTasks.length} 项</summary><div className="task-overview-list">{activeTasks.map((task) => <div key={task.id}><span><strong>{task.name}</strong><small>{task.status === 'stopping' ? '正在终止' : task.kind === 'remote-build' ? '远端后台构建中' : task.kind === 'remote-training' ? '远程训练中' : task.kind === 'remote' ? '远程运行中' : '本地运行中'}</small></span><button onClick={() => stopRunningTask(task.id)} disabled={task.status === 'stopping'} aria-label={`终止 ${task.name}`}>终止</button></div>)}</div></details> : null}
          <span className="local-status"><i />本地运行</span><button aria-label="设置"><icons.Settings size={18} /></button><button aria-label="帮助"><icons.CircleHelp size={18} /></button>
        </div>
      </header>
      <FunctionSidebar
        functions={functions}
        groups={groups}
        selectedId={selectedId}
        search={search}
        onSearch={setSearch}
        onSelect={setSelectedId}
        onEdit={setEditing}
        onCreateGroup={() => setGroupDraft(null)}
        onRenameGroup={setGroupDraft}
        onDeleteGroup={removeGroup}
        onReorderGroups={changeGroupOrder}
        onMoveFunction={moveItem}
        collapsed={sidebarCollapsed}
        onToggleCollapsed={toggleSidebar}
      />
      <FunctionWorkspace
        item={selected}
        values={values}
        output={selected ? outputs[selected.id] || '' : ''}
        running={selected ? runningIds.includes(selected.id) : false}
        activeTasks={activeTasks}
        onEdit={() => selected && setEditing(selected)}
        onBrowse={browse}
        onPathChange={updatePathValue}
        onParameterChange={updateParameterValue}
        onAddPath={() => appendPath().catch((error) => setNotice(error.message))}
        onAddParameter={() => appendParameter().catch((error) => setNotice(error.message))}
        onRun={run}
        onStop={stopRunningTask}
        yoloTrainingRecommendation={yoloTrainingRecommendation}
        workflowTodos={selected ? workflowTodos.filter((todo) => todo.targetHandlerId === selected.handlerId) : []}
        onApplyWorkflowTodo={applyWorkflowTodo}
        onDeleteWorkflowTodo={deleteWorkflowTodo}
        onWorkflowComplete={completeWorkflowTarget}
      />
      <FunctionEditor item={editing} onClose={() => setEditing(null)} onSave={save} onDelete={remove} />
      <GroupDialog group={groupDraft} onClose={() => setGroupDraft(undefined)} onSave={saveGroup} />
      {historyOpen ? <RunHistoryDialog onClose={closeHistory} /> : null}
      {workflowOpen ? <WorkflowTodoDialog items={workflowTodos} onApply={applyWorkflowTodoFromDialog} onDelete={deleteWorkflowTodo} onClose={() => setWorkflowOpen(false)} /> : null}
      {notice ? <div className="toast" role="alert"><icons.CircleAlert size={18} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="关闭提示">×</button></div> : null}
    </div>
  )
}
