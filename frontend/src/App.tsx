import { useCallback, useEffect, useMemo, useState } from 'react'
import { choosePaths, createGroup, deleteFunction, deleteGroup, getActiveTask, getWorkspace, moveFunction, renameGroup, reorderGroups, runFunction, stopActiveTask, updateFunction } from './api'
import { FunctionEditor } from './components/FunctionEditor'
import { GroupDialog } from './components/GroupDialog'
import { FunctionSidebar } from './components/FunctionSidebar'
import { FunctionWorkspace } from './components/FunctionWorkspace'
import { RunHistoryDialog } from './components/RunHistoryDialog'
import { icons } from './icons'
import type { FunctionDefinition, GroupDefinition, PlatformTaskStatus, WorkingValues, WorkspaceData, YoloTrainingRecommendation } from './types'

const storageKey = (id: string) => `processing-view:values:${id}`
const sidebarStorageKey = 'yolo-data-platform:sidebar-collapsed:v1'
const remoteDefaultsMigrationKey = 'yolo-data-platform:remote-defaults:v2'
const remotePreviewMigrationKey = 'yolo-data-platform:remote-preview:v3'
const yoloConnectionMigrationKey = 'yolo-data-platform:yolo-connection:v1'
const videoClipCopyMigrationKey = 'yolo-data-platform:video-clip-copy:v1'
const yoloTrainingRecommendationKey = 'yolo-data-platform:training-recommendation:v1'

function yoloRunName(date: Date): string {
  const twoDigits = (value: number) => String(value).padStart(2, '0')
  return `yolo26_${twoDigits(date.getMonth() + 1)}${twoDigits(date.getDate())}_${twoDigits(date.getHours())}${twoDigits(date.getMinutes())}`
}

function loadYoloTrainingRecommendation(): YoloTrainingRecommendation | null {
  try {
    const stored = localStorage.getItem(yoloTrainingRecommendationKey)
    if (!stored) return null
    const parsed = JSON.parse(stored) as Partial<YoloTrainingRecommendation>
    if (typeof parsed.data !== 'string' || typeof parsed.project !== 'string' || typeof parsed.updatedAt !== 'string') return null
    return {
      data: parsed.data,
      project: parsed.project,
      runName: typeof parsed.runName === 'string' ? parsed.runName : undefined,
      updatedAt: parsed.updatedAt,
    }
  } catch {
    return null
  }
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
  const [sidebarCollapsed, setSidebarCollapsed] = useState(loadSidebarCollapsed)
  const [yoloTrainingRecommendation, setYoloTrainingRecommendation] = useState<YoloTrainingRecommendation | null>(loadYoloTrainingRecommendation)
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
      setFunctions(workspace.functions)
      setGroups(workspace.groups)
      if (workspace.functions[0]) setSelectedId(workspace.functions[0].id)
    }).catch((error) => setNotice(error.message))
  }, [])

  useEffect(() => {
    let disposed = false
    const refresh = () => getActiveTask()
      .then((tasks) => {
        if (disposed) return
        setActiveTasks(tasks)
        setOutputs((current) => {
          let changed = false
          const next = { ...current }
          for (const task of tasks) {
            if (task.logs?.length) {
              next[task.functionId] = task.logs.join('\n')
              changed = true
            }
          }
          return changed ? next : current
        })
      })
      .catch(() => { /* The main request will surface server errors. */ })
    refresh()
    const timer = window.setInterval(refresh, 800)
    return () => {
      disposed = true
      window.clearInterval(timer)
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
          const completedAt = new Date()
          const recommendation: YoloTrainingRecommendation = {
            data: candidate.data,
            project: candidate.project,
            runName: yoloRunName(completedAt),
            updatedAt: completedAt.toISOString(),
          }
          setYoloTrainingRecommendation(recommendation)
          try { localStorage.setItem(yoloTrainingRecommendationKey, JSON.stringify(recommendation)) } catch { /* keep current session state */ }
          lines.push('', '已自动更新模型训练参数：', `data=${candidate.data}`, `project=${candidate.project}`, `name=${recommendation.runName}`)
        }
      }
      const onnxQuantDefaults = result.result.onnxQuantDefaults
      if (item.handlerId === 'model.export_jetson_onnx' && onnxQuantDefaults && typeof onnxQuantDefaults === 'object') {
        const candidate = onnxQuantDefaults as Record<string, unknown>
        const quantItem = functions.find((entry) => entry.handlerId === 'docker.quantize_onnx')
        if (quantItem && typeof candidate.inputOnnx === 'string') {
          setWorking((current) => {
            const currentValues = current[quantItem.id] || defaultsFor(quantItem)
            const next = {
              ...currentValues,
              paths: { ...currentValues.paths, input_onnx: candidate.inputOnnx as string },
            }
            try { localStorage.setItem(storageKey(quantItem.id), JSON.stringify(withoutSecrets(quantItem, next))) } catch { /* keep current session state */ }
            return { ...current, [quantItem.id]: next }
          })
          lines.push('', '已自动更新 Docker ONNX 量化的输入模型：', candidate.inputOnnx)
        }
      }
      setOutputs((current) => ({ ...current, [item.id]: lines.join('\n') }))
    } catch (error) {
      const message = error instanceof Error ? error.message : '运行失败'
      setOutputs((current) => ({ ...current, [item.id]: `[${started}] ${message}` }))
    } finally {
      setRunningIds((current) => current.filter((id) => id !== item.id))
      getActiveTask().then(setActiveTasks).catch(() => undefined)
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
      getActiveTask().then(setActiveTasks).catch(() => undefined)
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
      />
      <FunctionEditor item={editing} onClose={() => setEditing(null)} onSave={save} onDelete={remove} />
      <GroupDialog group={groupDraft} onClose={() => setGroupDraft(undefined)} onSave={saveGroup} />
      {historyOpen ? <RunHistoryDialog onClose={closeHistory} /> : null}
      {notice ? <div className="toast" role="alert"><icons.CircleAlert size={18} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="关闭提示">×</button></div> : null}
    </div>
  )
}
