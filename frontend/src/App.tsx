import { useEffect, useMemo, useState } from 'react'
import { choosePaths, createGroup, deleteFunction, deleteGroup, getWorkspace, moveFunction, renameGroup, reorderGroups, runFunction, updateFunction } from './api'
import { FunctionEditor } from './components/FunctionEditor'
import { GroupDialog } from './components/GroupDialog'
import { FunctionSidebar } from './components/FunctionSidebar'
import { FunctionWorkspace } from './components/FunctionWorkspace'
import { icons } from './icons'
import type { FunctionDefinition, GroupDefinition, WorkingValues, WorkspaceData } from './types'

const storageKey = (id: string) => `processing-view:values:${id}`

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
  const [running, setRunning] = useState(false)
  const [notice, setNotice] = useState('')

  const selected = useMemo(() => functions.find((item) => item.id === selectedId) || null, [functions, selectedId])
  const values = selected ? (working[selected.id] || defaultsFor(selected)) : { paths: {}, parameters: {} }

  useEffect(() => {
    getWorkspace().then((workspace) => {
      setFunctions(workspace.functions)
      setGroups(workspace.groups)
      if (workspace.functions[0]) setSelectedId(workspace.functions[0].id)
    }).catch((error) => setNotice(error.message))
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

  const run = async () => {
    if (!selected) return
    setRunning(true)
    const started = new Date().toLocaleTimeString('zh-CN', { hour12: false })
    setOutputs((current) => ({ ...current, [selected.id]: `[${started}] 正在准备运行……` }))
    try {
      const result = await runFunction(selected.id, values)
      const lines = [...result.messages]
      if (result.result.message) lines.push('', result.result.message)
      if (result.result.outputFolders?.length) lines.push('', '输出目录：', ...result.result.outputFolders)
      setOutputs((current) => ({ ...current, [selected.id]: lines.join('\n') }))
    } catch (error) {
      const message = error instanceof Error ? error.message : '运行失败'
      setOutputs((current) => ({ ...current, [selected.id]: `[${started}] ${message}` }))
    } finally { setRunning(false) }
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><icons.Code2 size={20} /></span><strong>YOLO数据处理平台</strong></div>
        <div className="top-actions"><span className="local-status"><i />本地运行</span><button aria-label="设置"><icons.Settings size={18} /></button><button aria-label="帮助"><icons.CircleHelp size={18} /></button></div>
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
      />
      <FunctionWorkspace
        item={selected}
        values={values}
        output={selected ? outputs[selected.id] || '' : ''}
        running={running}
        onEdit={() => selected && setEditing(selected)}
        onBrowse={browse}
        onPathChange={updatePathValue}
        onParameterChange={updateParameterValue}
        onAddPath={() => appendPath().catch((error) => setNotice(error.message))}
        onAddParameter={() => appendParameter().catch((error) => setNotice(error.message))}
        onRun={run}
      />
      <FunctionEditor item={editing} onClose={() => setEditing(null)} onSave={save} onDelete={remove} />
      <GroupDialog group={groupDraft} onClose={() => setGroupDraft(undefined)} onSave={saveGroup} />
      {notice ? <div className="toast" role="alert"><icons.CircleAlert size={18} /><span>{notice}</span><button onClick={() => setNotice('')} aria-label="关闭提示">×</button></div> : null}
    </div>
  )
}
