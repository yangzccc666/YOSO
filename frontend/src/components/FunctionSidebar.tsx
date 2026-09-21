import { useMemo, useRef, useState } from 'react'
import { icons } from '../icons'
import type { FunctionDefinition, GroupDefinition } from '../types'

type Props = {
  functions: FunctionDefinition[]
  groups: GroupDefinition[]
  selectedId: string
  search: string
  onSearch: (value: string) => void
  onSelect: (id: string) => void
  onEdit: (item: FunctionDefinition) => void
  onCreateGroup: () => void
  onRenameGroup: (group: GroupDefinition) => void
  onDeleteGroup: (group: GroupDefinition) => void
  onReorderGroups: (groupIds: string[]) => void
  onMoveFunction: (itemId: string, groupId: string | null, position?: number) => void
  collapsed: boolean
  onToggleCollapsed: () => void
}

type GroupBlock = {
  id: string | null
  name: string
  group: GroupDefinition | null
  items: FunctionDefinition[]
}

const ungroupedKey = '__ungrouped__'

export function FunctionSidebar(props: Props) {
  const [draggingId, setDraggingId] = useState<string | null>(null)
  const isComposingSearch = useRef(false)
  const normalizedSearch = props.search.trim().toLowerCase()

  const blocks = useMemo<GroupBlock[]>(() => {
    const orderedGroups = [...props.groups].sort((left, right) => left.order - right.order)
    const groupNameById = new Map(orderedGroups.map((group) => [group.id, group.name.toLowerCase()]))
    const visibleFunctions = props.functions.filter((item) => {
      if (!normalizedSearch) return true
      return item.name.toLowerCase().includes(normalizedSearch)
        || (item.groupId ? groupNameById.get(item.groupId)?.includes(normalizedSearch) : '未分组'.includes(normalizedSearch))
    })
    const result: GroupBlock[] = orderedGroups.map((group) => ({
      id: group.id,
      name: group.name,
      group,
      items: visibleFunctions.filter((item) => item.groupId === group.id).sort((left, right) => left.order - right.order),
    }))
    const ungrouped = visibleFunctions.filter((item) => !item.groupId).sort((left, right) => left.order - right.order)
    if (!normalizedSearch || ungrouped.length) result.push({ id: null, name: '未分组', group: null, items: ungrouped })
    return normalizedSearch ? result.filter((block) => block.items.length) : result
  }, [props.functions, props.groups, normalizedSearch])

  const reorderGroup = (groupId: string, direction: -1 | 1) => {
    const ids = [...props.groups].sort((left, right) => left.order - right.order).map((group) => group.id)
    const currentIndex = ids.indexOf(groupId)
    const targetIndex = currentIndex + direction
    if (currentIndex < 0 || targetIndex < 0 || targetIndex >= ids.length) return
    ;[ids[currentIndex], ids[targetIndex]] = [ids[targetIndex], ids[currentIndex]]
    props.onReorderGroups(ids)
  }

  const dropFunction = (event: React.DragEvent, groupId: string | null, position?: number) => {
    event.preventDefault()
    event.stopPropagation()
    const itemId = draggingId || event.dataTransfer.getData('text/plain')
    if (itemId) props.onMoveFunction(itemId, groupId, position)
    setDraggingId(null)
  }

  return (
    <aside className={`function-sidebar ${props.collapsed ? 'is-collapsed' : ''}`}>
      {props.collapsed ? (
        <button className="sidebar-rail-toggle" onClick={props.onToggleCollapsed} title="展开功能栏" aria-label="展开功能栏">
          <icons.PanelLeftOpen size={21} />
        </button>
      ) : (
        <>
          <div className="sidebar-heading">
            <div><h1>功能分组</h1><span>{props.functions.length}</span></div>
            <button className="add-group-button" onClick={props.onCreateGroup}><icons.FolderPlus size={17} />新建分组</button>
          </div>
          <label className="search-box">
            <icons.Search size={18} />
            <input
              defaultValue={props.search}
              inputMode="text"
              lang="zh-CN"
              autoComplete="off"
              spellCheck={false}
              placeholder="搜索功能或分组"
              onCompositionStart={() => { isComposingSearch.current = true }}
              onCompositionEnd={(event) => {
                isComposingSearch.current = false
                props.onSearch(event.currentTarget.value)
              }}
              onChange={(event) => {
                if (!isComposingSearch.current) props.onSearch(event.currentTarget.value)
              }}
            />
          </label>
          <div className="group-display-toolbar">
            <span>功能列表</span>
            <button onClick={props.onToggleCollapsed} title="收起功能栏" aria-label="收起功能栏">
              <icons.PanelLeftClose size={15} />
              收起侧栏
            </button>
          </div>
          <div className="function-list">
            {blocks.map((block) => {
              const collapseKey = block.id || ungroupedKey
              const groupIndex = block.group ? props.groups.findIndex((group) => group.id === block.group!.id) : -1
              return (
                <section className="function-group" key={collapseKey} onDragOver={(event) => event.preventDefault()} onDrop={(event) => dropFunction(event, block.id)}>
                  <div className="group-header">
                    <div className="group-label">
                      <icons.Folder size={17} />
                      <strong title={block.name}>{block.name}</strong>
                      <span>{block.items.length}</span>
                    </div>
                    {block.group ? (
                      <div className="group-actions">
                        <button onClick={() => reorderGroup(block.group!.id, -1)} disabled={groupIndex === 0} title="上移分组" aria-label={`上移 ${block.name}`}><icons.ArrowUp size={14} /></button>
                        <button onClick={() => reorderGroup(block.group!.id, 1)} disabled={groupIndex === props.groups.length - 1} title="下移分组" aria-label={`下移 ${block.name}`}><icons.ArrowDown size={14} /></button>
                        <button onClick={() => props.onRenameGroup(block.group!)} title="重命名分组" aria-label={`重命名 ${block.name}`}><icons.Pencil size={14} /></button>
                        <button className="danger-action" onClick={() => props.onDeleteGroup(block.group!)} title="删除分组" aria-label={`删除 ${block.name}`}><icons.Trash2 size={14} /></button>
                      </div>
                    ) : null}
                  </div>
                  <div className={`group-functions ${draggingId ? 'accepting-drop' : ''}`}>
                  {block.items.map((item, index) => (
                    <div
                      key={item.id}
                      className={`function-item ${props.selectedId === item.id ? 'active' : ''} ${draggingId === item.id ? 'dragging' : ''}`}
                      role="button"
                      tabIndex={0}
                      draggable
                      onClick={() => props.onSelect(item.id)}
                      onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') props.onSelect(item.id) }}
                      onDragStart={(event) => { setDraggingId(item.id); event.dataTransfer.setData('text/plain', item.id); event.dataTransfer.effectAllowed = 'move' }}
                      onDragEnd={() => setDraggingId(null)}
                      onDragOver={(event) => event.preventDefault()}
                      onDrop={(event) => dropFunction(event, block.id, index)}
                    >
                      <span className="drag-handle" title="拖拽排序"><icons.GripVertical size={15} /></span>
                      <span className="function-icon"><icons.FileCog size={18} strokeWidth={1.7} /></span>
                      <span className="function-name" title={item.name}>{item.name}</span>
                      <label className="move-function" title="移动到其他分组" onClick={(event) => event.stopPropagation()}>
                        <icons.FolderInput size={15} />
                        <select aria-label={`移动 ${item.name} 到分组`} value={item.groupId || ''} onChange={(event) => props.onMoveFunction(item.id, event.target.value || null)}>
                          {props.groups.map((group) => <option value={group.id} key={group.id}>{group.name}</option>)}
                          <option value="">未分组</option>
                        </select>
                      </label>
                      <button className="edit-icon" aria-label={`编辑 ${item.name}`} onClick={(event) => { event.stopPropagation(); props.onEdit(item) }}><icons.Pencil size={15} /></button>
                    </div>
                  ))}
                  {!block.items.length ? <p className="empty-group">拖拽功能到这里</p> : null}
                  </div>
                </section>
              )
            })}
            {!blocks.length ? <p className="no-functions">没有匹配的功能或分组</p> : null}
          </div>
          <p className="sidebar-note"><icons.Info size={14} />拖拽可排序，文件夹按钮可快速移动</p>
        </>
      )}
    </aside>
  )
}
