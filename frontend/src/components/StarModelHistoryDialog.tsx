import { useEffect, useRef, useState } from 'react'
import { deleteStarModelHistory, getStarModelHistory, saveStarModelHistory } from '../api'
import { icons } from '../icons'
import { matchesStarModelHistory } from '../remoteStarModels'
import type { StarModel, StarModelHistoryRecord } from '../remoteStarModels'

type Props = {
  currentModels: StarModel[]
  onSelect: (model: StarModel) => string | null
  onClose: () => void
}

export function StarModelHistoryDialog({ currentModels, onSelect, onClose }: Props) {
  const dialogRef = useRef<HTMLDialogElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)
  const [records, setRecords] = useState<StarModelHistoryRecord[]>([])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [confirmDelete, setConfirmDelete] = useState('')
  const [reload, setReload] = useState(0)

  useEffect(() => {
    const dialog = dialogRef.current
    dialog?.showModal()
    searchRef.current?.focus()
    return () => dialog?.close()
  }, [])

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError('')
    getStarModelHistory().then(({ models }) => {
      if (!cancelled) setRecords(models)
    }).catch((cause) => {
      if (!cancelled) setError(cause instanceof Error ? cause.message : '无法读取历史模型')
    }).finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [reload])

  const saveCurrent = async () => {
    setBusy(true); setError(''); setNotice('')
    try {
      const models = currentModels.filter((model) => model.path.trim() || model.labels.trim())
      if (!models.length) throw new Error('当前没有可保存的模型，请先填写模型路径和类别名称。')
      const result = await saveStarModelHistory(models)
      setRecords(result.models)
      setNotice('当前模型配置已保存；以后清空或移除当前模型，不会删除历史。')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '保存历史模型失败') }
    finally { setBusy(false) }
  }

  const remove = async (record: StarModelHistoryRecord) => {
    if (confirmDelete !== record.id) { setConfirmDelete(record.id); return }
    setBusy(true); setError(''); setNotice('')
    try {
      await deleteStarModelHistory(record.id)
      setRecords((previous) => previous.filter((row) => row.id !== record.id))
      setConfirmDelete('')
      setNotice('已删除这条历史记录，模型文件和当前模型配置未受影响。')
    } catch (cause) { setError(cause instanceof Error ? cause.message : '删除历史模型失败') }
    finally { setBusy(false) }
  }

  const visible = records.filter((record) => matchesStarModelHistory(record, query))
  return <dialog ref={dialogRef} className="history-modal model-history-modal" aria-modal="true" aria-labelledby="star-model-history-title"
    onCancel={onClose} onClick={(event) => { if (event.target === event.currentTarget) {
      const bounds = event.currentTarget.getBoundingClientRect()
      if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) onClose()
    } }}>
    <header><div><h2 id="star-model-history-title">历史模型库</h2><p>启动推理时自动保存 · 长期保留全部模型配置，不受 5 条运行历史限制</p></div>
      <button type="button" className="icon-button" aria-label="关闭历史模型库" onClick={onClose}><icons.X size={19} /></button></header>
    <div className="model-history-toolbar">
      <label className="model-history-search"><icons.Search size={17} /><input ref={searchRef} type="search" aria-label="搜索历史模型" placeholder="搜索模型文件名、路径或类别名称" value={query} onChange={(event) => { setQuery(event.target.value); setConfirmDelete('') }} /></label>
      <button type="button" disabled={busy || loading} onClick={() => void saveCurrent()}>保存当前配置</button>
    </div>
    <div className="model-history-feedback" aria-live="polite">
      {error && <p className="history-error" role="alert">{error}</p>}
      {notice && <p className="model-history-notice" role="status">{notice}</p>}
    </div>
    <div className="history-content model-history-content" aria-busy={loading}>
      {loading ? <p className="history-empty">正在加载历史模型…</p>
        : !visible.length ? <p className="history-empty">{records.length ? '没有找到匹配的模型，请更换关键词。' : '暂无历史模型。运行推理后会自动记录，也可以先保存当前配置。'}</p>
          : <ol className="model-history-list">{visible.map((record) => <li className="model-history-record" key={record.id}>
            <div className="model-history-title"><strong>{record.path.split(/[\\/]/).pop()}</strong><span className={record.available ? 'model-file-available' : 'model-file-missing'}>{record.available ? '文件可用' : '文件已移动或不存在'}</span></div>
            <p className="model-history-path">{record.path}</p>
            <div className="model-history-labels"><span>类别名称（按模型顺序）</span><p>{record.labels}</p></div>
            <div className="model-history-meta"><span>置信度 {record.conf} · 启动 {record.useCount} 次 · {record.lastUsedAt ? `最近使用 ${new Date(record.lastUsedAt).toLocaleString('zh-CN', { hour12: false })}` : '手动保存，尚未运行'}</span>
              <div className="model-history-row-actions">
                <button type="button" className="model-history-delete" disabled={busy} onClick={() => void remove(record)} title="仅删除历史记录，不删除模型文件">{confirmDelete === record.id ? '确认删除' : '删除记录'}</button>
                <button type="button" disabled={busy || !record.available} onClick={() => {
                  const selectionError = onSelect({ path: record.path, labels: record.labels, conf: record.conf })
                  if (selectionError) setError(selectionError)
                  else onClose()
                }}>使用此模型</button>
              </div>
            </div>
          </li>)}</ol>}
    </div>
    <footer><span>显示 {visible.length} / {records.length} 条 · 优先填入空位，否则新增模型，不覆盖其他配置</span><button type="button" disabled={busy || loading} onClick={() => setReload((value) => value + 1)}>刷新</button></footer>
  </dialog>
}
