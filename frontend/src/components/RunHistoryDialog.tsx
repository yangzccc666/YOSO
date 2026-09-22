import { useCallback, useEffect, useState } from 'react'
import { deleteRunHistory, getRunHistory } from '../api'
import { icons } from '../icons'
import type { RunHistoryRecord } from '../types'

const statusLabels = { completed: '已完成', failed: '失败', stopped: '已终止' }

function displayTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function executionLabel(kind: RunHistoryRecord['kind']) {
  if (kind === 'remote-training') return '远程训练'
  if (kind === 'remote-build') return '远程构建'
  return kind === 'remote' ? '远程' : '本地'
}

export function RunHistoryDialog({ onClose }: { onClose: () => void }) {
  const [records, setRecords] = useState<RunHistoryRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deletingId, setDeletingId] = useState('')

  const refresh = useCallback(async () => {
    try {
      const latest = await getRunHistory()
      setRecords(latest)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '读取全局运行历史失败')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 3000)
    const onKeyDown = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose() }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('keydown', onKeyDown)
    }
  }, [onClose, refresh])

  const remove = async (record: RunHistoryRecord) => {
    if (!window.confirm(`确定删除“${record.name}”的这条运行历史吗？\n只删除历史记录，不会删除任务输出文件。`)) return
    setDeletingId(record.id)
    try {
      await deleteRunHistory(record.id)
      setRecords((current) => current.filter((item) => item.id !== record.id))
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '删除运行历史失败')
    } finally {
      setDeletingId('')
    }
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="history-modal" role="dialog" aria-modal="true" aria-labelledby="history-title">
      <header>
        <div><h2 id="history-title">全局运行历史</h2><p>汇总所有功能最近完成的 5 条记录；各功能页面仍独立保留自己的 5 条记录。</p></div>
        <button className="icon-button" onClick={onClose} aria-label="关闭全局运行历史"><icons.X size={18} /></button>
      </header>
      <div className="history-content">
        {error ? <div className="history-error" role="alert"><span>{error}</span><button onClick={() => void refresh()}>重试</button></div> : null}
        {loading ? <p className="history-empty">正在读取全局运行记录…</p> : !records.length ? <p className="history-empty">还没有完成的运行。任务结束后会自动显示在这里。</p> :
          <ol className="history-list">{records.map((record) => <li className="history-row" key={record.id}>
            <details className="history-entry">
              <summary>
                <span className={`history-status ${record.status}`}>{statusLabels[record.status]}</span>
                <span className="history-heading"><strong>{record.name}</strong><small>{displayTime(record.finishedAt)} · {executionLabel(record.kind)}</small></span>
                <icons.ChevronDown size={17} />
              </summary>
              <div className="history-detail">
                <p>{record.message || statusLabels[record.status]}</p>
                <div><span>开始</span><strong>{displayTime(record.startedAt)}</strong></div>
                <div><span>结束</span><strong>{displayTime(record.finishedAt)}</strong></div>
                {Object.entries(record.details).map(([label, value]) => <div key={label}><span>{label}</span><strong title={value}>{value}</strong></div>)}
                {record.outputs.length ? <div><span>输出位置</span><strong>{record.outputs.join('\n')}</strong></div> : null}
                <details className="history-log"><summary>查看运行日志（{record.logs.length} 条）</summary><pre>{record.logs.length ? record.logs.join('\n') : record.message || '没有更多日志'}</pre></details>
              </div>
            </details>
            <button className="history-delete" type="button" onClick={() => void remove(record)} disabled={deletingId === record.id} aria-label={`删除 ${record.name} 的这条运行历史`} title="只删除历史记录"><icons.Trash2 size={15} />{deletingId === record.id ? '删除中' : '删除'}</button>
          </li>)}</ol>}
      </div>
      <footer><span>全局展示最近 5 条 · 数据保存在本机 runtime/run_history.json</span><button onClick={() => void refresh()}>刷新</button></footer>
    </section>
  </div>
}
