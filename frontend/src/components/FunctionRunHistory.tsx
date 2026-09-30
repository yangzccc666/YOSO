import { useCallback, useEffect, useRef, useState } from 'react'
import { deleteRunHistory, getRunHistory } from '../api'
import { icons } from '../icons'
import type { RunHistoryRecord } from '../types'

const statusLabels = { running: '正在运行', completed: '已完成', failed: '失败', stopped: '已终止' }

function displayTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

function executionLabel(kind: RunHistoryRecord['kind']) {
  if (kind === 'remote-build') return '远端后台'
  if (kind === 'remote-training') return '远程训练'
  return kind === 'remote' ? '远程' : '本地'
}

export function FunctionRunHistory({ functionId }: { functionId: string }) {
  const [records, setRecords] = useState<RunHistoryRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [deletingId, setDeletingId] = useState('')
  const requestVersion = useRef(0)
  const runningCount = records.filter((record) => record.status === 'running').length
  const finishedCount = records.length - runningCount

  const refresh = useCallback(async () => {
    const version = ++requestVersion.current
    try {
      const latest = await getRunHistory(functionId)
      if (version !== requestVersion.current) return
      setRecords(latest)
      setError('')
    } catch (reason) {
      if (version !== requestVersion.current) return
      setError(reason instanceof Error ? reason.message : '读取运行历史失败')
    } finally {
      if (version === requestVersion.current) setLoading(false)
    }
  }, [functionId])

  useEffect(() => {
    void refresh()
    const timer = window.setInterval(() => void refresh(), 3000)
    return () => {
      requestVersion.current += 1
      window.clearInterval(timer)
    }
  }, [refresh])

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

  return <section className="workspace-section function-history-section">
    <header className="function-history-header">
      <div className="section-title"><icons.Clock3 size={21} /><h2>本功能运行历史</h2><span>{runningCount ? `进行中 ${runningCount} · ` : ''}已结束 {finishedCount}/5 条</span></div>
      <button type="button" onClick={() => void refresh()} disabled={loading}>刷新</button>
    </header>
    {error ? <div className="history-error" role="alert"><span>{error}</span><button onClick={() => void refresh()}>重试</button></div> : null}
    {loading ? <p className="history-empty">正在读取本功能的运行记录…</p> : !records.length ? <p className="history-empty">本功能还没有完成的运行。成功、失败或手动终止后都会记录在这里。</p> :
      <ol className="history-list">{records.map((record) => <li className="history-row" key={record.id}>
        <details className="history-entry">
          <summary>
            <span className={`history-status ${record.status}`}>{statusLabels[record.status]}</span>
            <span className="history-heading"><strong>{record.name}</strong><small>{record.status === 'running' ? `开始于 ${displayTime(record.startedAt)}` : displayTime(record.finishedAt)} · {executionLabel(record.kind)}</small></span>
            <icons.ChevronDown size={17} />
          </summary>
          <div className="history-detail">
            <p>{record.message || statusLabels[record.status]}</p>
            <div><span>开始</span><strong>{displayTime(record.startedAt)}</strong></div>
            {record.status !== 'running' ? <div><span>结束</span><strong>{displayTime(record.finishedAt)}</strong></div> : null}
            {Object.entries(record.details).map(([label, value]) => <div key={label}><span>{label}</span><strong title={value}>{value}</strong></div>)}
            {record.outputs.length ? <div><span>输出位置</span><strong>{record.outputs.join('\n')}</strong></div> : null}
            <details className="history-log"><summary>查看运行日志（{record.logs.length} 条）</summary><pre>{record.logs.length ? record.logs.join('\n') : record.message || '没有更多日志'}</pre></details>
          </div>
        </details>
        {record.status !== 'running' ? <button className="history-delete" type="button" onClick={() => void remove(record)} disabled={deletingId === record.id} aria-label={`删除 ${record.name} 的这条运行历史`} title="只删除历史记录"><icons.Trash2 size={15} />{deletingId === record.id ? '删除中' : '删除'}</button> : <span className="history-running-note">不占历史名额</span>}
      </li>)}</ol>}
    <p className="function-history-note">正在进行的模型训练会持续保留且不占名额；已结束任务独立保留最近 5 条。</p>
  </section>
}
