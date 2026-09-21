import { useCallback, useEffect, useState } from 'react'
import { getRunHistory } from '../api'
import { icons } from '../icons'
import type { RunHistoryRecord } from '../types'

const statusLabels = { completed: '已完成', failed: '失败', stopped: '已终止' }

function displayTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString('zh-CN', { hour12: false })
}

export function RunHistoryDialog({ onClose }: { onClose: () => void }) {
  const [records, setRecords] = useState<RunHistoryRecord[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const refresh = useCallback(async () => {
    try {
      const latest = await getRunHistory()
      setRecords(latest)
      setError('')
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '读取运行历史失败')
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

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <section className="history-modal" role="dialog" aria-modal="true" aria-labelledby="history-title">
      <header>
        <div><h2 id="history-title">最近运行记录</h2><p>只保留最近完成的 5 条；第 6 条完成时自动删除最早的一条。</p></div>
        <button className="icon-button" onClick={onClose} aria-label="关闭运行历史"><icons.X size={18} /></button>
      </header>
      <div className="history-content">
        {error ? <div className="history-error" role="alert">{error}<button onClick={() => void refresh()}>重试</button></div> : null}
        {loading ? <p className="history-empty">正在读取运行记录…</p> : !records.length ? <p className="history-empty">还没有完成的运行。任务结束后会自动显示在这里。</p> :
          <ol className="history-list">{records.map((record) => <li key={record.id}>
            <details className="history-entry">
              <summary>
                <span className={`history-status ${record.status}`}>{statusLabels[record.status]}</span>
                <span className="history-heading"><strong>{record.name}</strong><small>{displayTime(record.finishedAt)} · {record.kind === 'remote' ? '远程' : '本地'}</small></span>
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
          </li>)}</ol>}
      </div>
      <footer><span>记录保存在本机 runtime/run_history.json</span><button onClick={() => void refresh()}>刷新</button></footer>
    </section>
  </div>
}
