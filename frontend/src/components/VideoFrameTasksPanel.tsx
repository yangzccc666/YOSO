import { useEffect, useState } from 'react'
import type { PlatformTaskStatus } from '../types'

type Resources = { cpuCount: number; cpuPercent: number | null; availableMemoryMb: number | null; maxWorkers: number; recommendedWorkers: number; activeWorkers: number; warning: string }

export function VideoFrameTasksPanel({ tasks, onStop }: { tasks: PlatformTaskStatus[]; onStop: (id: string) => void }) {
  const [resources, setResources] = useState<Resources | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let disposed = false
    let timer: number | undefined
    const refresh = async () => {
      try {
        const response = await fetch('/api/video-frames/resources', { cache: 'no-store' })
        const value = await response.json()
        if (!response.ok) throw new Error(value.error || '无法读取资源状态')
        if (!disposed) { setResources(value); setError('') }
      } catch { if (!disposed) setError('暂时无法获取资源状态，运行时仍会进行资源检查。') }
      finally { if (!disposed) timer = window.setTimeout(refresh, 3000) }
    }
    void refresh()
    return () => { disposed = true; if (timer !== undefined) window.clearTimeout(timer) }
  }, [])

  return <section className="workspace-section video-frame-tasks">
    <div className="section-title"><h2>并行任务与机器资源</h2></div>
    <p>可修改输入路径或参数后继续启动新任务；相同配置运行中不重复提交。每批的视频并行数按“同时处理的视频数”设置，0 为自动。</p>
    {resources ? <div className="video-resource-summary" role="status">
      <span>CPU：{resources.cpuCount} 线程{resources.cpuPercent === null ? '' : ` · 占用 ${resources.cpuPercent}%`}</span>
      <span>可用内存：{resources.availableMemoryMb === null ? '未知' : `${(resources.availableMemoryMb / 1024).toFixed(1)} GB`}</span>
      <span>建议并行：{resources.recommendedWorkers} · 正在抽帧：{resources.activeWorkers} 个视频</span>
      {resources.warning ? <strong>{resources.warning}</strong> : null}
    </div> : <p>{error || '正在读取机器资源…'}</p>}
    {tasks.length ? <div className="video-frame-task-list">{tasks.map((task) => <article key={task.id}>
      <header><div><strong>{task.name}</strong><small>{new Date(task.startedAt).toLocaleString('zh-CN')} · {task.id.slice(0, 6)} · {task.status === 'stopping' ? '正在终止' : '运行中 / 等待资源'}</small></div>
        <button className="stop-run-button" onClick={() => onStop(task.id)} disabled={task.status === 'stopping'}>{task.status === 'stopping' ? '正在终止…' : '终止此任务'}</button>
      </header>
      <pre>{task.logs?.join('\n') || '任务正在准备，请稍候…'}</pre>
    </article>)}</div> : <p>暂无运行中的抽帧任务。完成的任务可在下方运行历史中查看。</p>}
  </section>
}
