import type { TensorRTQueueStatus } from '../types'
import { tensorRTDispatchStatus } from '../tensorRTQueue'
import { HelpTip } from './HelpTip'

type Props = {
  queue: TensorRTQueueStatus | null; busy: boolean
  onPause: (paused: boolean) => void; onRemove: (id: string) => void; onStop: (id: string) => void; onClear: () => void
  onReconnect: (id: string) => void
}
const states = { queued: '等待中', running: '正在构建', stopping: '正在终止', disconnected: '待重新连接', completed: '已完成', failed: '失败', stopped: '已终止' }

export function TensorRTQueuePanel({ queue, busy, onPause, onRemove, onStop, onClear, onReconnect }: Props) {
  const dispatch = tensorRTDispatchStatus(queue)
  return <section className="trt-queue-panel" aria-label="TensorRT 构建队列">
    <header><strong className="help-heading">串行构建队列<HelpTip label="构建队列使用说明"><p>加入队列即开启自动执行，当前完成后自动执行下一项，可切换页面。手动暂停后点击“继续队列”或再次加入任务可恢复；失败、终止、缺少凭据时会暂停并明确显示原因。</p><p>任务和等待队列会保存。重开后只有等待项时自动调度（手动暂停和异常除外）；有待重连任务时先确认状态，重连确认完成后自动继续，不会重复构建。未记住密码时请在上方选择对应设备并输入密码。</p><p>成功任务写入运行历史后自动移出队列，可在本功能底部或顶部“运行历史”查看结果和日志；不删除远端引擎文件。等待项最多 200 项，另保留最近 5 条失败／终止记录；历史保存失败时保留成功记录。</p></HelpTip></strong><span>运行 {queue?.runningCount || 0} 项 · 等待 {queue?.queuedCount || 0} 项{queue?.disconnectedCount ? ` · 待重连 ${queue.disconnectedCount} 项` : ''}{queue?.paused ? ' · 已暂停后续' : ''}</span></header>
    <div className={`trt-dispatch-status ${dispatch.blocked ? 'is-blocked' : 'is-auto'}`} role="status" aria-label="构建调度状态"><strong className="help-heading">{dispatch.title}{!dispatch.blocked ? <HelpTip label="构建调度说明">{dispatch.detail}</HelpTip> : null}</strong>{dispatch.blocked ? <span>{dispatch.detail}</span> : null}</div>
    <div className="trt-queue-actions">
      <button onClick={() => onPause(!queue?.paused)} disabled={busy || !queue || queue.closed || Boolean(queue.paused && queue.disconnectedCount)}>{queue?.paused ? '继续队列' : '暂停后续'}</button>
      <button onClick={onClear} disabled={busy || !queue?.queuedCount}>清空等待</button>
    </div>
    {queue?.paused && queue.pauseReason ? <p className="trt-queue-warning" role="status">{queue.pauseReason}</p> : null}
    <div className="trt-queue-list">
      {queue?.items.length ? queue.items.map((item, index) => <article key={item.id} data-queue-id={item.id}>
        <header><strong>{index + 1}. {item.source}</strong><span className={`trt-queue-state ${item.status}`}>{item.status === 'queued' ? queue.paused ? '等待中（已暂停）' : '等待自动执行' : states[item.status]}</span></header>
        <small>{item.username}@{item.host}:{item.port} · {item.buildMode}</small>
        <p>{item.message}</p>
        <div className="trt-queue-actions">{item.status === 'running' || item.status === 'stopping'
          ? <button disabled={busy || item.status === 'stopping'} onClick={() => onStop(item.id)}>终止当前</button>
          : item.status === 'disconnected' ? <><button disabled={busy || queue.closed} onClick={() => onReconnect(item.id)}>重新连接 / 检查结果</button><button disabled={busy} onClick={() => onRemove(item.id)}>仅移除本地记录</button></>
          : <button disabled={busy} onClick={() => onRemove(item.id)}>{item.status === 'queued' ? '移出等待队列' : '删除队列记录'}</button>}</div>
        {item.remoteDirectory ? <small>远端任务目录：{item.remoteDirectory}</small> : null}
        {item.outputs.length ? <p>结果：{item.outputs.join('；')}</p> : null}
        {item.logs.length ? <details><summary>查看此任务日志</summary><pre>{item.logs.join('\n')}</pre></details> : null}
      </article>) : <p className="section-empty">暂无构建任务。填写远端 ONNX 路径并加入队列即可开始。</p>}
    </div>
  </section>
}
