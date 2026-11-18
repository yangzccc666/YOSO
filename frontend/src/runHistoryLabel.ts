import type { RunHistoryRecord } from './types'

const pathName = (value: string) => value.trim().replace(/^['"]|['"]$/g, '').replace(/[\\/]+$/, '').split(/[\\/]/).pop() || value
const lines = (value: string) => value.split(/\r?\n/).map(line => line.trim()).filter(Boolean)

/** Read-only labels also work for histories written before summaries existed. */
export function runHistoryLabel(record: RunHistoryRecord) {
  const details = record.details || {}
  const training = record.functionId === 'yolo_model_training' || record.kind === 'remote-training'
  const inputEntries = Object.entries(details).filter(([key, value]) => value.trim()
    && /视频|图片|输入|待标注|权重|模型|数据配置|源文件/.test(key)
    && !/数量|总数|已保存|模式|精度|类型|类别|架构|保存|输出|方式|参数/.test(key))
  inputEntries.sort(([a], [b]) => {
    const priority = (key: string) => /视频|图片|数据配置|待标注/.test(key) ? 0 : 1
    return priority(a) - priority(b)
  })
  const paths = [...new Set(inputEntries.flatMap(([, value]) => lines(value)))]
  const fallback = record.outputs?.filter(Boolean) || []
  const sources = paths.length ? paths : fallback
  const names = [...new Set(sources.map(pathName))]
  let title = record.name || '运行任务'
  // Training names already contain the user/generated run name. Keep it first.
  if (!(training && title.includes('·'))) {
    const fresh = names.filter(name => !title.includes(name))
    if (fresh.length) title += ` · ${fresh.slice(0, 2).join(' + ')}${fresh.length > 2 ? ` 等${fresh.length}项` : ''}`
    else if (!sources.length && !title.includes('·')) title += ` · #${record.id.slice(-6)}`
  }
  const context = sources.slice(0, 2).join(' ｜ ') + (sources.length > 2 ? ` ｜ 共${sources.length}项` : '')
  const parameters = details['关键参数'] || [details['构建精度'], details['自动标注模式'], details['图片保存方式']].filter(Boolean).join(' · ')
  return { title, context, parameters }
}
