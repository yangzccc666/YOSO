import type { RunHistoryRecord } from '../types'
import { runHistoryLabel } from '../runHistoryLabel'

export function RunHistoryHeading({ record, children }: { record: RunHistoryRecord; children: React.ReactNode }) {
  const label = runHistoryLabel(record)
  return <span className="history-heading">
    <strong title={label.title}>{label.title}</strong>
    {label.parameters ? <span className="history-parameters" title={label.parameters}>{label.parameters}</span> : null}
    {label.context ? <span className="history-source" title={label.context}>{label.context}</span> : null}
    <small>{children}</small>
  </span>
}
