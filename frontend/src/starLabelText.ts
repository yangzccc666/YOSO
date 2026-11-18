const LABEL_SEPARATOR = /[,，\r\n]+/
const CLASS_INDEX_PREFIX = /^\d+\s*[:：]\s*/

/** Format pasted labels without sorting, deduplicating or changing their names. */
export function formatStarLabelText(raw: string): string {
  const entries = raw.split(LABEL_SEPARATOR).map(value => value.trim()).filter(Boolean)
  if (!entries.length) throw new Error('请先粘贴类别名称。')
  return entries.map((entry, index) => {
    const label = entry.replace(CLASS_INDEX_PREFIX, '').trim()
    if (!label) throw new Error(`第 ${index + 1} 项只有序号，没有类别名称，请补全后重试。`)
    if (label.includes('\0')) throw new Error('类别名称不能包含空字符。')
    return label
  }).join('\n')
}
