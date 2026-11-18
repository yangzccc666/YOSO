/** Use the same newline/JSON-list input as the batch exporter; preserve model order. */
export function parseExportModelPaths(raw: string): string[] {
  const text = raw.trim()
  if (!text) return []
  let candidates: unknown
  try { candidates = text.startsWith('[') ? JSON.parse(text) : text.split(/\r?\n/) }
  catch { throw new Error('PT 路径格式不正确，请每行填写一个路径。') }
  if (!Array.isArray(candidates) || candidates.some((value) => typeof value !== 'string')) throw new Error('PT 路径必须是文本，每行一个，或一个路径数组。')
  return [...new Set(candidates.map((value: string) => value.trim()).filter(Boolean))]
}
