export type Kind = 'assessment' | 'asset'
export type Node = {id: string; parent: string | null; name: string; kind: 'project' | 'line' | 'station'}
export type Entry = {id: string; node: string; kind: Kind; data: Record<string, string>; revision: number}
export type Field = {id: string; name: string; kind: Kind; format: 'text' | 'multiline' | 'select'; options: string[]}
export type Detail = {nodes: Node[]; records: Entry[]; fields: Field[]}
export type Catalog = {sortMode?: "edited" | "created" | "manual"; projects: Node[]; dimensions: string[]; types: string[]}
export const levels = {project: '项目', line: '产线', station: '工位'}
export const tabs: Record<Kind, string> = {assessment: '接手评估与方案', asset: '路径与资产配置'}
export function lineage(nodes: Node[], id: string): Node[] {
  const result: Node[] = []
  let current = nodes.find(n => n.id === id)
  while (current) {
    result.unshift(current)
    const parent = current.parent
    current = nodes.find(n => n.id === parent)
  }
  return result
}
export function scopeName(nodes: Node[], id: string) {
  return lineage(nodes, id).map(n => n.name).join(' / ')
}
export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message) }
}
export async function api<T>(url: string, payload?: object): Promise<T> {
  const res = await fetch(url, payload ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)} : undefined)
  const data = await res.json()
  if (!res.ok) throw new ApiError(data.error || '读取失败', res.status)
  return data as T
}
export const change = <T = {id: string}>(payload: object) => api<T>('/api/projects/manage', payload)
