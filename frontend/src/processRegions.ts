export type Region = [number, number, number, number]
export type RegionConfig = { version: 1; root: string; regions: Record<string, Region> }
export type RegionPreview = { root: string; folders: string[]; folderKey: string; sampleName: string; width: number; height: number; image: string }

export function validRegion(value: unknown): value is Region {
  return Array.isArray(value) && value.length === 4 && value.every(n => typeof n === 'number' && Number.isFinite(n))
    && value[0] >= 0 && value[1] >= 0 && value[2] <= 1 && value[3] <= 1 && value[0] < value[2] && value[1] < value[3]
}

export function readRegions(raw: string): RegionConfig | null {
  try {
    const config = JSON.parse(raw)
    if (config?.version !== 1 || typeof config.root !== 'string' || !config.regions || typeof config.regions !== 'object' || Array.isArray(config.regions)) return null
    if (!Object.values(config.regions).every(validRegion)) return null
    return config
  } catch { return null }
}

export function rectangle(a: [number, number], b: [number, number]): Region {
  const clamp = (n: number) => Math.max(0, Math.min(1, n))
  return [clamp(Math.min(a[0], b[0])), clamp(Math.min(a[1], b[1])), clamp(Math.max(a[0], b[0])), clamp(Math.max(a[1], b[1]))]
}
