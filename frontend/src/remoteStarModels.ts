export type StarModel = {
  path: string
  labels: string
  conf: number
}

export type StarModelMergeAction = 'added' | 'updated' | 'replaced-empty' | 'full'

export function parseStarModels(
  value: unknown,
  legacyPath: string,
  legacyLabels: string,
  legacyConf: number,
): StarModel[] {
  try {
    const parsed: unknown = JSON.parse(String(value || '[]'))
    if (Array.isArray(parsed) && parsed.length) {
      const models = parsed.flatMap((item): StarModel[] => {
        if (item === null || typeof item !== 'object' || typeof item.path !== 'string') return []
        const confidence = Number(item.conf ?? legacyConf)
        return [{
          path: item.path,
          labels: typeof item.labels === 'string' ? item.labels : '',
          conf: Number.isFinite(confidence) ? confidence : legacyConf,
        }]
      })
      if (models.length) return models
    }
  } catch { /* use saved single-model values */ }
  return [{ path: legacyPath, labels: legacyLabels, conf: legacyConf }]
}

export function upsertStarModel(
  models: StarModel[],
  incoming: StarModel,
  limit = 8,
): { models: StarModel[]; action: StarModelMergeAction } {
  const matchingIndex = models.findIndex((model) => model.path.trim() === incoming.path.trim())
  if (matchingIndex >= 0) {
    return {
      models: models.map((model, index) => index === matchingIndex
        ? { ...model, path: incoming.path, labels: incoming.labels }
        : model),
      action: 'updated',
    }
  }

  if (models.length === 1 && !models[0].path.trim() && !models[0].labels.trim()) {
    return { models: [incoming], action: 'replaced-empty' }
  }
  if (models.length >= limit) return { models, action: 'full' }
  return { models: [...models, incoming], action: 'added' }
}
