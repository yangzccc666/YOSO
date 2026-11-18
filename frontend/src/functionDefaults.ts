import type { FunctionDefinition, WorkingValues } from './types'

const privateKey = (key: string) => key !== 'remember_password' && /(?:^|_)(?:password|passphrase|token|secret)$/i.test(key)

function validValues(raw: unknown): WorkingValues {
  const value = raw && typeof raw === 'object' ? raw as Partial<WorkingValues> : {}
  const entries = (item: unknown): Array<[string, unknown]> => item && typeof item === 'object' && !Array.isArray(item) ? Object.entries(item) : []
  return {
    paths: Object.fromEntries(entries(value.paths).filter((entry): entry is [string, string] => typeof entry[1] === 'string')),
    parameters: Object.fromEntries(entries(value.parameters).filter((entry): entry is [string, string | number | boolean] => typeof entry[1] === 'string' || typeof entry[1] === 'boolean' || typeof entry[1] === 'number' && Number.isFinite(entry[1]))),
  }
}

export function withoutSecrets(item: FunctionDefinition, values: WorkingValues): WorkingValues {
  const secretIds = new Set(item.parameters.filter(field => field.type === 'password').map(field => field.id))
  const safe = validValues(values)
  return {
    paths: safe.paths,
    parameters: Object.fromEntries(Object.entries(safe.parameters).map(([key, value]) => [key, secretIds.has(key) || privateKey(key) ? '' : value])),
  }
}

export function defaultValuesFor(item: FunctionDefinition): WorkingValues {
  const saved = validValues(item.savedDefaults?.values)
  return withoutSecrets(item, {
    paths: { ...Object.fromEntries(item.pathFields.map(field => [field.id, ''])), ...saved.paths },
    parameters: { ...Object.fromEntries(item.parameters.map(field => [field.id, field.default])), ...saved.parameters },
  })
}

export function workingValuesFor(item: FunctionDefinition, stored?: string | null): WorkingValues {
  const defaults = defaultValuesFor(item)
  // Explicit user defaults win on a fresh launch, independently of the old
  // embedded-browser port/profile. Temporary edits remain in App's live state.
  if (item.savedDefaults) return defaults
  try {
    const previous = validValues(stored ? JSON.parse(stored) : null)
    return withoutSecrets(item, { paths: { ...defaults.paths, ...previous.paths }, parameters: { ...defaults.parameters, ...previous.parameters } })
  } catch { return defaults }
}

export function trainingParametersSnapshot(values: Record<string, string | number | boolean>): string {
  return JSON.stringify({ ...Object.fromEntries(Object.entries(values).filter(([key]) => !privateKey(key) && key !== 'training_parameters')), device: '' })
}

export function parseTrainingParameters(raw: string | number | boolean | undefined): Record<string, string | number | boolean> {
  if (typeof raw !== 'string') return {}
  try { return validValues({ parameters: JSON.parse(raw) }).parameters } catch { return {} }
}
