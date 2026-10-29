/** Keep one path per line; quoted copied paths and repeated picker additions are safe. */
export function appendFolderPaths(current: string, incoming: string[]): string {
  const paths = new Set<string>()
  for (const raw of [...current.split(/\r?\n/), ...incoming]) {
    let value = raw.trim()
    if (value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) value = value.slice(1, -1).trim()
    if (value) paths.add(value)
  }
  return [...paths].join('\n')
}
