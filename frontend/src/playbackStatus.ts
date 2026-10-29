import type { RemoteInferenceStatus } from './types'

/** A delayed status poll must not undo an acknowledged step or frame counter. */
export function retainNewestPlaybackStatus(current: RemoteInferenceStatus | null, next: RemoteInferenceStatus): RemoteInferenceStatus {
  if (!current || current.id !== next.id) return next
  const before = current.controlGeneration ?? 0, after = next.controlGeneration ?? 0
  if (after < before) return current
  if (after === before) {
    if ((next.frameVersion ?? next.frameCount) < (current.frameVersion ?? current.frameCount)) return current
    if (!current.frameStepPending && next.frameStepPending && next.frameVersion === current.frameVersion) return current
  }
  return next
}
