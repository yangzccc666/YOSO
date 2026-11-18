/** Keep portalled help inside the viewport, including short desktop windows. */
export function helpTipPosition(anchor: { left: number; top: number; bottom: number }, size: { width: number; height: number }, viewport: { width: number; height: number }) {
  const margin = 12
  const gap = 8
  const below = Math.max(0, viewport.height - anchor.bottom - gap - margin)
  const above = Math.max(0, anchor.top - gap - margin)
  const height = Math.min(size.height, 400, Math.max(0, viewport.height - margin * 2))
  const useBelow = below >= height || below >= above
  const maxHeight = Math.min(400, useBelow ? below : above)
  return {
    left: Math.max(margin, Math.min(anchor.left, viewport.width - size.width - margin)),
    top: useBelow ? anchor.bottom + gap : Math.max(margin, anchor.top - gap - Math.min(height, maxHeight)),
    maxHeight,
  }
}
