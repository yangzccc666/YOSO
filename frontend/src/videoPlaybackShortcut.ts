/** Only claim Space when it belongs to the visible video, not another control. */
export function getVideoPlaybackAction(event: KeyboardEvent, preview: HTMLElement | null): 'toggle' | 'backward' | 'forward' | null {
  const action = [' ', 'Spacebar', 'Space'].includes(event.key) ? 'toggle'
    : [',', '，'].includes(event.key) ? 'backward' : ['.', '。', '．'].includes(event.key) ? 'forward' : null
  if (!preview || !preview.isConnected || event.defaultPrevented || event.isComposing || event.keyCode === 229
    || event.ctrlKey || event.metaKey || event.altKey || event.shiftKey
    || !action) return null

  const document = preview.ownerDocument
  if (document.hidden || document.querySelector('[aria-modal="true"], dialog[open]')) return null
  const target = event.target && (event.target as Element).nodeType === 1
    ? event.target as Element : document.activeElement
  if (!target) return action
  if ((target as HTMLElement).isContentEditable
    || target.closest('textarea, select, [role="textbox"], [role="searchbox"], [role="combobox"], [role="spinbutton"]')) return null

  const input = target.closest('input')
  // Space also works after seeking, but must never edit a form field or checkbox.
  if (input && !(input.type === 'range' && preview.contains(input))) return null
  const control = target.closest('button, a[href], summary, [role="button"], [role="menuitem"], [role="slider"], [role="checkbox"], [role="radio"], [role="switch"]')
  return !control || preview.contains(control) ? action : null
}

export function shouldToggleVideoOnSpace(event: KeyboardEvent, preview: HTMLElement | null): boolean {
  return getVideoPlaybackAction(event, preview) === 'toggle'
}
