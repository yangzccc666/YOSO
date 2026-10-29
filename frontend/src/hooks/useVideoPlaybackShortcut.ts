import { useEffect, useRef } from 'react'
import type { RefObject } from 'react'
import { getVideoPlaybackAction } from '../videoPlaybackShortcut'

/** One stable subscription per mounted player; polling never duplicates it. */
export function useVideoPlaybackShortcut(
  previewRef: RefObject<HTMLDivElement>, enabled: boolean, busy: boolean, toggle: () => Promise<void>,
  stepEnabled = false, step?: (direction: -1 | 1) => Promise<boolean | void>,
) {
  const current = useRef({ enabled, busy, toggle, stepEnabled, step })
  useEffect(() => { current.current = { enabled, busy, toggle, stepEnabled, step } }, [enabled, busy, toggle, stepEnabled, step])
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      const action = getVideoPlaybackAction(event, previewRef.current)
      if (!action || (action === 'toggle' ? !current.current.enabled : !current.current.stepEnabled)) return
      // Consume repeats and pending requests too, so holding Space cannot scroll
      // the page or enqueue multiple remote playback commands.
      event.preventDefault()
      if (event.repeat || current.current.busy) return
      if (action === 'toggle') void current.current.toggle()
      else void current.current.step?.(action === 'backward' ? -1 : 1)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [previewRef])
}
