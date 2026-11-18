import { useEffect, useRef } from 'react'
import type { SettingsSection, SetupGuide } from '../setupGuidance'

export function SetupGuidanceDialog({ guide, onClose, onSettings }: {
  guide?: SetupGuide; onClose: () => void; onSettings: (section: SettingsSection) => void
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const isOpen = Boolean(guide)
  useEffect(() => {
    const node = dialog.current
    if (isOpen && node && !node.open) node.showModal()
    return () => { if (node?.open) node.close() }
  }, [isOpen])

  return <dialog ref={dialog} className="setup-guidance-dialog" aria-labelledby="setup-guidance-title"
    onCancel={event => { event.preventDefault(); onClose() }}>
    {guide ? <>
      <header><p>功能未能运行 · 配置引导</p><h2 id="setup-guidance-title">{guide.title}</h2></header>
      <div className="setup-guidance-content">
        <p><strong>需要检查的位置：</strong>{guide.location}</p>
        <ol>{guide.steps.map(step => <li key={step}>{step}</li>)}</ol>
        <small>原始原因保留在功能提示或运行日志中。平台不会自动安装依赖，也不会自动重跑任务。</small>
      </div>
      <footer>
        <button type="button" className="outline-button" autoFocus onClick={onClose}>返回</button>
        {guide.settings ? <button type="button" className="save-button" onClick={() => onSettings(guide.settings!)}>
          {guide.settings === 'dependencies' ? '查看依赖检查' : '打开对应设置'}
        </button> : null}
      </footer>
    </> : null}
  </dialog>
}
