import { useEffect, useRef, useState } from 'react'
import { icons } from '../icons'
import type { GroupDefinition } from '../types'

type Props = {
  group: GroupDefinition | null | undefined
  onClose: () => void
  onSave: (name: string) => Promise<void>
}

export function GroupDialog({ group, onClose, onSave }: Props) {
  const [name, setName] = useState('')
  const [saving, setSaving] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const open = group !== undefined

  useEffect(() => {
    if (!open) return
    setName(group?.name || '')
    requestAnimationFrame(() => inputRef.current?.focus())
  }, [group, open])

  if (!open) return null

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!name.trim() || saving) return
    setSaving(true)
    try {
      await onSave(name.trim())
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
      <form className="group-modal" onSubmit={submit} role="dialog" aria-modal="true" aria-labelledby="group-dialog-title">
        <header>
          <div>
            <h2 id="group-dialog-title">{group ? '重命名分组' : '新建分组'}</h2>
            <p>分组只负责整理功能，不会改变数据处理逻辑。</p>
          </div>
          <button type="button" className="icon-button" onClick={onClose} aria-label="关闭"><icons.X size={18} /></button>
        </header>
        <label className="group-name-field">
          <span>分组名称</span>
          <input ref={inputRef} value={name} maxLength={30} onChange={(event) => setName(event.target.value)} placeholder="例如：视频处理" />
        </label>
        <footer>
          <button type="button" className="cancel-button" onClick={onClose}>取消</button>
          <button type="submit" className="save-button" disabled={!name.trim() || saving}><icons.Check size={16} />{saving ? '保存中' : '保存'}</button>
        </footer>
      </form>
    </div>
  )
}
