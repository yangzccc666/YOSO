import { useEffect, useId, useState } from 'react'
import { getSshConnectionHistory, SSH_HISTORY_UPDATED_EVENT } from '../api'
import type { SshConnectionProfile } from '../types'

type Props = {
  host: string
  placeholder?: string
  disabled?: boolean
  onHostChange: (host: string) => void
  onSelect: (profile: SshConnectionProfile) => void
}

export function SshConnectionHistoryInput({ host, placeholder, disabled, onHostChange, onSelect }: Props) {
  const listId = useId()
  const [profiles, setProfiles] = useState<SshConnectionProfile[]>([])

  useEffect(() => {
    let cancelled = false
    const load = () => {
      getSshConnectionHistory().then((items) => {
        if (!cancelled) setProfiles(items)
      }).catch(() => undefined)
    }
    load()
    window.addEventListener(SSH_HISTORY_UPDATED_EVENT, load)
    return () => {
      cancelled = true
      window.removeEventListener(SSH_HISTORY_UPDATED_EVENT, load)
    }
  }, [])

  const changeHost = (value: string) => {
    onHostChange(value)
    const profile = profiles.find((item) => item.host === value)
    if (profile) onSelect(profile)
  }

  return <>
    <input list={listId} value={host} disabled={disabled} placeholder={placeholder || (profiles.length ? '输入或选择历史服务器' : '请输入服务器 IP / 主机名')} onChange={(event) => changeHost(event.target.value)} />
    <datalist id={listId}>{profiles.map((profile) => <option value={profile.host} key={`${profile.username}@${profile.host}:${profile.port}`} label={`${profile.username} · 端口 ${profile.port}${profile.passwordRemembered ? ' · 已保存密码' : ''}`} />)}</datalist>
  </>
}
