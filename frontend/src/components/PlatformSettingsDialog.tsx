import { useEffect, useState } from 'react'
import { choosePaths, getPlatformSettings, savePlatformSettings } from '../api'
import { icons } from '../icons'
import type { ExternalAppSettings, PlatformSettings } from '../types'

type Props = { open: boolean; onClose: () => void; onSaved: (message: string) => void }

export function PlatformSettingsDialog({ open, onClose, onSaved }: Props) {
  const [settings, setSettings] = useState<PlatformSettings | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    if (!open) return
    setError('')
    getPlatformSettings().then(setSettings).catch((reason) => setError(reason instanceof Error ? reason.message : '设置加载失败'))
  }, [open])

  if (!open) return null

  const updateApp = (id: string, key: keyof ExternalAppSettings, value: string) => {
    setSettings((current) => current ? {
      ...current,
      externalApps: current.externalApps.map((app) => app.id === id ? { ...app, [key]: value } : app),
    } : current)
  }

  const browse = async (id: string, key: 'executable' | 'workingDirectory' | 'condaExecutable') => {
    try {
      const paths = await choosePaths(key === 'workingDirectory' ? 'directory' : 'file')
      if (paths[0]) updateApp(id, key, paths[0])
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '选择路径失败')
    }
  }

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!settings || saving) return
    setSaving(true)
    setError('')
    try {
      const saved = await savePlatformSettings(settings)
      setSettings(saved)
      onSaved('本机设置已保存，外部工具将使用新路径启动。')
      onClose()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '设置保存失败')
    } finally {
      setSaving(false)
    }
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <form className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="platform-settings-title" onSubmit={submit}>
      <header><div><h2 id="platform-settings-title">本机设置</h2><p>这些路径只保存在当前电脑，换电脑后可单独配置，不会写入项目代码。</p></div><button type="button" className="icon-button" onClick={onClose} aria-label="关闭"><icons.X size={18} /></button></header>
      <div className="settings-content">
        {!settings && !error ? <p className="settings-loading">正在加载设置……</p> : null}
        {settings?.externalApps.map((app) => <section className="external-app-settings" key={app.id}>
          <div className="settings-section-title"><icons.Box size={19} /><span><strong>{app.name}</strong><small>{app.description}</small></span></div>
          <label><span>启动程序 <em>可填绝对路径，或 PATH 中的命令名</em></span><div><input value={app.executable} onChange={(event) => updateApp(app.id, 'executable', event.target.value)} placeholder="例如：/home/user/app 或 python" /><button type="button" onClick={() => void browse(app.id, 'executable')}>浏览</button></div></label>
          <label><span>工作目录 <em>脚本依赖相对路径时填写</em></span><div><input value={app.workingDirectory} onChange={(event) => updateApp(app.id, 'workingDirectory', event.target.value)} placeholder="可留空" /><button type="button" onClick={() => void browse(app.id, 'workingDirectory')}>浏览</button></div></label>
          <label><span>启动参数 <em>每行一个完整参数</em></span><textarea rows={2} value={app.arguments} onChange={(event) => updateApp(app.id, 'arguments', event.target.value)} placeholder="例如：app.py" /></label>
          <div className="settings-two-columns">
            <label><span>Conda 环境 <em>不需要则留空</em></span><input value={app.condaEnvironment} onChange={(event) => updateApp(app.id, 'condaEnvironment', event.target.value)} placeholder="例如：xlabel" /></label>
            <label><span>Conda 可执行文件 <em>留空自动查找</em></span><div><input value={app.condaExecutable} onChange={(event) => updateApp(app.id, 'condaExecutable', event.target.value)} placeholder="例如：~/miniconda3/bin/conda" /><button type="button" onClick={() => void browse(app.id, 'condaExecutable')}>浏览</button></div></label>
          </div>
        </section>)}
        {settings ? <p className="settings-file">配置文件：{settings.configFile}</p> : null}
        {error ? <p className="settings-error" role="alert">{error}</p> : null}
      </div>
      <footer><button type="button" className="cancel-button" onClick={onClose}>取消</button><button type="submit" className="save-button" disabled={!settings || saving}><icons.Check size={16} />{saving ? '保存中……' : '保存设置'}</button></footer>
    </form>
  </div>
}
