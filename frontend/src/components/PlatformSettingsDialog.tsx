import { useEffect, useRef, useState } from 'react'
import { confirmClose } from '../confirmActions'
import { choosePaths, getPlatformDependencies, getPlatformSettings, savePlatformSettings } from '../api'
import { icons } from '../icons'
import type { ExternalAppSettings, PlatformDependencyReport, PlatformSettings } from '../types'
import type { SettingsSection } from '../setupGuidance'

type Props = { open: boolean; focusSection?: SettingsSection; onClose: () => void; onSaved: (message: string, settings: PlatformSettings) => void }

export function PlatformSettingsDialog({ open, focusSection, onClose: close, onSaved }: Props) {
  const [settings, setSettings] = useState<PlatformSettings | null>(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [dependencies, setDependencies] = useState<PlatformDependencyReport | null>(null)
  const [dependencyError, setDependencyError] = useState('')
  const content = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    let cancelled = false
    setError('')
    setSettings(null)
    setDependencies(null)
    setDependencyError('')
    getPlatformDependencies().then(report => { if (!cancelled) setDependencies(report) }).catch(reason => { if (!cancelled) setDependencyError(String(reason)) })
    getPlatformSettings().then(value => { if (!cancelled) setSettings(value) }).catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : '设置加载失败') })
    return () => { cancelled = true }
  }, [open])

  const settingsReady = Boolean(settings)
  useEffect(() => {
    if (!open || !focusSection) return
    const section = content.current?.querySelector<HTMLElement>(`[data-settings-section="${focusSection}"]`)
    section?.scrollIntoView({ block: 'start' })
    section?.focus({ preventScroll: true })
  }, [open, focusSection, settingsReady])

  if (!open) return null

  const onClose = () => {
    if (!saving && confirmClose('本机设置', '未保存的设置修改将不会保留。')) close()
  }

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
      onSaved('本机设置已保存，外部工具菜单已更新。', saved)
      close()
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : '设置保存失败')
    } finally {
      setSaving(false)
    }
  }

  return <div className="modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose() }}>
    <form className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="platform-settings-title" onSubmit={submit}>
      <header><div><h2 id="platform-settings-title">本机设置</h2><p>这些路径只保存在当前电脑，换电脑后可单独配置，不会写入项目代码。</p></div><button type="button" className="icon-button" onClick={onClose} aria-label="关闭"><icons.X size={18} /></button></header>
      <div className="settings-content" ref={content}>
        <section className="external-app-settings" data-settings-section="dependencies" tabIndex={-1}>
          <div className="settings-section-title"><icons.Info size={19} /><span><strong>分发依赖检查</strong><small>{dependencies ? `${dependencies.system} · 构建类型：${dependencies.profile}` : '正在检查本机依赖，不连接 SSH 或 Docker…'}</small></span></div>
          {dependencies?.checks.map(check => <div className="dependency-check" key={check.name}><strong>{check.name}<em>{({ ready: '已内置／可用', detected: '已发现，待验证', missing: '尚未集成／缺失', external: '需外部环境', remote: '需目标设备环境', optional: '可选外部工具' } as Record<string, string>)[check.status] || check.status}</em></strong><p>{check.detail}</p></div>)}
          {dependencies ? <p className="settings-file">{dependencies.note}</p> : null}
          {dependencyError ? <p role="alert">依赖检查失败：{dependencyError}</p> : null}
        </section>
        {!settings && !error ? <p className="settings-loading">正在加载设置……</p> : null}
        {settings ? <section className="external-app-settings" data-settings-section="ai" tabIndex={-1}>
          <div className="settings-section-title"><icons.Box size={19} /><span><strong>本地 AI 运行时</strong><small>AI 集成版无需配置；标准版可选任意已安装 Ultralytics/PyTorch 的 Python，不要求 conda 或固定环境名。</small></span></div>
          <label><span>AI Python 路径 <em>留空优先使用平台内置环境</em></span><div><input value={settings.aiPython || ''} onChange={(event) => setSettings({ ...settings, aiPython: event.target.value })} placeholder="可选，标准版可指定其他 Python" /><button type="button" onClick={() => { void choosePaths('file').then(paths => { if (paths[0]) setSettings(current => current ? { ...current, aiPython: paths[0] } : current) }).catch(reason => setError(String(reason))) }}>浏览</button></div></label>
        </section> : null}
        {settings ? <section className="external-app-settings" data-settings-section="external" tabIndex={-1}>
          <div className="settings-section-title"><icons.Box size={19} /><span><strong>外部工具管理</strong><small>保存后自动显示在菜单中；删除仅移除入口，不卸载软件或结束已启动的程序。最多40个。</small></span></div>
          <button className="external-app-add" type="button" disabled={saving || settings.externalApps.length >= 40} onClick={() => setSettings(current => current ? { ...current, externalApps: [...current.externalApps, { id: `tool_${crypto.randomUUID().replaceAll('-', '')}`, name: '', description: '', executable: '', workingDirectory: '', arguments: '', condaEnvironment: '', condaExecutable: '' }] } : current)}><icons.Plus size={16} />添加外部工具</button>
          {!settings.externalApps.length ? <p className="settings-file">尚未添加外部工具。</p> : null}
        </section> : null}
        {settings?.externalApps.map((app) => <section className="external-app-settings external-app-card" key={app.id} aria-label={`外部工具配置 ${app.name || '新工具'}`}>
          <div className="settings-section-title"><icons.Box size={19} /><span><strong>{app.name || '新工具'}</strong><small>{app.description || '配置启动方式'}</small></span><button className="external-app-remove" type="button" disabled={saving} onClick={() => { if (window.confirm(`确定移除“${app.name || '新工具'}”的菜单入口吗？\n保存设置后生效，不会删除软件或文件。`)) setSettings(current => current ? { ...current, externalApps: current.externalApps.filter(item => item.id !== app.id) } : current) }}><icons.Trash2 size={15} />移除工具</button></div>
          <label><span>工具名称 <em>显示在外部工具菜单中</em></span><input required maxLength={80} value={app.name} onChange={event => updateApp(app.id, 'name', event.target.value)} placeholder="例如：我的图片查看器" /></label>
          <label><span>工具说明 <em>可选</em></span><input maxLength={300} value={app.description} onChange={event => updateApp(app.id, 'description', event.target.value)} placeholder="简单说明用途" /></label>
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
