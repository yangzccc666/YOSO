import { Fragment } from 'react'
import { icons } from '../icons'
import { InferenceVideoHistory } from './InferenceVideoHistory'
import { VideoClipOptions } from './VideoClipOptions'
import { CLIP_REPLACE_SOURCE } from '../videoClipOptions'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import { RemoteInferencePanel } from './RemoteInferencePanel'
import { LocalPTInferencePanel } from './LocalPTInferencePanel'
import { RemoteYoloDatasetPanel } from './RemoteYoloDatasetPanel'
import { RemoteCalibrationPanel } from './RemoteCalibrationPanel'
import { YoloTrainingPanel } from './YoloTrainingPanel'
import { RemoteTensorRTBuildPanel } from './RemoteTensorRTBuildPanel'
import { LocalStarPackagePanel } from './LocalStarPackagePanel'
import { FunctionRunHistory } from './FunctionRunHistory'
import { WorkflowTodoPanel } from './WorkflowTodoPanel'
import { SshConnectionHistoryInput } from './SshConnectionHistoryInput'
import { VideoFrameTasksPanel } from './VideoFrameTasksPanel'
import { JetsonModelLabels } from './JetsonModelLabels'
import { HelpTip } from './HelpTip'
import { AutoAnnotationPanel } from './AutoAnnotationPanel'
import { DedupRegionSelector } from './DedupRegionSelector'
import { CollapsibleSection } from './CollapsibleSection'
import type { FunctionDefinition, FunctionParameter, PlatformTaskStatus, SshConnectionProfile, WorkingValues, YoloTrainingRecommendation } from '../types'
import type { WorkflowTodo } from '../workflowTodos'

type Props = {
  item: FunctionDefinition | null
  values: WorkingValues
  output: string
  running: boolean
  activeTasks: PlatformTaskStatus[]
  onEdit: () => void
  onSaveDefaults: () => void
  onLoadDefaults: () => void
  savingDefaults: boolean
  defaultsVersion: number
  defaultsFeedback: string
  onBrowse: (fieldId: string, mode: string) => void
  onPathChange: (fieldId: string, value: string) => void
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  onAddPath: () => void
  onAddParameter: () => void
  onRun: (parameterOverrides?: Record<string, string | number | boolean>) => void
  onStop: (taskId: string) => void
  yoloTrainingRecommendation: YoloTrainingRecommendation | null
  workflowTodos: WorkflowTodo[]
  onApplyWorkflowTodo: (item: WorkflowTodo) => void
  onDeleteWorkflowTodo: (id: string) => void
  onWorkflowComplete: (handlerId: string) => void
}

function ParameterValue({ field, value, onChange, sshHistory }: { field: FunctionParameter; value: string | number | boolean; onChange: (value: string | number | boolean) => void; sshHistory?: { onSelect: (profile: SshConnectionProfile) => void } }) {
  if (field.id === 'star_labels') return <textarea rows={5} value={String(value)} onChange={(event) => onChange(event.target.value)} placeholder={'MARK\nOKP\nNGO'} />
  if (field.id === 'clip_output_name') return <div className="clip-output-name"><input value={String(value)} onChange={(event) => onChange(event.target.value)} placeholder="例如：TCL_工位1_光线变化_训练素材" /><HelpTip label="裁剪输出命名说明">只填文件名，无需填写路径或扩展名。留空沿用自动命名；批量追加原视频名，训练/测试分别追加 _train / _test，重名加序号，不覆盖文件。</HelpTip></div>
  if (field.id === 'remove_time_ranges') return <textarea className="clip-remove-ranges" rows={5} value={String(value)} onChange={(event) => onChange(event.target.value)} placeholder={'00:10-00:20\n01:30-02:00\n02:50-'} />
  if (sshHistory && (field.id === 'host' || field.id === 'remote_host')) return <SshConnectionHistoryInput host={String(value)} onHostChange={onChange} onSelect={sshHistory.onSelect} />
  if (field.type === 'boolean') return <label className="switch-control"><input type="checkbox" checked={Boolean(value)} onChange={(event) => onChange(event.target.checked)} /><span /><em>{value ? '开启' : '关闭'}</em></label>
  if (field.type === 'select') return <span className="workspace-select"><select value={String(value)} onChange={(event) => onChange(event.target.value)}><option value="">请选择</option>{field.options.map((option) => <option value={option} key={option}>{option}</option>)}</select><icons.ChevronDown size={16} /></span>
  const passwordRemembered = field.type === 'password' && value === REMEMBERED_PASSWORD_VALUE
  const displayedValue = passwordRemembered ? '已保存密码' : String(value)
  const placeholder = field.id === 'labels'
    ? '必填，例如：person,car'
    : field.id === 'remote_password'
      ? '请输入本次 SSH 连接密码'
    : field.type === 'password'
      ? '首次连接或未保存时输入'
      : '请输入参数值'
  return <input type={field.type === 'number' ? 'number' : field.type === 'password' ? 'password' : 'text'} autoComplete={field.type === 'password' ? 'current-password' : undefined} value={displayedValue} placeholder={placeholder} onFocus={() => { if (passwordRemembered) onChange('') }} onChange={(event) => onChange(field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
}

export function FunctionWorkspace(props: Props) {
  if (!props.item) return <main className="empty-workspace"><span><icons.Box size={34} /></span><h2>暂无可用功能</h2><p>后续接入新的处理脚本后，功能会显示在左侧分组中。</p></main>
  const item = props.item
  const isYoloDatasetWorkspace = item.handlerId === 'yolo.split_dataset'
  const isYoloTrainingWorkspace = item.handlerId === 'yolo.train'
  const isCalibrationWorkspace = item.handlerId === 'calib.select_dataset'
  const isRemoteStarWorkspace = item.handlerId === 'remote.star_inference'
  const isRemoteTrtWorkspace = item.handlerId === 'remote.build_tensorrt'
  const isLocalStarPackageWorkspace = item.handlerId === 'model.package_star'
  const isJetsonOnnxExportWorkspace = item.handlerId === 'model.export_jetson_onnx'
  const isVideoFramesWorkspace = item.handlerId === 'video.extract_frames'
  const isVideoClipWorkspace = item.handlerId === 'video.clip'
  const clipReplaceSource = isVideoClipWorkspace && props.values.parameters.clip_save_mode === CLIP_REPLACE_SOURCE
  const clipExactTime = isVideoClipWorkspace && (clipReplaceSource || props.values.parameters.clip_exact_time !== false)
  const isLocalInferenceWorkspace = item.handlerId === 'local.pt_inference'
  const isImageDedupWorkspace = item.handlerId === 'image.smart_dedup'
  const isAutoAnnotationWorkspace = item.handlerId === 'dataset.auto_annotate'
  const starAnnotation = props.values.parameters.annotation_mode === '远程 STAR 标注'
  const cycleMode = props.values.parameters.dedup_mode === '完整工序循环去重'
  const dedupMode = String(props.values.parameters.dedup_mode || '差异优先去重（自动）')
  const automaticDedup = dedupMode === '差异优先去重（自动）' || dedupMode === '仅精确重复'
  const processDedup = dedupMode === '工位工序去重（忽略背景）'
  const dedupRollback = props.values.parameters.operation === '回滚最近一次'
  const localImageMode = String(props.values.parameters.input_mode || '视频检测') === '图片文件夹检测'
  const currentTask = isYoloTrainingWorkspace || isVideoFramesWorkspace ? undefined : props.activeTasks.find((task) => task.functionId === item.id && !task.name.startsWith('YOLO 训练 ·'))
  const visibleParameters = item.parameters.filter((field) =>
    (!isVideoClipWorkspace || !['clip_save_mode', 'clip_exact_time'].includes(field.id)
      && (!clipReplaceSource || !['clip_output_name', 'split_train_test'].includes(field.id))
      && (!clipExactTime || field.id !== 'encoding_mode')) &&
    (!isAutoAnnotationWorkspace || field.id !== 'annotation_mode') && (!field.visibleWhen
      || (field.visibleWhen.equals !== undefined && props.values.parameters[field.visibleWhen.fieldId] === field.visibleWhen.equals)
      || (field.visibleWhen.notEquals !== undefined && props.values.parameters[field.visibleWhen.fieldId] !== field.visibleWhen.notEquals))
    && (!isRemoteStarWorkspace || !['labels', 'conf', 'additional_models', 'star_models'].includes(field.id))
    && (!isImageDedupWorkspace || (field.id !== 'dedup_mode'
      && !['minimum_keep_ratio', 'max_consecutive_skips', 'process_regions'].includes(field.id)
      && (field.id !== 'dedup_strength' || !automaticDedup && !processDedup && !dedupRollback)
      && (field.id !== 'process_strength' || processDedup && !dedupRollback)
      && (!field.id.startsWith('cycle_') || !dedupRollback)))
    && (!isLocalInferenceWorkspace || (field.id !== 'input_mode' && (field.id !== 'recursive_images' || localImageMode) && (!localImageMode || !['realtime', 'replay_grace_seconds'].includes(field.id)))))
  const visiblePathFields = item.pathFields.filter((field) => (!isRemoteStarWorkspace || field.id !== 'local_model_file')
    && (!clipReplaceSource || field.id !== 'output_folder')
    && (!isImageDedupWorkspace || !field.id.startsWith('cycle_') || (cycleMode && !dedupRollback))
    && (!isLocalInferenceWorkspace || (field.id !== 'image_folder' || localImageMode) && (field.id !== 'video_file' || !localImageMode)))
  const remoteYoloPaths = isYoloDatasetWorkspace && Boolean(
    String(props.values.parameters.remote_host || '').trim()
    || String(props.values.parameters.remote_username || '').trim()
    || String(props.values.parameters.remote_password || ''),
  )
  const showGenericSshHistory = isRemoteStarWorkspace || isYoloDatasetWorkspace || isAutoAnnotationWorkspace
  const sshPrefix = isYoloDatasetWorkspace || isAutoAnnotationWorkspace ? 'remote_' : ''
  return (
    <main className="function-workspace">
      <header className="workspace-header"><div><h1>{item.name}<HelpTip label={`${item.name}使用说明`}>{item.description || '在这里配置数据路径和运行参数'}</HelpTip><button aria-label="编辑功能" onClick={props.onEdit}><icons.Pencil size={19} /></button></h1></div><button className="outline-button" onClick={props.onEdit}><icons.Pencil size={17} />编辑功能</button></header>
      <section className="function-defaults-bar" aria-label="功能默认配置"><div><strong className="help-heading">默认配置<HelpTip label="默认配置说明">调整路径和参数后，可保存为下次打开时的默认值。{isYoloTrainingWorkspace ? '训练设备 device 每次手动填写，不保存为默认值。' : ''}</HelpTip></strong>{item.savedDefaults ? <span>{`已保存 · ${new Date(item.savedDefaults.savedAt).toLocaleString('zh-CN', { hour12: false })}`}</span> : null}{props.defaultsFeedback ? <small className="defaults-feedback" role="status">{props.defaultsFeedback}</small> : null}</div><div><button type="button" className="outline-button" disabled={props.savingDefaults || isYoloTrainingWorkspace && typeof props.values.parameters.training_parameters !== 'string'} onClick={props.onSaveDefaults}><icons.Check size={16} />{props.savingDefaults ? '保存中…' : '保存为默认值'}</button><button type="button" className="outline-button" disabled={props.savingDefaults} onClick={props.onLoadDefaults}><icons.Download size={16} />载入默认值</button></div></section>
      {currentTask ? <section className={`active-task-banner ${currentTask.status}`} role="status">
        <span><i /><strong>{currentTask.status === 'stopping' ? '正在终止任务' : '任务正在运行'}</strong><em>“{currentTask.name}”正在运行；其他功能仍可独立使用。</em></span>
        <button className="stop-run-button" onClick={() => props.onStop(currentTask.id)} disabled={currentTask.status === 'stopping'}><icons.X size={17} />{currentTask.status === 'stopping' ? '正在终止…' : '终止运行'}</button>
      </section> : null}
      <WorkflowTodoPanel items={props.workflowTodos} onApply={props.onApplyWorkflowTodo} onDelete={props.onDeleteWorkflowTodo} />
      {!isYoloTrainingWorkspace && !isCalibrationWorkspace && !isRemoteTrtWorkspace && !isLocalStarPackageWorkspace ? <>
      <CollapsibleSection key={`${item.id}:paths`} className="workspace-section" count={visiblePathFields.length} heading={
        <div className="section-title"><icons.Folder size={21} /><h2>数据路径</h2>{isVideoFramesWorkspace ? <HelpTip label="抽帧目录与调度说明">视频文件夹支持每行一个路径，也可连续点击“添加文件夹”追加。递归扫描后可在“图片保存方式”选择按源文件夹分别保存，或全部视频合并保存；图片均保留视频标识与抽帧信息。指定输出位置后，会在其中创建分组结果目录或一个汇总结果目录。“同时处理的视频数”设为 0 自动调度，数量上限仍按每个视频计算。</HelpTip> : null}</div>}>
        {isAutoAnnotationWorkspace ? <div className="local-inference-source"><label><span>自动标注模式</span><select aria-label="自动标注模式" value={starAnnotation ? '远程 STAR 标注' : '本地 PT 标注'} disabled={props.running || Boolean(currentTask)} onChange={event => props.onParameterChange('annotation_mode', event.target.value)}><option>本地 PT 标注</option><option>远程 STAR 标注</option></select></label></div> : null}
        {isImageDedupWorkspace ? <div className="local-inference-source"><label><span>去重模式</span><select aria-label="图片去重模式" value={dedupMode} disabled={props.running || Boolean(currentTask)} onChange={(event) => props.onParameterChange('dedup_mode', event.target.value)}>
          <option>差异优先去重（自动）</option><option>工位工序去重（忽略背景）</option><option>仅精确重复</option><option>连续相似帧去重</option><option>完整工序循环去重</option>
        </select></label><HelpTip label="去重模式说明">{processDedup
          ? '适合目标工位重复作业、旁边工位或背景持续变化的图片。先读取真实目录并为每个工位框选作业区域，只在区域内比较结构、局部动作和颜色；允许有限姿态变化及标注框抖动，不要求全图像素一致。无固定轮次时长，不自动识别完整工序边界；按相近动作画面保留代表图，并非证明工序语义相同。微小缺陷可能被忽略，先分析抽查对照样本。区域外标注变化会被忽略，区域内类别/数量或明显框位置变化保留。'
          : cycleMode ? '指定第一轮完整工序的起止图片（包含两端），参考轮及之前的图片全部保留。允许后续轮次快慢不同，逐张核验原图；局部变化或判断不确定时整轮保留。各工位需独立参考轮，请关闭“包含子文件夹”逐目录处理。'
          : automaticDedup ? dedupMode === '仅精确重复' ? '仅移除完整像素一致且标注一致的重复图片，每组保留第一张，不合并噪声或压缩差异。'
          : '不限相邻图片，检索候选后复核整张原图及局部差异，只容忍很小的噪声；背景或框位置变化也可能导致保留。无需阈值、不设数量配额，但不保证能识别所有微小缺陷。背景干扰多时建议改用“工位工序去重”。'
          : '连续帧相似度只提名候选，再复核原图局部差异与标注，不设数量配额。'}开启“包含子文件夹”逐目录独立处理，不跨工位比较。分析不移动图片；执行后图片及同名标注移入备份，可回滚，不为凑数删图。</HelpTip></div> : null}
        {isLocalInferenceWorkspace ? <div className="local-inference-source"><label><span>输入类型</span><select aria-label="本地推理输入类型" value={localImageMode ? '图片文件夹检测' : '视频检测'} disabled={Boolean(currentTask)} onChange={(event) => props.onParameterChange('input_mode', event.target.value)}><option>视频检测</option><option>图片文件夹检测</option></select></label><HelpTip label="本地推理输入说明">{localImageMode ? '选择图片文件夹与输出目录，逐张检测并保存带框图片。支持 JPG、PNG、BMP、WEBP、TIF 等；可包含子文件夹。结果保留目录结构，重名加后缀，不修改原图。未检出目标的图片也会保存。' : '使用本地视频进行检测与实时预览；输出文件夹可留空，仅预览而不保存视频。'}</HelpTip></div> : null}
        {isVideoClipWorkspace ? <VideoClipOptions values={props.values} disabled={props.running || Boolean(currentTask)} onChange={props.onParameterChange} /> : null}
        <div className="path-fields">
          {visiblePathFields.map((field) => {
            const multiFolders = isVideoFramesWorkspace && field.id === 'video_folder'
            const multiModels = isJetsonOnnxExportWorkspace && field.id === 'weights_file'
            const inferenceVideo = isRemoteStarWorkspace && field.id === 'local_video_file' || isLocalInferenceWorkspace && field.id === 'video_file'
            return <Fragment key={field.id}><label className="path-field"><span>{isAutoAnnotationWorkspace && field.id === 'weights_file' ? starAnnotation ? '本机 STAR 检测模型' : '训练后的 PT 检测模型' : field.label}</span><div>{multiFolders || multiModels
              ? <textarea rows={multiFolders ? 4 : 6} value={props.values.paths[field.id] || ''} onChange={(event) => props.onPathChange(field.id, event.target.value)} placeholder={multiFolders ? '/path/to/视频文件夹1\n/path/to/视频文件夹2' : '/path/to/model_a.pt\n/path/to/model_b.pt'} />
              : <input value={props.values.paths[field.id] || ''} onChange={(event) => props.onPathChange(field.id, event.target.value)} placeholder={remoteYoloPaths ? '请输入远程服务器上的绝对路径' : `请选择或粘贴${field.mode === 'file' ? '文件' : '文件夹'}路径`} />}
              <button onClick={() => props.onBrowse(field.id, multiFolders ? 'append-directory' : multiModels ? 'multiple' : field.mode)} disabled={remoteYoloPaths} title={remoteYoloPaths ? '远程模式请手动填写服务器路径' : multiFolders ? '选择一个文件夹并追加到列表，可重复添加' : undefined}>{remoteYoloPaths ? '远程路径' : multiFolders ? '添加文件夹' : multiModels ? '选择多个 PT' : '浏览'}</button></div></label>
              {inferenceVideo ? <InferenceVideoHistory path={props.values.paths[field.id] || ''} scene={String(props.values.parameters.test_video_scene || '')} station={String(props.values.parameters.test_video_station || '')} onPath={value => props.onPathChange(field.id, value)} onScene={value => props.onParameterChange('test_video_scene', value)} onStation={value => props.onParameterChange('test_video_station', value)} /> : null}</Fragment>
          })}
          {!visiblePathFields.length ? <p className="section-empty">尚未定义数据路径，接入功能时再添加。</p> : null}
        </div>
        {!isRemoteStarWorkspace ? <button className="text-action" onClick={props.onAddPath}><icons.Plus size={17} />添加路径</button> : null}
        {isImageDedupWorkspace && processDedup && !dedupRollback ? <DedupRegionSelector key={`${props.values.paths.images_folder}|${Boolean(props.values.parameters.recursive)}`} folder={props.values.paths.images_folder || ''} recursive={Boolean(props.values.parameters.recursive)} value={String(props.values.parameters.process_regions || '')} disabled={props.running || Boolean(currentTask)} onChange={value => props.onParameterChange('process_regions', value)} /> : null}
        {isJetsonOnnxExportWorkspace ? <JetsonModelLabels values={props.values} running={props.running || Boolean(currentTask)} /> : null}
        {isJetsonOnnxExportWorkspace ? <div className="help-heading">ONNX 导出与部署精度<HelpTip label="ONNX 导出精度说明">当前导出统一生成未量化 FP32 ONNX。“后续部署精度”决定下一步：INT8 沿用 Docker 量化待办；FP16/FP32 跳过量化，直接用导出的 ONNX 在盒子构建对应引擎。此选项不把现有量化模型升精度，也不自动上传到远端；请手动复制文件并填写盒子上的路径。</HelpTip></div> : null}
      </CollapsibleSection>
      <CollapsibleSection key={`${item.id}:parameters`} className="workspace-section parameter-section" count={visibleParameters.length} heading={
        <div className="section-title"><icons.SlidersHorizontal size={21} /><h2>运行参数</h2>
        {isImageDedupWorkspace ? <HelpTip label="批量去重与回滚说明">开启“包含子文件夹”后，自动扫描任意深度子目录；上层目录自身有图片也会单独处理。每个实际图片目录独立去重，不跨目录比较；空目录、去重备份和符号链接不参与。各目录串行处理并输出独立报告与汇总，失败目录会提示并跳过。完整工序循环模式需关闭此选项，单独指定各工位参考轮。保持同一上层路径和此选项，选择“回滚最近一次”，只恢复最近一次有移出文件的批量任务，不回滚无关的单目录任务。中途停止会保留此前已完成的目录结果，也可回滚。此选项可保存为默认值。</HelpTip> : null}
        {isImageDedupWorkspace && cycleMode && !dedupRollback ? <HelpTip label="循环去重参数说明">最短/最长倍数限制每轮图片张数，不是固定时长。例如参考轮 200 张，默认搜索 80～500 张的循环。力度越强越容易移除，请先从“保守”或“较保守”分析；过于静止、无法区分轮次的参考片段会提示重新选择。</HelpTip> : null}
        {item.handlerId === 'video.clip' && props.values.parameters.clip_mode === '删除多个时间段并拼接剩余' ? <HelpTip label="删除片段与拼接说明">每行填写一个不要的时间段，支持秒数、MM:SS、HH:MM:SS，也可用分号分隔。所有时间均对应原视频；结束时间留空表示删除到结尾。重叠区间自动合并，剩余片段按原顺序拼接。{clipReplaceSource ? '成功验证并备份后替换原视频。' : '另存新视频，原视频不变。'}{clipExactTime || props.values.parameters.encoding_mode === '精确裁剪（重新编码）' ? '精确模式会重新编码视频，并保留首条音轨。' : '当前原画质模式受关键帧限制，删除边界可能有误差；要求严格删除时请开启按帧精确裁剪。'}</HelpTip> : null}</div>}>
        {visibleParameters.length ? <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{visibleParameters.map((field) => <label className="parameter-row" key={field.id}><span>{field.label}{isVideoFramesWorkspace && field.id === 'output_layout' ? <HelpTip label="抽帧图片保存方式说明">按源文件夹分别保存：同一实际父目录中的视频共用一个结果目录，不同父目录分开。全部视频合并保存：所有输入目录、子目录和单个视频的图片，放入同一个新结果目录。输出位置留空时，分组模式保存在各自源目录；合并模式保存在排序后首个视频的父目录。每次运行新建结果目录，不覆盖旧结果；图片带视频编号和抽帧信息，数量上限仍按每个视频计算。可保存为默认值。</HelpTip> : null}</span><ParameterValue field={field} value={props.values.parameters[field.id] ?? field.default} onChange={(value) => props.onParameterChange(field.id, value)} sshHistory={showGenericSshHistory ? { onSelect: (profile) => {
          props.onParameterChange(`${sshPrefix}port`, profile.port)
          props.onParameterChange(`${sshPrefix}username`, profile.username)
          props.onParameterChange('remember_password', true)
          props.onParameterChange(`${sshPrefix}password`, profile.passwordRemembered ? REMEMBERED_PASSWORD_VALUE : '')
        } } : undefined} /></label>)}</div> : <p className="section-empty parameter-empty">当前没有运行参数。具体功能确定后，可通过“添加参数”配置。</p>}
        {!isRemoteStarWorkspace ? <button className="text-action" onClick={props.onAddParameter}><icons.Plus size={17} />添加参数</button> : null}
      </CollapsibleSection>
      </> : null}
        {item.handlerId === 'remote.star_inference' ? <RemoteInferencePanel itemId={item.id} values={props.values} onParameterChange={props.onParameterChange} platformTask={currentTask || null} onCompleted={() => props.onWorkflowComplete('remote.star_inference')} />
        : isAutoAnnotationWorkspace ? <AutoAnnotationPanel key={`annotation:${item.id}`} values={props.values} output={props.output} running={props.running || Boolean(currentTask)} onParameterChange={props.onParameterChange} onRun={props.onRun} />
        : item.handlerId === 'local.pt_inference' ? <LocalPTInferencePanel itemId={item.id} values={props.values} platformTask={currentTask || null} onParameterChange={props.onParameterChange} />
        : isRemoteTrtWorkspace ? <RemoteTensorRTBuildPanel itemId={item.id} values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} />
        : isLocalStarPackageWorkspace ? <LocalStarPackagePanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onBrowse={props.onBrowse} onPathChange={props.onPathChange} onParameterChange={props.onParameterChange} onRun={props.onRun} />
        : isCalibrationWorkspace ? <RemoteCalibrationPanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} />
        : isYoloDatasetWorkspace ? <RemoteYoloDatasetPanel values={props.values} output={props.output} running={props.running || Boolean(currentTask)} platformTask={currentTask || null} onParameterChange={props.onParameterChange} onRun={props.onRun} />
        : isYoloTrainingWorkspace ? <YoloTrainingPanel values={props.values} defaultsVersion={props.defaultsVersion} onParameterChange={props.onParameterChange} recommendation={props.yoloTrainingRecommendation} onCompleted={() => props.onWorkflowComplete('yolo.train')} /> : <section className="run-section">
        <div className="run-actions"><button className="run-button" onClick={() => props.onRun()} disabled={!isVideoFramesWorkspace && (props.running || Boolean(currentTask))}><icons.Play size={18} fill="currentColor" />{isVideoFramesWorkspace ? '启动抽帧任务' : currentTask ? '正在运行' : props.running ? '正在启动' : isImageDedupWorkspace ? dedupRollback ? '回滚最近一次' : props.values.parameters.operation === '仅分析（不移动文件）' ? '开始分析' : '执行去重' : '一键运行'}</button>
          {isImageDedupWorkspace && !dedupRollback && props.values.parameters.operation !== '仅分析（不移动文件）' ? <button className="outline-button" disabled={props.running || Boolean(currentTask)} onClick={() => props.onRun({ operation: '仅分析（不移动文件）' })}>先分析，不移动图片</button> : null}
          <span className={item.handlerReady ? 'ready' : ''}><icons.Info size={17} />{item.handlerReady ? '处理逻辑已接入' : '处理逻辑尚未接入'}</span></div>
        {isVideoFramesWorkspace ? <VideoFrameTasksPanel tasks={props.activeTasks.filter((task) => task.functionId === item.id)} onStop={props.onStop} /> : null}
        <details className="run-output" open><summary><icons.ChevronRight size={18} />运行输出</summary><pre>{props.output || '运行后将在这里显示处理进度和结果'}</pre></details>
      </section>}
      <FunctionRunHistory key={item.id} functionId={item.id} />
    </main>
  )
}
