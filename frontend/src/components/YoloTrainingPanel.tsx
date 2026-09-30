import { useEffect, useRef, useState } from 'react'
import {
  choosePaths,
  deleteYoloTrainingSession,
  forgetYoloRemotePassword,
  getYoloRemoteCredentialStatus,
  getYoloTrainingProfiles,
  getYoloTrainingSessions,
  getYoloTrainingStatus,
  reconnectYoloTraining,
  startYoloTraining,
  stopYoloTraining,
  testYoloRemoteConnection,
} from '../api'
import { icons } from '../icons'
import { REMEMBERED_PASSWORD_VALUE } from '../types'
import type { WorkingValues, YoloConnectionTestResult, YoloTrainingRecommendation, YoloTrainingSession, YoloTrainingSummary, YoloTrainingValues } from '../types'

type Props = {
  values: WorkingValues
  onParameterChange: (fieldId: string, value: string | number | boolean) => void
  recommendation: YoloTrainingRecommendation | null
  onCompleted?: () => void
}

type TrainingField = {
  id: string
  label: string
  type?: 'text' | 'number' | 'boolean' | 'select'
  options?: string[]
  step?: string
  browse?: 'file' | 'directory'
  help: string
  range?: string
}

type ConnectionCheck = {
  signature: string
  state: 'checking' | 'success' | 'failed'
  result?: YoloConnectionTestResult
  error?: string
}

const fieldGroups: Array<{ title: string; description: string; defaultOpen?: boolean; fields: TrainingField[] }> = [
  {
    title: '常用训练配置',
    description: '最常修改的数据、模型、轮次、批量大小、尺寸、GPU 和输出位置。',
    defaultOpen: true,
    fields: [
      { id: 'data', label: 'data.yaml 路径', browse: 'file', help: '数据集配置文件，定义 train/val/test 路径、类别数量和类别名称。' },
      { id: 'model', label: '预训练模型 model', help: '预训练 .pt 权重或模型 YAML。使用预训练权重通常能更快收敛。' },
      { id: 'epochs', label: '训练轮次 epochs', type: 'number', help: '完整遍历训练集的最大次数；与 patience 共同决定实际训练时长。', range: '建议 50–500，当前工业任务默认 200' },
      { id: 'patience', label: '早停等待 patience', type: 'number', help: '验证指标连续多少轮没有改善后停止；0 表示关闭早停。', range: '≥ 0，常用 30–100' },
      { id: 'imgsz', label: '输入尺寸 imgsz', type: 'number', help: '训练输入图像尺寸。更大有利于小目标，但显存和训练时间明显增加。', range: '通常为 32 的倍数，如 640、960、1280' },
      { id: 'batch', label: '批量大小 batch', type: 'number', step: 'any', help: '每批图像数；-1 自动估算，0–1 可表示显存占用比例，正整数为固定批量。', range: '-1、0–1 或正整数' },
      { id: 'device', label: '训练设备 device', help: 'GPU 编号或列表，例如 0、0,1；也可填写 cpu。多任务训练时应合理分配 GPU。' },
      { id: 'workers', label: '数据线程 workers', type: 'number', help: '每个训练进程的数据加载线程数。过高可能占满 CPU 或共享内存。', range: '≥ 0，常用 4–16' },
      { id: 'project', label: '训练输出目录 project', browse: 'directory', help: '训练结果的父目录；最终结果保存在 project/name 下。' },
      { id: 'run_name', label: '本次训练名称 name', help: '本次实验子目录名。留空时平台自动生成 yolo26_日期_时间。' },
    ],
  },
  {
    title: '数据增强',
    description: '先选择工业场景策略，再按需要微调；切换策略只修改本组增强参数。',
    defaultOpen: true,
    fields: [
      { id: 'hsv_h', label: '色相扰动 hsv_h', type: 'number', step: '0.001', help: '按色轮比例随机改变色相，增强不同灯光颜色下的泛化能力。', range: '官方范围 0–1，默认 0.015' },
      { id: 'hsv_s', label: '饱和度扰动 hsv_s', type: 'number', step: '0.01', help: '随机改变颜色饱和度，模拟颜色浓淡和成像差异。', range: '官方范围 0–1，默认 0.7' },
      { id: 'hsv_v', label: '亮度扰动 hsv_v', type: 'number', step: '0.01', help: '随机改变明暗，是处理光线变化最直接的增强参数。', range: '官方范围 0–1，默认 0.4' },
      { id: 'mosaic', label: '拼图概率 mosaic', type: 'number', step: '0.01', help: '把 4 张训练图组合为一张，可增加场景和尺度多样性，对小目标常有帮助。', range: '官方范围 0–1' },
      { id: 'mixup', label: '图像混合概率 mixup', type: 'number', step: '0.01', help: '混合两张图及其标签，提高泛化但会引入视觉与标签噪声。', range: '官方范围 0–1' },
      { id: 'cutmix', label: '区域混合概率 cutmix', type: 'number', step: '0.01', help: '将另一张图的局部区域贴入当前图，适合增强遮挡鲁棒性。', range: '官方范围 0–1；检测/分割/姿态/OBB 可用' },
      { id: 'degrees', label: '旋转角度 degrees', type: 'number', step: '0.1', help: '在正负该角度内随机旋转。固定方向工位不宜设置过大。', range: '官方范围 0–180°' },
      { id: 'translate', label: '平移比例 translate', type: 'number', step: '0.01', help: '按图像尺寸比例随机水平和垂直平移，有助于学习部分可见目标。', range: '官方范围 0–1' },
      { id: 'scale', label: '缩放幅度 scale', type: 'number', step: '0.01', help: '随机缩放图像，模拟目标距离变化；数值越大尺度变化越强。', range: '官方常用范围 0–1' },
      { id: 'shear', label: '剪切角度 shear', type: 'number', step: '0.1', help: '随机剪切图像。工业固定相机通常只需很小的值。', range: '官方范围 0–180°' },
      { id: 'perspective', label: '透视变化 perspective', type: 'number', step: '0.0001', help: '增加透视形变；固定机位应谨慎使用，过大会扭曲工件。', range: '官方范围 0–0.001' },
      { id: 'fliplr', label: '水平翻转概率 fliplr', type: 'number', step: '0.01', help: '水平翻转概率。左右方向具有业务含义时应设为 0。', range: '官方范围 0–1' },
      { id: 'flipud', label: '垂直翻转概率 flipud', type: 'number', step: '0.01', help: '垂直翻转概率。多数固定工业相机建议保持 0。', range: '官方范围 0–1' },
      { id: 'bgr', label: 'RGB/BGR 通道交换概率 bgr', type: 'number', step: '0.01', help: '按概率交换 RGB 与 BGR 通道；仅在颜色不具有明确业务含义时谨慎尝试。', range: '官方范围 0–1' },
      { id: 'copy_paste', label: '复制粘贴概率 copy_paste', type: 'number', step: '0.01', help: '复制目标实例进行增强，官方主要支持分割和旋转框任务。', range: '官方范围 0–1；detect 通常保持 0' },
    ],
  },
  {
    title: '训练稳定性与性能',
    description: '学习率、优化器、精度、复现、缓存和训练后期稳定策略。',
    fields: [
      { id: 'optimizer', label: '优化器 optimizer', type: 'select', options: ['SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp', 'auto'], help: '选择权重更新算法；auto 让 Ultralytics 根据训练配置选择。' },
      { id: 'lr0', label: '初始学习率 lr0', type: 'number', step: '0.0001', help: '训练开始时的学习率。过大会震荡，过小会收敛缓慢。', range: '通常 SGD 约 0.01，Adam 类约 0.001' },
      { id: 'lrf', label: '最终学习率比例 lrf', type: 'number', step: '0.001', help: '最终学习率相对于 lr0 的比例。', range: '0–1' },
      { id: 'momentum', label: '动量 momentum', type: 'number', step: '0.001', help: 'SGD 动量或 Adam beta1，帮助平滑梯度并加速收敛。', range: '0–1，官方默认 0.937' },
      { id: 'weight_decay', label: '权重衰减 weight_decay', type: 'number', step: '0.0001', help: 'L2 正则强度，用于抑制过拟合。', range: '≥ 0，官方默认 0.0005' },
      { id: 'cos_lr', label: '余弦学习率 cos_lr', type: 'boolean', help: '使用余弦曲线逐步降低学习率，通常适合较长训练。' },
      { id: 'warmup_epochs', label: '预热轮次 warmup_epochs', type: 'number', step: '0.1', help: '训练开始时逐步提高学习率，降低初期不稳定。', range: '≥ 0，官方默认 3' },
      { id: 'warmup_momentum', label: '预热初始动量 warmup_momentum', type: 'number', step: '0.01', help: '预热阶段起始动量，随后过渡到 momentum。', range: '0–1，官方默认 0.8' },
      { id: 'warmup_bias_lr', label: '偏置预热学习率 warmup_bias_lr', type: 'number', step: '0.01', help: '预热阶段偏置参数的初始学习率。', range: '0–1，官方默认 0.1' },
      { id: 'amp', label: '混合精度 amp', type: 'boolean', help: 'CUDA 上使用 FP16 混合精度，通常能降低显存并加速训练。发生数值异常时可关闭。' },
      { id: 'close_mosaic', label: '最后关闭 Mosaic 轮次 close_mosaic', type: 'number', help: '最后 N 轮关闭 Mosaic，使训练末期回到自然图像分布并稳定收敛。', range: '≥ 0，官方默认 10；0 表示不关闭' },
      { id: 'cache', label: '图像缓存 cache', type: 'select', options: ['False', 'ram', 'disk'], help: '缓存训练图像以减少读取开销。ram 最快但占内存，disk 较省内存但占磁盘。' },
      { id: 'rect', label: '矩形训练 rect', type: 'boolean', help: '按相近宽高比分批以减少填充；可能影响随机打乱和部分增强方式。' },
      { id: 'multi_scale', label: '多尺度训练 multi_scale', type: 'number', step: '0.01', help: '训练时随机改变 imgsz 的幅度，提高对不同输入尺度的适应性。', range: '0–1，0 表示关闭' },
      { id: 'seed', label: '随机种子 seed', type: 'number', help: '控制数据打乱和随机增强，配合 deterministic 提高复现性。', range: '≥ 0' },
      { id: 'deterministic', label: '确定性训练 deterministic', type: 'boolean', help: '尽量使用确定性算法，便于复现实验；可能略微降低速度。' },
      { id: 'save_period', label: '定期保存 save_period', type: 'number', help: '每隔 N 轮额外保存检查点；-1 表示只采用默认保存策略。', range: '-1 或正整数' },
      { id: 'plots', label: '生成训练图表 plots', type: 'boolean', help: '生成损失曲线、PR 曲线和样本预测等可视化结果。' },
    ],
  },
  {
    title: '高级配置与损失权重',
    description: '一般保持默认，只有明确实验依据时再调整。',
    fields: [
      { id: 'task', label: '训练任务类型 task', type: 'select', options: ['detect', 'segment', 'classify', 'pose', 'obb'], help: 'Ultralytics 任务类型；当前数据集划分主要面向 detect。' },
      { id: 'yolo_executable', label: 'YOLO 命令或可执行文件路径', help: '本地/远程 yolo 可执行文件。填写 yolo 时，远程会自动查找匹配模型系列的 Conda 环境。' },
      { id: 'nbs', label: '标称批量 nbs', type: 'number', help: '用于按实际 batch 缩放部分超参数的名义批量大小。', range: '> 0，官方常用默认 64' },
      { id: 'cls', label: '分类损失权重 cls', type: 'number', step: '0.01', help: '类别预测损失的权重。类别混淆严重时可实验性调整。', range: '≥ 0' },
      { id: 'box', label: '边框损失权重 box', type: 'number', step: '0.01', help: '边界框回归损失权重，影响定位精度。', range: '≥ 0' },
      { id: 'dfl', label: 'DFL 损失权重 dfl', type: 'number', step: '0.01', help: '分布焦点损失权重，参与更精细的边框定位。', range: '≥ 0' },
    ],
  },
]

const augmentationFields = ['hsv_h', 'hsv_s', 'hsv_v', 'mosaic', 'mixup', 'cutmix', 'degrees', 'translate', 'scale', 'shear', 'perspective', 'fliplr', 'flipud', 'bgr', 'copy_paste'] as const
type AugmentationField = typeof augmentationFields[number]
type AugmentationPreset = { name: string; description: string; values: Record<AugmentationField, number> }

const augmentationPresets: AugmentationPreset[] = [
  { name: '工业保守（推荐）', description: '固定相机、方向明确的常规工位，避免制造不符合现场规律的样本。', values: { hsv_h: 0.01, hsv_s: 0.35, hsv_v: 0.3, mosaic: 0.2, mixup: 0, cutmix: 0, degrees: 3, translate: 0.06, scale: 0.25, shear: 0, perspective: 0, fliplr: 0, flipud: 0, bgr: 0, copy_paste: 0 } },
  { name: '光线变化', description: '跨白天、夜间、曝光和灯光颜色变化，增强 HSV 并保持几何变化适中。', values: { hsv_h: 0.025, hsv_s: 0.65, hsv_v: 0.55, mosaic: 0.2, mixup: 0, cutmix: 0, degrees: 3, translate: 0.06, scale: 0.25, shear: 0, perspective: 0, fliplr: 0, flipud: 0, bgr: 0, copy_paste: 0 } },
  { name: '小目标加强', description: '目标在画面中占比较小，使用更强的 Mosaic、缩放和平移增加尺度与位置变化。', values: { hsv_h: 0.015, hsv_s: 0.5, hsv_v: 0.4, mosaic: 1, mixup: 0.05, cutmix: 0, degrees: 5, translate: 0.12, scale: 0.5, shear: 0, perspective: 0, fliplr: 0, flipud: 0, bgr: 0, copy_paste: 0 } },
  { name: '遮挡与复杂背景', description: '人员、工具或工件容易互相遮挡，增加 Mosaic、MixUp 和 CutMix。', values: { hsv_h: 0.015, hsv_s: 0.5, hsv_v: 0.4, mosaic: 0.7, mixup: 0.1, cutmix: 0.15, degrees: 5, translate: 0.1, scale: 0.4, shear: 1, perspective: 0.0002, fliplr: 0, flipud: 0, bgr: 0, copy_paste: 0 } },
  { name: 'Ultralytics 官方默认', description: '恢复官方通用增强默认值；水平翻转可能不适合方向固定的工业场景。', values: { hsv_h: 0.015, hsv_s: 0.7, hsv_v: 0.4, mosaic: 1, mixup: 0, cutmix: 0, degrees: 0, translate: 0.1, scale: 0.5, shear: 0, perspective: 0, fliplr: 0.5, flipud: 0, bgr: 0, copy_paste: 0 } },
  { name: '关闭可配置增强', description: '将界面可控制的增强参数全部设为 0；Ultralytics 安装 Albumentations 时仍可能有内置轻量增强。', values: Object.fromEntries(augmentationFields.map((field) => [field, 0])) as Record<AugmentationField, number> },
]

const terminalStates = new Set<YoloTrainingSession['status']>(['completed', 'stopped', 'failed'])

function automaticTrainingRunName(date = new Date()): string {
  const twoDigits = (value: number) => String(value).padStart(2, '0')
  return `yolo26_${twoDigits(date.getMonth() + 1)}${twoDigits(date.getDate())}_${twoDigits(date.getHours())}${twoDigits(date.getMinutes())}`
}

export function YoloTrainingPanel({ values, onParameterChange, recommendation, onCompleted }: Props) {
  const [trainingValues, setTrainingValues] = useState<YoloTrainingValues>({})
  const [profilesLoading, setProfilesLoading] = useState(true)
  const [augmentationPreset, setAugmentationPreset] = useState('自定义（当前参数）')
  const [notice, setNotice] = useState('')
  const [error, setError] = useState('')
  const [connectionCheck, setConnectionCheck] = useState<ConnectionCheck | null>(null)
  const [credentialNotice, setCredentialNotice] = useState('')
  const [sessionId, setSessionId] = useState(() => sessionStorage.getItem('processing-view:yolo-training-session') || '')
  const [session, setSession] = useState<YoloTrainingSession | null>(null)
  const [sessions, setSessions] = useState<YoloTrainingSummary[]>([])
  const [submitting, setSubmitting] = useState(false)
  const checkedCredentialIdentity = useRef('')
  const appliedRecommendation = useRef('')
  const completedSession = useRef('')

  const host = String(values.parameters.remote_host || '').trim()
  const port = Number(values.parameters.remote_port || 22)
  const username = String(values.parameters.remote_username || '').trim()
  const password = String(values.parameters.remote_password || '')
  const rememberPassword = values.parameters.remember_password !== false
  const remoteRequested = Boolean(host || username || password)
  const identitySignature = JSON.stringify([host, port, username])
  const passwordSignature = password === REMEMBERED_PASSWORD_VALUE ? '' : password
  const connectionSignature = JSON.stringify([host, port, username, passwordSignature, rememberPassword])
  const currentConnectionCheck = connectionCheck?.signature === connectionSignature ? connectionCheck : null
  const connectionReady = currentConnectionCheck?.state === 'success'
  const testingConnection = currentConnectionCheck?.state === 'checking'
  const activeSession = Boolean(session && !terminalStates.has(session.status))

  useEffect(() => {
    let cancelled = false
    const refresh = () => getYoloTrainingSessions().then((items) => {
      if (cancelled) return
      setSessions(items)
      if (!sessionStorage.getItem('processing-view:yolo-training-session') && items[0]) {
        sessionStorage.setItem('processing-view:yolo-training-session', items[0].id)
        setSessionId(items[0].id)
      }
    }).catch(() => { /* The selected-session request reports failures. */ })
    void refresh()
    const timer = window.setInterval(refresh, 2000)
    return () => { cancelled = true; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    getYoloTrainingProfiles().then((payload) => {
      setTrainingValues(payload.profiles[0] ? { ...payload.profiles[0].values } : payload.defaults)
    }).catch((loadError) => {
      setError(loadError instanceof Error ? loadError.message : '训练参数加载失败')
    }).finally(() => setProfilesLoading(false))
  }, [])

  useEffect(() => {
    if (profilesLoading || !recommendation || appliedRecommendation.current === recommendation.updatedAt) return
    appliedRecommendation.current = recommendation.updatedAt
    setTrainingValues((current) => ({
      ...current,
      data: recommendation.data,
      project: recommendation.project,
      run_name: '',
    }))
    setNotice('已自动填入刚刚划分的数据集路径和训练输出目录；训练名称将在启动时自动生成。')
    setError('')
  }, [profilesLoading, recommendation])

  useEffect(() => {
    if (!rememberPassword) {
      checkedCredentialIdentity.current = ''
      if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('remote_password', '')
      return
    }
    if (!host || !username) return
    if (password === REMEMBERED_PASSWORD_VALUE) {
      if (checkedCredentialIdentity.current === identitySignature) return
      checkedCredentialIdentity.current = ''
      onParameterChange('remote_password', '')
      return
    }
    if (password || checkedCredentialIdentity.current === identitySignature) return
    let cancelled = false
    checkedCredentialIdentity.current = identitySignature
    getYoloRemoteCredentialStatus(values.parameters).then((result) => {
      if (cancelled || !result.remembered) return
      onParameterChange('remote_password', REMEMBERED_PASSWORD_VALUE)
      setCredentialNotice('已自动填充该服务器上次保存的 SSH 密码。')
    }).catch(() => {
      if (!cancelled) checkedCredentialIdentity.current = ''
    })
    return () => { cancelled = true }
  }, [host, identitySignature, password, port, rememberPassword, username])

  useEffect(() => {
    if (!sessionId) return
    let cancelled = false
    let timer: number | undefined
    const refresh = async () => {
      try {
        const next = await getYoloTrainingStatus(sessionId)
        if (cancelled) return
        setSession(next)
        if (!terminalStates.has(next.status)) timer = window.setTimeout(refresh, 800)
      } catch (refreshError) {
        if (cancelled) return
        sessionStorage.removeItem('processing-view:yolo-training-session')
        setSessionId('')
        setSession(null)
        setError(refreshError instanceof Error ? refreshError.message : '无法读取训练状态')
      }
    }
    refresh()
    return () => {
      cancelled = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [sessionId])

  useEffect(() => {
    if (session?.status !== 'completed' || completedSession.current === session.id) return
    completedSession.current = session.id
    onCompleted?.()
  }, [onCompleted, session?.id, session?.status])

  const changeTrainingValue = (fieldId: string, value: string | number | boolean) => {
    setTrainingValues((current) => ({ ...current, [fieldId]: value }))
    if (augmentationFields.includes(fieldId as AugmentationField)) setAugmentationPreset('自定义（当前参数）')
    setNotice('')
  }

  const applyAugmentationPreset = (name: string) => {
    setAugmentationPreset(name)
    const preset = augmentationPresets.find((item) => item.name === name)
    if (!preset) return
    setTrainingValues((current) => ({ ...current, ...preset.values }))
    setNotice(`已应用数据增强策略“${preset.name}”，仍可继续微调下方参数。`)
    setError('')
  }

  const browseTrainingPath = async (field: TrainingField) => {
    if (!field.browse || remoteRequested) return
    try {
      const paths = await choosePaths(field.browse)
      if (paths[0]) changeTrainingValue(field.id, paths[0])
    } catch (browseError) {
      setError(browseError instanceof Error ? browseError.message : '无法打开选择窗口')
    }
  }

  const testConnection = async () => {
    const signature = connectionSignature
    setConnectionCheck({ signature, state: 'checking' })
    setCredentialNotice('')
    setError('')
    try {
      const result = await testYoloRemoteConnection(values.parameters)
      if (result.passwordRemembered) {
        checkedCredentialIdentity.current = identitySignature
        setConnectionCheck({ signature: JSON.stringify([host, port, username, '', true]), state: 'success', result })
        onParameterChange('remote_password', REMEMBERED_PASSWORD_VALUE)
      } else {
        setConnectionCheck({ signature, state: 'success', result })
      }
    } catch (connectionError) {
      setConnectionCheck({ signature, state: 'failed', error: connectionError instanceof Error ? connectionError.message : 'SSH 连接测试失败' })
    }
  }

  const forgetPassword = async () => {
    setSubmitting(true)
    setError('')
    try {
      const result = await forgetYoloRemotePassword(values.parameters)
      onParameterChange('remote_password', '')
      onParameterChange('remember_password', false)
      checkedCredentialIdentity.current = ''
      setConnectionCheck(null)
      setCredentialNotice(result.message)
    } catch (forgetError) {
      setError(forgetError instanceof Error ? forgetError.message : '清除保存密码失败')
    } finally {
      setSubmitting(false)
    }
  }

  const start = async () => {
    if (remoteRequested && (!connectionReady || !currentConnectionCheck?.result)) {
      setError('请先测试 SSH 连接，连接成功后再开始远程模型训练。')
      return
    }
    setSubmitting(true)
    setError('')
    try {
      const runName = String(trainingValues.run_name || '').trim() || automaticTrainingRunName()
      const preparedTrainingValues = { ...trainingValues, run_name: runName }
      setTrainingValues(preparedTrainingValues)
      const parameters: YoloTrainingValues = {
        ...preparedTrainingValues,
        remote_host: host,
        remote_port: port,
        remote_username: username,
        remote_password: password,
        remember_password: rememberPassword,
      }
      if (remoteRequested && currentConnectionCheck?.result) {
        parameters.remote_connection_token = currentConnectionCheck.result.connectionToken
      }
      const next = await startYoloTraining(parameters)
      sessionStorage.setItem('processing-view:yolo-training-session', next.id)
      setSession(next)
      setSessionId(next.id)
      const summary: YoloTrainingSummary = {
        id: next.id, remote: next.remote, model: next.model, device: next.device,
        output: next.output, host: next.host, port: next.port, username: next.username,
        status: next.status, message: next.message,
        startedAt: next.startedAt, finishedAt: next.finishedAt, error: next.error, result: next.result,
      }
      setSessions((current) => [summary, ...current.filter((item) => item.id !== next.id)])
      if (remoteRequested) onParameterChange('remote_password', rememberPassword ? REMEMBERED_PASSWORD_VALUE : '')
    } catch (startError) {
      setError(startError instanceof Error ? startError.message : '训练启动失败')
    } finally {
      setSubmitting(false)
    }
  }

  const stop = async () => {
    if (!sessionId) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await stopYoloTraining(sessionId))
    } catch (stopError) {
      setError(stopError instanceof Error ? stopError.message : '停止训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  const deleteSelectedSession = async () => {
    if (!session || !terminalStates.has(session.status)) return
    if (!window.confirm(`删除训练任务“${session.output.split('/').pop() || session.id}”的界面记录？服务器上的训练输出和文件不会删除。`)) return
    setSubmitting(true)
    setError('')
    try {
      await deleteYoloTrainingSession(session.id)
      const remaining = sessions.filter((item) => item.id !== session.id)
      setSessions(remaining)
      const nextId = remaining[0]?.id || ''
      if (nextId) sessionStorage.setItem('processing-view:yolo-training-session', nextId)
      else sessionStorage.removeItem('processing-view:yolo-training-session')
      setSessionId(nextId)
      setSession(null)
      setNotice('训练任务记录已删除，服务器上的训练文件未改动。')
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '删除训练任务记录失败')
    } finally {
      setSubmitting(false)
    }
  }

  const reconnect = async () => {
    if (!sessionId || !session) return
    setSubmitting(true)
    setError('')
    try {
      setSession(await reconnectYoloTraining(sessionId, {
        remote_host: session.host, remote_port: session.port, remote_username: session.username,
        remote_password: password, remember_password: rememberPassword,
      }))
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '重新连接训练失败')
    } finally {
      setSubmitting(false)
    }
  }

  const selectSession = (id: string) => {
    sessionStorage.setItem('processing-view:yolo-training-session', id)
    setSessionId(id)
    setSession(null)
    setError('')
  }

  const renderTrainingField = (field: TrainingField) => {
    const value = trainingValues[field.id] ?? ''
    if (field.type === 'boolean') {
      return <label className="switch-control"><input type="checkbox" checked={Boolean(value)} onChange={(event) => changeTrainingValue(field.id, event.target.checked)} /><span /><em>{value ? '开启' : '关闭'}</em></label>
    }
    if (field.type === 'select') {
      return <span className="workspace-select"><select value={String(value)} onChange={(event) => changeTrainingValue(field.id, event.target.value)}>{field.options?.map((option) => <option value={option} key={option}>{option}</option>)}</select><icons.ChevronDown size={16} /></span>
    }
    const input = <input type={field.type === 'number' ? 'number' : 'text'} step={field.step} value={String(value)} onChange={(event) => changeTrainingValue(field.id, field.type === 'number' && event.target.value !== '' ? Number(event.target.value) : event.target.value)} />
    if (!field.browse) return input
    return <span className="training-path-input">{input}<button onClick={() => browseTrainingPath(field)} disabled={remoteRequested}>{remoteRequested ? '远程路径' : '浏览'}</button></span>
  }

  const displayedPassword = password === REMEMBERED_PASSWORD_VALUE ? '已保存密码' : password
  const connectionError = currentConnectionCheck?.state === 'failed' ? currentConnectionCheck.error : ''
  const statusLabel = session ? {
    starting: '正在准备训练', running: '训练运行中', disconnected: '等待重新连接', stopping: '正在终止', completed: '训练完成', stopped: '已停止', failed: '训练失败',
  }[session.status] : remoteRequested ? (connectionReady ? 'SSH 连接成功' : '等待连接测试') : '本地训练'

  return (
    <div className="yolo-training-workspace">
      {notice ? <div className="credential-notice training-parameter-notice" role="status"><icons.Check size={18} /><span>{notice}</span></div> : null}
      {fieldGroups.map((group) => <details className="training-parameter-group" open={group.defaultOpen} key={group.title}>
        <summary><span><strong>{group.title}</strong><small>{group.description}</small></span><icons.ChevronRight size={18} /></summary>
        {group.title === '数据增强' ? <div className="augmentation-preset-panel">
          <label><span>数据增强策略</span><span className="workspace-select"><select value={augmentationPreset} onChange={(event) => applyAugmentationPreset(event.target.value)}><option value="自定义（当前参数）">自定义（当前参数）</option>{augmentationPresets.map((preset) => <option value={preset.name} key={preset.name}>{preset.name}</option>)}</select><icons.ChevronDown size={16} /></span></label>
          <p>{augmentationPresets.find((preset) => preset.name === augmentationPreset)?.description || '当前参数经过手动调整；可继续修改，或重新选择一个预设策略。'}</p>
        </div> : null}
        <div className="parameter-table"><div className="parameter-head"><span>参数名称</span><span>参数值</span></div>{group.fields.map((field) => <div className="parameter-row" key={field.id}><span className="parameter-label"><span>{field.label}</span><button type="button" className="parameter-help" aria-label={`查看 ${field.label} 说明`} data-tooltip={`${field.help}${field.range ? `\n范围：${field.range}` : ''}`}><icons.CircleHelp size={15} /></button></span>{renderTrainingField(field)}</div>)}</div>
      </details>)}

      <section className="workspace-section training-connection-section">
        <div className="section-title"><icons.Code2 size={21} /><h2>训练位置与 SSH</h2></div>
        <p className="section-description">服务器信息全部留空时在本机执行；填写 IP 和用户名后在远程服务器执行。远程模式需要先测试连接。</p>
        <div className="training-connection-grid">
          <label><span>服务器 IP / 主机名</span><input value={host} onChange={(event) => onParameterChange('remote_host', event.target.value)} placeholder="留空则本地训练" /></label>
          <label><span>SSH 端口</span><input type="number" value={port} onChange={(event) => onParameterChange('remote_port', Number(event.target.value))} /></label>
          <label><span>SSH 用户名</span><input value={username} onChange={(event) => onParameterChange('remote_username', event.target.value)} /></label>
          <label><span>SSH 密码</span><input type="password" autoComplete="current-password" value={displayedPassword} onFocus={() => { if (password === REMEMBERED_PASSWORD_VALUE) onParameterChange('remote_password', '') }} onChange={(event) => onParameterChange('remote_password', event.target.value)} placeholder="首次连接或未保存时输入" /></label>
          <label className="training-remember-password"><span>记住 SSH 密码</span><span className="switch-control"><input type="checkbox" checked={rememberPassword} onChange={(event) => onParameterChange('remember_password', event.target.checked)} /><span /><em>{rememberPassword ? '开启' : '关闭'}</em></span></label>
        </div>
        <div className="remote-run-actions training-connection-actions">
          <button className="connection-test-button" onClick={testConnection} disabled={!remoteRequested || testingConnection || submitting}><icons.Settings size={17} />{testingConnection ? '正在连接…' : connectionReady ? '重新测试连接' : '测试 SSH 连接'}</button>
          <span className={`remote-status ${connectionReady ? 'connected' : testingConnection ? 'connecting' : 'idle'}`}><i />{remoteRequested ? (connectionReady ? '连接成功' : '尚未测试连接') : '本地训练'}</span>
          {remoteRequested && rememberPassword ? <button className="forget-password-button" onClick={forgetPassword} disabled={submitting || testingConnection}>清除已保存密码</button> : null}
        </div>
        {connectionReady && currentConnectionCheck?.result ? <div className="connection-success" role="status"><icons.Check size={18} /><span>{currentConnectionCheck.result.message}<small>设备指纹：{currentConnectionCheck.result.fingerprint}</small></span></div> : null}
        {credentialNotice ? <div className="credential-notice" role="status"><icons.Check size={18} /><span>{credentialNotice}</span></div> : null}
      </section>

      <section className="remote-inference-section training-run-section">
        <div className="section-title"><icons.Play size={21} /><h2>训练任务</h2></div>
        <p className="section-description">可同时启动多个训练；每个任务独立运行、查看日志和终止。请为每个任务设置不同的输出名称，并按显存容量分配 GPU。</p>
        {sessions.length ? <div className="training-session-list" aria-label="训练任务列表">{sessions.map((item) => <button type="button" key={item.id} className={item.id === sessionId ? 'selected' : ''} aria-pressed={item.id === sessionId} onClick={() => selectSession(item.id)}>
          <strong>{item.output.split('/').pop() || item.id}</strong><span>{item.model.split('/').pop()} · {item.remote ? item.host : '本地'} · GPU {item.device}</span><em>{item.status === 'running' ? '训练中' : item.status === 'starting' ? '准备中' : item.status === 'disconnected' ? '待重新连接' : item.status === 'stopping' ? '终止中' : item.status === 'completed' ? '已完成' : item.status === 'failed' ? '失败' : '已停止'}</em>
        </button>)}</div> : <p className="section-empty">当前没有训练任务。</p>}
        <div className="remote-run-actions">
          <button className="run-button" onClick={start} disabled={submitting || testingConnection || profilesLoading || (remoteRequested && !connectionReady)}>
            <icons.Play size={18} fill="currentColor" />
            {submitting ? '请稍候…' : remoteRequested ? '再开一个远程训练' : '再开一个本地训练'}
          </button>
          {session?.status === 'disconnected' ? <button onClick={reconnect} disabled={submitting}>重新连接远端训练</button> : null}
          {activeSession && session?.status !== 'disconnected' ? <button className="stop-run-button" onClick={stop} disabled={submitting || session?.status === 'stopping'}><icons.X size={17} />终止选中任务</button> : null}
          {session && terminalStates.has(session.status) ? <button className="stop-run-button" onClick={deleteSelectedSession} disabled={submitting}><icons.X size={17} />删除选中任务记录</button> : null}
          <span className={`remote-status ${session?.status || (connectionReady ? 'connected' : 'idle')}`}><i />{statusLabel}</span>
          <small>{session?.status === 'disconnected'
            ? `将使用任务保存的连接信息 ${session.username}@${session.host}:${session.port}；若密码未保存，请先在上方 SSH 密码框输入。`
            : '远程训练在独立 tmux 会话中运行；关闭平台不会终止，重开后可重新连接查看日志。本地训练仍随平台关闭而停止。'}</small>
        </div>
        {connectionError || error || session?.error ? <div className="remote-error" role="alert"><icons.CircleAlert size={18} /><span>{connectionError || error || session?.error}</span></div> : null}
        <details className="run-output" open>
          <summary><icons.ChevronRight size={18} />训练日志</summary>
          <pre>{session?.logs.length ? session.logs.join('\n') : '开始训练后，将在这里持续显示 Ultralytics YOLO 的训练日志'}</pre>
        </details>
      </section>
    </div>
  )
}
