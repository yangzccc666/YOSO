export type SettingsSection = 'dependencies' | 'ai' | 'external'

export type SetupGuide = {
  code: string
  title: string
  location: string
  steps: string[]
  settings?: SettingsSection
}

export const SETUP_GUIDANCE_EVENT = 'yolo-setup-guidance'

const ai: SetupGuide = {
  code: 'ai-python', title: '本机 AI Python 环境不可用', location: '这台电脑', settings: 'ai',
  steps: [
    '已有 YOLO 环境：选择安装了 Ultralytics 和 PyTorch 的解释器（Windows 为 python.exe，Linux/macOS 为 bin/python）。不要选择平台的 exe 或 Conda 程序。',
    '没有环境：Windows 可解压安装包内的 AI-Environment-Setup，按其中的 README 使用 install_yolo_env.bat 创建 CPU 环境；也可用 Python 的 venv 创建环境，再用该环境的 python -m pip install ultralytics 安装 CPU 推理依赖。GPU 用户按显卡驱动选择对应的 PyTorch。',
    '在「本机设置 → 本地 AI 运行时」选择该解释器并保存，再回到功能页重试。若环境变量 PROCESSING_VIEW_YOLO_PYTHON 已设置，请同步修正或取消旧路径。',
  ],
}

type Scope = 'run' | 'local-ai' | 'export' | 'remote' | 'train' | 'build' | 'ssh' | 'external'
type Evidence = { error?: unknown; message?: unknown; logs?: unknown; messages?: unknown; code?: unknown; status?: unknown; id?: unknown }

export function setupGuideFor(scope: Scope, evidence: Evidence): SetupGuide | undefined {
  if (evidence.code === 'task_cancelled' || evidence.code === 'activation_required' || evidence.status === 'stopped') return
  const lines = [evidence.error, evidence.message,
    ...(Array.isArray(evidence.logs) ? evidence.logs.slice(-40) : []),
    ...(Array.isArray(evidence.messages) ? evidence.messages.slice(-40) : [])]
  const text = lines.filter(line => typeof line === 'string').map(line => line.slice(0, 2000)).join('\n')
  const guide = (code: string, title: string, location: string, steps: string[], settings?: SettingsSection): SetupGuide =>
    ({ code, title, location, steps, settings })

  if (scope === 'external' && /启动程序|工作目录不存在|外部工具未配置|未找到 Conda|启动失败/.test(text)) {
    return guide('external', '外部工具无法启动', '这台电脑', [
      '确认外部软件已安装或解压，按原日志定位失败的工具。', '在「本机设置 → 外部工具管理」填实际的启动程序；需要时补充工作目录、参数和 Conda 环境。保存后从外部工具菜单重试。',
    ], 'external')
  }
  if (/No usable CUDA backend|TensorRT Python is not installed|No module named ['"]tensorrt['"]|远端 Python 解释器路径|请填写远端 Python 命令/.test(text) && ['remote', 'run'].includes(scope)) {
    return guide('remote-inference', '核对盒子的推理环境', 'SSH 连接的推理盒子', [
      '在盒子上使用与 GPU 驱动、TensorRT 版本匹配的 Python，确认能导入 TensorRT、OpenCV 和 NumPy，并能访问 CUDA。不要在本机安装这些库来修复盒子的报错。',
      '将验证过的解释器绝对路径填入当前功能的「远端 Python」参数；同时核对 STAR 与盒子 GPU 及 TensorRT 版本匹配。重新测试 SSH 后运行。',
    ])
  }
  if (['ssh', 'remote', 'train', 'build', 'run'].includes(scope) && /本机尚未安装 SSH 组件/.test(text)) {
    return guide('ssh-client', '平台的 SSH 组件不可用', '这台电脑', [
      '标准安装包已内置 Paramiko。若使用安装包，请重新完整解压；若从源码运行，请用平台本身使用的解释器安装 requirements.txt 后重开平台。',
    ], 'dependencies')
  }
  if (['ssh', 'remote', 'train', 'build', 'run'].includes(scope) && /请先.{0,20}测试.{0,8}SSH|请填写并测试 SSH|请输入 SSH 用户名|请输入本次 SSH 登录密码|尚未保存此设备的 SSH 密码|SSH.{0,12}(认证失败|身份验证失败|连接失败|登录失败|握手失败)|连接被拒绝：|网络不可达：|连接超时：|无法解析设备地址|没有开放 SSH 服务|Authentication failed/i.test(text)) {
    return guide('ssh', '检查 SSH 连接配置', '这台电脑与目标设备', [
      '核对当前功能区的服务器 IP／主机名、端口、用户名和密码。本机必须能访问目标设备；网络超时和密码错误的处理不同，请看原日志。',
      '点击「测试 SSH 连接」，通过后再启动任务。测试连接不会启动训练、构建或推理。',
    ])
  }
  if (scope === 'train' && /服务器缺少 tmux|远程 YOLO 可执行文件不存在|远程登录环境找不到|远程 Ultralytics 源码目录不存在|No module named ['"](?:ultralytics|torch)['"]/i.test(text)) {
    return guide('training', '核对远程训练环境', 'SSH 训练服务器', [
      '根据日志检查服务器上的 tmux、YOLO 可执行文件和 Python 依赖。使用自定义 Ultralytics 时还需核对源码目录和依赖版本。',
      '在「YOLO 模型训练」填写可执行文件绝对路径（例如环境的 bin/yolo），核对源码目录、data.yaml、训练设备，再重新测试 SSH 并运行。',
    ])
  }
  if (['build', 'run'].includes(scope) && /远端 trtexec 不存在|既没有 tmux，也没有 setsid|无法读取远端 TensorRT 版本|构建需盒子可运行 python3/.test(text)) {
    return guide('trtexec', '核对远程 TensorRT 构建环境', 'SSH 连接的推理盒子', [
      '在盒子里确认 TensorRT 与驱动匹配、trtexec 可执行；后台构建还需要 tmux 或 setsid。FP16/FP32 的远端检查需要 python3。',
      '将 trtexec 的盒子端绝对路径填入当前构建功能，确认设备后先点「检查远程环境」，通过后再提交。',
    ])
  }
  if (scope === 'run' && /远程 Python 缺少 Pillow/.test(text)) {
    return guide('calibration-pillow', '量化数据集环境缺少 Pillow', 'SSH 服务器', [
      '用制作量化数据集所选的远程解释器运行 python -m pip install Pillow，或改用服务器上已安装 Pillow 的 Python。',
      '在当前功能的「远端 Python」填写该解释器的路径，再次运行。图片和输出路径也应填写服务器上的路径。',
    ])
  }
  if (scope === 'run' && /本机未找到 Docker|无法访问 Docker 容器|Cannot connect to the Docker daemon|Error response from daemon: No such container|No module named ['"]easyonnxquant|EnvironmentNameNotFound|容器内 Conda 路径必须是绝对路径/i.test(text)) {
    return guide('docker', '核对 Docker 量化环境', '这台电脑和量化容器', [
      '确认 Docker 已安装并启动，当前系统账号可访问目标容器。容器内需要项目使用的 EasyONNXQuant 和对应 Python 环境。',
      '在「Docker ONNX 量化」填实际的容器名、容器内 Conda 路径和环境名，确认模型和图片所在目录已经挂载，再测试 Docker 环境。',
    ])
  }
  if (scope === 'run' && /未找到 FFmpeg/.test(text)) {
    return guide('ffmpeg', '未找到 FFmpeg', '这台电脑', [
      '安装包用户先确认已完整解压下载的压缩包，并从解压目录启动。源码运行用户在运行平台的 Python 环境安装 imageio-ffmpeg，或将系统 FFmpeg 加入 PATH。',
      '依赖调整后可在「本机设置 → 分发依赖检查」查看，再试一次视频裁剪。',
    ], 'dependencies')
  }
  if (scope === 'run' && /请填写当前电脑的 package_tool 路径|打包工具不存在|打包工具不可执行/.test(text)) {
    return guide('star-package', 'STAR 打包工具未配置或不可用', '这台电脑', [
      '从工具提供方获取适合当前操作系统的 package_tool；普通安装包不附带第三方打包程序。',
      '在「本地 STAR 模型打包」选择它的实际文件路径，确认有执行权限后重试。TOML 留空时可使用平台内置模板。',
    ])
  }
  if (['export', 'run'].includes(scope) && /YOLOv5 模型需要填写源码|YOLOv5 仓库不完整|经典 YOLOv5 类别读取依赖|无法启动该页配置的 YOLO Python|(?:^|\n)(?:服务异常：)?YOLO Python 不存在/.test(text)) {
    return guide('export', '核对 ONNX 导出的 Python 和模型版本', '这台电脑', [
      '在 Jetson ONNX 导出页选择训练权重对应的 YOLO26 或 YOLOv5；如填写了本页的 YOLO Python，确认路径指向可运行的解释器。',
      '经典 YOLOv5 需提供含 export.py 和 models 的匹配源码仓库及其 Python 环境。YOLO26 可使用本机设置的 AI Python 并安装相应导出依赖。',
    ], 'ai')
  }
  if (['local-ai', 'export', 'run'].includes(scope)) {
    if (/ONNX Runtime 不支持 CUDA|CUDAExecutionProvider.{0,60}not (?:available|in available)/i.test(text)) {
      return guide('onnx-cuda', '当前 ONNX 环境不能使用所选 GPU', '这台电脑', [
        '可将当前功能的推理设备改为 cpu 后重试。需要 GPU 时，在选定的 AI Python 环境中安装与显卡驱动、CUDA 和 cuDNN 匹配的 onnxruntime-gpu。',
      ], 'ai')
    }
    if (/AI Python 缺少 onnx|No module named ['"](?:onnxruntime|onnx|onnxslim)['"]/i.test(text)) {
      return guide('onnx', '本机 AI 环境缺少 ONNX 依赖', '这台电脑', [
        '在平台实际选择的 AI Python 环境中安装日志提示的包。ONNX 推理通常需要 onnx 和 onnxruntime；如导出日志提示缺少 onnxslim，也需在导出所用环境补装。',
        '核对「本机设置 → 本地 AI 运行时」及导出页单独填写的 Python 路径，保存后重试。',
      ], 'ai')
    }
    if (/未内置可用的 Ultralytics\/PyTorch|缺少 ultralytics\/PyTorch|指定的 YOLO Python 不存在|No module named ['"](?:ultralytics|torch|torchvision)['"]/i.test(text)) return ai
  }
}

type ApiPayload = Evidence & { addedIds?: unknown; items?: unknown; sessions?: unknown }

export function createSetupGuidanceMonitor(publish: (guide: SetupGuide) => void) {
  const watching = new Set<string>()
  const watch = (scope: Scope, id: unknown) => {
    if (typeof id !== 'string' || !id) return
    watching.add(`${scope}:${id}`)
    if (watching.size > 256) watching.delete(watching.values().next().value!)
  }
  const report = (scope: Scope, data: Evidence) => {
    const guidance = setupGuideFor(scope, data)
    if (guidance) publish(guidance)
  }
  const observeTask = (scope: Scope, data: Evidence) => {
    if (!['failed', 'stopped', 'completed'].includes(String(data.status))) return
    if (!watching.delete(`${scope}:${data.id}`)) return
    if (data.status === 'failed') report(scope, data)
  }
  return (url: string, method: string, ok: boolean, data: ApiPayload) => {
    if (!data || typeof data !== 'object') return
    const parts = url.split('?')[0].split('/')
    const endpoint = parts[2]
    const action = parts.at(-1)
    const scopes: Record<string, Scope> = {
      'local-pt-inference': 'local-ai', 'remote-inference': 'remote', 'jetson-onnx-export': 'export',
      'yolo-training': 'train', 'tensorrt-build': 'build', apps: 'external',
    }
    const scope = scopes[endpoint] || 'run'
    if (method === 'POST' && !ok) {
      if (action === 'test-connection') report('ssh', data)
      else if ((endpoint === 'functions' && action === 'run') ||
        (scopes[endpoint] && ['start', 'launch', 'detect-labels', 'queue', 'reconnect'].includes(action || ''))) report(scope, data)
      return
    }
    if (!ok) return
    if (method === 'POST' && ['start', 'reconnect'].includes(action || '') && ['local-ai', 'remote', 'train'].includes(scope)) {
      watch(scope, data.id)
      observeTask(scope, data)
    } else if (method === 'GET' && action === 'status' && ['local-ai', 'remote', 'train'].includes(scope)) {
      observeTask(scope, data)
    } else if (endpoint === 'yolo-training' && method === 'GET' && action === 'sessions' && Array.isArray(data.sessions)) {
      data.sessions.forEach(session => observeTask('train', session))
    } else if (endpoint === 'tensorrt-build' && Array.isArray(data.items)) {
      if (method === 'POST' && Array.isArray(data.addedIds)) data.addedIds.forEach(id => watch('build', id))
      data.items.forEach(item => { if (item) observeTask('build', item) })
    }
  }
}
