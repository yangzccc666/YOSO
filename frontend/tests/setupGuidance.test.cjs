const { test } = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const vm = require('node:vm')
const ts = require('typescript')
const context = { exports: {} }
const source = fs.readFileSync(path.join(__dirname, '../src/setupGuidance.ts'), 'utf8')
vm.runInNewContext(ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } }).outputText, context)
const { setupGuideFor, createSetupGuidanceMonitor } = context.exports

test('local PT and ONNX failures open the correct setup section without echoing paths', () => {
  const guide = setupGuideFor('local-ai', { error: '指定的 YOLO Python 不存在：C:\\示例\\python.exe' })
  assert.equal(guide.code, 'ai-python')
  assert.equal(guide.settings, 'ai')
  assert(!JSON.stringify(guide).includes('示例'))
  assert.equal(setupGuideFor('local-ai', { error: "No module named 'onnxruntime'" }).code, 'onnx')
  assert.equal(setupGuideFor('local-ai', { error: '当前 ONNX Runtime 不支持 CUDA，请选择 cpu' }).code, 'onnx-cuda')
})

test('remote errors explain the remote machine, not the local Python settings', () => {
  assert.equal(setupGuideFor('remote', { logs: ['ERROR: No usable CUDA backend.'] }).code, 'remote-inference')
  assert.equal(setupGuideFor('train', { error: "No module named 'ultralytics'" }).location, 'SSH 训练服务器')
  assert.equal(setupGuideFor('build', { error: '远端 trtexec 不存在或不可执行' }).settings, undefined)
  assert.equal(setupGuideFor('run', { error: '远程 Python 缺少 Pillow' }).location, 'SSH 服务器')
  assert.equal(setupGuideFor('ssh', { error: 'SSH 身份验证失败：请检查用户名和密码。' }).code, 'ssh')
})

test('actionable runtime errors found in worker logs are classified without masking the original error', () => {
  assert.equal(setupGuideFor('run', { error: 'Docker ONNX 量化异常结束', messages: ["No module named 'easyonnxquant'"] }).code, 'docker')
  assert.equal(setupGuideFor('run', { error: '请填写当前电脑的 package_tool 路径。' }).code, 'star-package')
  assert.equal(setupGuideFor('export', { error: 'YOLOv5 仓库不完整' }).code, 'export')
  assert.equal(setupGuideFor('external', { error: 'X-AnyLabeling 启动程序不存在' }).code, 'external')
  assert.equal(setupGuideFor('run', { error: '未找到 FFmpeg。' }).settings, 'dependencies')
})

test('no configuration popup for cancellation, activation, invalid input or malformed models', () => {
  assert.equal(setupGuideFor('run', { code: 'task_cancelled', messages: ['未找到 FFmpeg'] }), undefined)
  assert.equal(setupGuideFor('run', { code: 'activation_required', error: '未找到 FFmpeg' }), undefined)
  assert.equal(setupGuideFor('run', { error: '视频文件不存在' }), undefined)
  assert.equal(setupGuideFor('local-ai', { error: 'ONNX 格式无效' }), undefined)
})

test('asynchronous failures are shown once for new submissions, not historical failures', () => {
  const notices = []
  const observe = createSetupGuidanceMonitor(guide => notices.push(guide.code))
  const failed = { id: 'id', status: 'failed', error: 'No usable CUDA backend' }
  observe('/api/remote-inference/id/status', 'GET', true, failed)
  observe('/api/remote-inference/start', 'POST', true, { id: 'id', status: 'uploading' })
  observe('/api/remote-inference/id/status', 'GET', true, failed)
  observe('/api/remote-inference/id/status', 'GET', true, failed)
  observe('/api/tensorrt-build/queue', 'POST', true, { addedIds: ['new'], items: [] })
  observe('/api/tensorrt-build/queue', 'GET', true, { items: [
    { id: 'old', status: 'failed', message: '远端 trtexec 不存在' },
    { id: 'new', status: 'failed', message: '远端 trtexec 不存在' },
  ] })
  observe('/api/functions/local_model/run', 'POST', false, { error: '未内置可用的 Ultralytics/PyTorch' })
  assert.deepEqual(notices, ['remote-inference', 'trtexec', 'ai-python'])
})
