"""Local PT auto annotation with cancellable, isolated inference."""
from __future__ import annotations

import json
import math
import os
import queue
import signal
import subprocess
import tempfile
import threading
from pathlib import Path

from .handlers import RunContext, register_handler
from .jetson_onnx_export import _resolve_python


def run_auto_annotation(context: RunContext) -> dict:
    folder = context.paths.get('image_folder')
    weights = context.paths.get('weights_file')
    if not folder or not folder.is_dir():
        raise ValueError('请选择存在的本机图片文件夹。')
    if not weights or not weights.is_file() or weights.suffix.lower() != '.pt':
        raise ValueError('请选择存在的 PT 模型文件。')
    p = context.parameters
    config = {'folder': str(folder.resolve()), 'weights': str(weights.resolve()),
              'recursive': bool(p.get('recursive', True)), 'existing': p.get('existing', '跳过'),
              'empty_xml': bool(p.get('empty_xml', True)), 'device': str(p.get('device') or '').strip()}
    if config['existing'] not in ('跳过', '备份后覆盖'):
        raise ValueError('请选择有效的已有标注处理方式。')
    for field, default in [('conf', .5), ('iou', .7)]:
        value = float(p.get(field, default))
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{field} 必须在 0–1 之间。')
        config[field] = value
    size = float(p.get('imgsz', 640))
    if not math.isfinite(size) or not size.is_integer() or not 32 <= size <= 4096 or size % 32:
        raise ValueError('推理尺寸必须是 32–4096 范围内的 32 的整数倍。')
    config['imgsz'] = int(size)
    try:
        config['classes'] = [int(v.strip()) for v in str(p.get('classes') or '').replace('，', ',').split(',') if v.strip()]
        if any(v < 0 for v in config['classes']):
            raise ValueError()
    except ValueError:
        raise ValueError('类别编号必须是用逗号分隔的非负整数。') from None
    python = _resolve_python(p.get('python_path'))
    context.report(f'自动标注环境：{python}；XML 保存于原图片旁。')
    context.check_cancelled()
    with tempfile.TemporaryDirectory(prefix='yolo-annotation-') as temp:
        settings = Path(temp) / 'config.json'
        settings.write_text(json.dumps(config, ensure_ascii=False), encoding='utf-8')
        process = subprocess.Popen([str(python), '-u', '-B', str(Path(__file__).with_name('auto_annotation_worker.py')), str(settings)],
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        messages = queue.Queue()
        def read():
            try:
                for line in process.stdout:
                    messages.put(line.rstrip())
            finally:
                messages.put(None)
        threading.Thread(target=read, daemon=True).start()
        result = None
        last = ''
        try:
            while True:
                context.check_cancelled()
                try:
                    line = messages.get(timeout=.2)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if line.startswith('RESULT_JSON:'):
                    result = json.loads(line[len('RESULT_JSON:'):])
                else:
                    last = line or last
                    context.report(line)
            if process.wait() != 0 or result is None:
                raise RuntimeError(last or '自动标注进程异常退出。')
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            process.stdout.close()
    summary = f"生成 {result['written']} 份 XML，{result['objects']} 个目标；空标注 {result['empty']}，跳过 {result['skipped']}，失败 {result['failed']}。"
    context.report(summary)
    if result['backup']:
        context.report(f"原 XML 备份：{result['backup']}")
    if result['failed']:
        raise RuntimeError(summary + ' 已成功生成的 XML 保留，请根据日志检查失败图片。')
    return {'message': summary, 'outputFolders': [str(folder.resolve())], **result}


register_handler('dataset.auto_annotate', run_auto_annotation)
