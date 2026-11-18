"""Bounded SSH image inference; XML and backups always stay on the local PC."""
from __future__ import annotations

import json
import math
import re
import shlex
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from .auto_annotation_worker import collect_images, run
from .remote_inference import RemoteInferenceManager
from .remote_yolo_dataset import resolve_tested_yolo_connection
from .star_labels import detect_star_labels


def model_labels(raw, weights):
    if not str(raw or '').strip():
        return detect_star_labels(weights)['labels']
    entries = [v.strip() for v in re.split(r'[,，\r\n]+', str(raw)) if v.strip()]
    labels = []
    for index, entry in enumerate(entries):
        match = re.match(r'^(\d+)\s*[:：]\s*(.*)$', entry)
        if match:
            if int(match[1]) != index:
                raise ValueError('类别编号必须从 0 开始连续排列，并与模型顺序一致。')
            entry = match[2].strip()
        if not entry or any(ord(c) < 32 for c in entry):
            raise ValueError('类别名称不能为空或含控制字符。')
        labels.append(entry)
    if not labels or len(labels) > 10000:
        raise ValueError('请填写有效的类别名称，按模型类别 ID 顺序排列。')
    return labels


def validate_prediction(payload, identity, labels, classes):
    if payload.get('type') != 'result' or payload.get('id') != identity:
        raise RuntimeError('远端标注响应与当前图片不匹配，已停止，避免生成错误标注。')
    if payload.get('error'):
        raise ValueError(str(payload['error']))
    shape = payload.get('shape')
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
        raise RuntimeError('远端图片尺寸无效。')
    items = payload.get('detections')
    if not isinstance(items, list) or len(items) > 300:
        raise RuntimeError('远端检测框数据无效。')
    boxes, ids = [], []
    for item in items:
        cls, box, score = item.get('class_id'), item.get('box'), item.get('score')
        if type(cls) is not int or not 0 <= cls < len(labels):
            raise RuntimeError('检测类别编号超出类别名称范围，请核对模型和类别顺序。')
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)
                or type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
                or box[2] <= box[0] or box[3] <= box[1]):
            raise RuntimeError('远端检测框坐标或置信度无效。')
        if not classes or cls in classes:
            boxes.append(box)
            ids.append(cls)
    return SimpleNamespace(orig_shape=shape, boxes=SimpleNamespace(cls=_Values(ids), xyxy=_Values(boxes)))


class _Values:
    def __init__(self, value): self.value = value
    def cpu(self): return self
    def tolist(self): return self.value


class StarSession:
    task = 'detect'

    def __init__(self, context, config, labels):
        self.context, self.config, self.names = context, config, labels
        self.client = self.sftp = self.channel = None
        self.root = '/tmp/yolo-star-annotation-' + uuid.uuid4().hex
        self.files = ['remote_star_annotation_worker.py', 'bundled_run_star_video.py',
                      'model.star', 'config.json', 'image.bin', 'pid']
        self.buffer = b''
        self.error = None
        self.created = False
        self.last_log = ''

    def check(self):
        self.context.check_cancelled()
        if self.error:
            raise RuntimeError(self.error)

    def upload(self, source, name):
        self.check()
        last = time.monotonic()
        def progress(done, total):
            nonlocal last
            self.check()
            if time.monotonic() - last >= 5:
                self.context.report(f'上传 {name}：{done / 1048576:.1f}/{total / 1048576:.1f} MiB')
                last = time.monotonic()
        self.sftp.put(str(source), self.root + '/' + name,
                      callback=progress)

    def start(self):
        connection = resolve_tested_yolo_connection(self.context.parameters, '执行远程 STAR 标注')
        python = str(self.context.parameters.get('remote_python') or 'python3').strip()
        if not python or any(c in python for c in '\0\r\n'):
            raise ValueError('请填写远端 Python 命令或解释器绝对路径。')
        self.python = python
        backend = str(self.context.parameters.get('cuda_backend') or 'auto')
        if backend not in ('auto', 'cuda-driver-ctypes', 'cuda-driver', 'cuda-python', 'pycuda'):
            raise ValueError('CUDA 后端选项无效。')
        paramiko = RemoteInferenceManager._load_paramiko()
        self.client = paramiko.SSHClient()
        self.client.load_system_host_keys()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.context.report(f"正在连接标注设备：{connection['username']}@{connection['host']}:{connection['port']}")
        try:
            self.client.connect(hostname=connection['host'], port=connection['port'],
                                username=connection['username'], password=connection['password'],
                                look_for_keys=False, allow_agent=False, timeout=12, banner_timeout=12, auth_timeout=12)
        except Exception as exc:
            raise RuntimeError(RemoteInferenceManager._connection_error(
                exc, paramiko, connection['host'], connection['port'])) from exc
        self.check()
        self.sftp = self.client.open_sftp()
        self.sftp.get_channel().settimeout(8)
        self.sftp.mkdir(self.root, mode=0o700)
        self.created = True
        for name in self.files[:2]:
            self.upload(Path(__file__).with_name(name), name)
        self.upload(self.config['weights'], 'model.star')
        with self.sftp.open(self.root + '/config.json', 'w') as stream:
            stream.write(json.dumps({'conf': self.config['conf'], 'iou': self.config['iou'],
                                     'num_classes': len(self.names), 'cuda_backend': backend}))
        self.context.report('脚本与 STAR 已上传，正在加载模型；逐张上传图片，不会跳帧。')
        self.channel = self.client.get_transport().open_session(timeout=12)
        self.channel.exec_command(shlex.join([python, '-u', self.root + '/remote_star_annotation_worker.py']))
        ready = self.receive(180)
        if ready.get('type') != 'ready' or ready.get('version') != 1:
            raise RuntimeError('远端标注协议版本不兼容。')
        self.context.report(f"STAR 已加载，模型输入尺寸：{ready.get('input_shape')}；类别：{self.names}")

    def receive(self, timeout=120):
        deadline, heartbeat = time.monotonic() + timeout, time.monotonic() + 5
        while True:
            self.check()
            if self.channel.recv_stderr_ready():
                text = self.channel.recv_stderr(65536).decode('utf-8', errors='replace').strip()
                if text:
                    self.last_log = text[-4000:]
                    self.context.report(text)
            if b'\n' in self.buffer:
                line, self.buffer = self.buffer.split(b'\n', 1)
                try:
                    payload = json.loads(line)
                except (ValueError, UnicodeError) as exc:
                    raise RuntimeError('远端标注响应不是有效 JSON，已停止以保护标注。') from exc
                if not isinstance(payload, dict):
                    raise RuntimeError('远端标注响应格式错误。')
                if payload.get('type') == 'fatal':
                    raise RuntimeError('远端 STAR 标注失败：' + str(payload.get('error')))
                return payload
            if self.channel.recv_ready():
                self.buffer += self.channel.recv(65536)
                if len(self.buffer) > 2 * 1024 * 1024:
                    raise RuntimeError('远端标注响应超过安全大小。')
                continue
            if self.channel.closed or self.channel.exit_status_ready():
                raise RuntimeError('远端标注进程已退出。' + self.last_log)
            now = time.monotonic()
            if now >= deadline:
                raise RuntimeError('远端标注响应超时，请检查盒子资源和远程 Python 环境。')
            if now >= heartbeat:
                self.context.report('仍在等待远端加载/推理，任务未断开…')
                heartbeat = now + 5
            time.sleep(.05)

    def predict(self, source, **_):
        try:
            identity = uuid.uuid4().hex
            self.upload(source, 'image.bin')
            self.channel.sendall((json.dumps({'id': identity}) + '\n').encode())
            response = self.receive()
            return [validate_prediction(response, identity, self.names, self.config['classes'])]
        except ValueError:
            # A single unreadable image may fail without aborting the whole folder.
            raise
        except Exception as exc:
            self.error = str(exc)
            raise

    def close(self):
        warnings = []
        if self.channel is not None:
            try:
                self.channel.shutdown_write()
            except Exception:
                pass
        if self.created and self.sftp is not None:
            # Kill only the exact worker from this task, never a shared Python process.
            try:
                with self.sftp.open(self.root + '/pid') as stream:
                    pid = int(stream.read(32))
                script = (f'import os, signal; p={pid}; '
                          f'c=open("/proc/%d/cmdline"%p,"rb").read().split(b"\\0"); '
                          f'os.kill(p, signal.SIGTERM) if {self.root + "/remote_star_annotation_worker.py"!r}.encode() in c else None')
                _, stdout, _ = self.client.exec_command(shlex.join([self.python, '-c', script]), timeout=5)
                end = time.monotonic() + 3
                while not stdout.channel.exit_status_ready() and time.monotonic() < end:
                    time.sleep(.05)
                stdout.channel.close()
            except Exception:
                # The process may already have exited before its pid file is read.
                pass
            for name in self.files:
                try:
                    self.sftp.remove(self.root + '/' + name)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    warnings.append(str(exc))
            try:
                self.sftp.rmdir(self.root)
            except Exception as exc:
                warnings.append(str(exc))
            self.context.report('远端本次上传的模型、图片和脚本已清理。' if not warnings
                                else '远端临时文件清理未完成，请检查：' + self.root)
        if self.channel is not None:
            self.channel.close()
        if self.sftp is not None:
            self.sftp.close()
        if self.client is not None:
            self.client.close()


def run_remote_annotation(context, config):
    context.check_cancelled()
    images = collect_images(config)
    context.check_cancelled()
    context.report(f'扫描到 {len(images)} 张本机图片，开始准备 STAR 标注。')
    labels = model_labels(context.parameters.get('star_labels'), config['weights'])
    if config['classes'] and not set(config['classes']).issubset(range(len(labels))):
        raise ValueError('类别编号超出 STAR 类别名称范围。')
    session = StarSession(context, config, labels)
    try:
        session.start()
        result = run(config, session, context.report, session.check, images=images)
    finally:
        session.close()
    summary = (f"STAR 自动标注：生成 {result['written']} 份 XML，{result['objects']} 个目标；"
               f"空标注 {result['empty']}，跳过 {result['skipped']}，失败 {result['failed']}。")
    context.report(summary)
    if result['backup']:
        context.report('原 XML 备份：' + result['backup'])
    if result['failed']:
        raise RuntimeError(summary + ' 已生成的 XML 保留，请检查失败图片。')
    return {'message': summary, 'outputFolders': [config['folder']], **result}
