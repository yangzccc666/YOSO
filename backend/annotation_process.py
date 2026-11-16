"""UTF-8, cancellable worker supervision shared by desktop annotation builds."""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import threading
import time

HEARTBEAT_SECONDS = 15
EXIT_TIMEOUT_SECONDS = 10


def stop_worker(process):
    """Stop this owned worker; Windows has no os.killpg/SIGKILL."""
    if process.poll() is not None:
        return
    try:
        if os.name == 'nt':
            process.terminate()
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            if os.name == 'nt':
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def supervise_annotation(command, report, check_cancelled):
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', PYTHONUNBUFFERED='1')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding='utf-8', errors='replace', env=environment,
                               start_new_session=os.name != 'nt')
    messages = queue.Queue()
    def read():
        try:
            for line in process.stdout:
                messages.put(('line', line.rstrip()))
        except Exception as exc:
            messages.put(('error', str(exc)))
        finally:
            messages.put(('eof', ''))
    reader = threading.Thread(target=read, name=f'annotation-log-{process.pid}', daemon=True)
    reader.start()
    started = last_output = last_notice = time.monotonic()
    ended = None
    result, last = None, ''
    try:
        while True:
            check_cancelled()
            now = time.monotonic()
            if now - last_notice >= HEARTBEAT_SECONDS:
                report(f'标注进程仍在运行：已运行 {int(now-started)} 秒，距上一条日志 {int(now-last_output)} 秒。'
                       '若长时间停在同一图片，可终止后检查该图片或模型。')
                last_notice = now
            if ended is not None:
                try:
                    code = process.wait(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    if now - ended >= EXIT_TIMEOUT_SECONDS:
                        raise RuntimeError('自动标注日志管道已关闭，但子进程未退出；已停止等待，请检查环境。')
                    continue
            try:
                kind, line = messages.get(timeout=.2)
            except queue.Empty:
                continue
            if kind == 'error':
                raise RuntimeError(f'自动标注日志读取失败：{line}')
            if kind == 'eof':
                ended = time.monotonic()
                continue
            last_output = time.monotonic()
            if line.startswith('RESULT_JSON:'):
                try:
                    result = json.loads(line[len('RESULT_JSON:'):])
                    if not isinstance(result, dict) or not all(key in result for key in ('written', 'objects', 'empty', 'skipped', 'failed', 'backup')):
                        raise ValueError('missing result fields')
                except (ValueError, TypeError) as exc:
                    raise RuntimeError('自动标注结果数据格式不正确。') from exc
            else:
                last = line or last
                report(line)
        if code != 0 or result is None:
            raise RuntimeError(f'自动标注进程异常退出（退出码 {code}）：{last or "未收到完整结果"}')
        return result
    finally:
        stop_worker(process)
        reader.join(timeout=2)
        process.stdout.close()
