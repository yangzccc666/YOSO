import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from unittest.mock import Mock, patch

from backend import annotation_process as worker
from backend.task_manager import TaskCancelled

RESULT = dict(written=2255, objects=2255, empty=0, skipped=0, failed=0, backup='')


class AnnotationProcessTests(unittest.TestCase):
    def supervise(self, code, logs=None, cancel=lambda: None):
        return worker.supervise_annotation([sys.executable, '-u', '-c', code], (logs if logs is not None else []).append, cancel)

    def test_large_utf8_progress_never_uses_parent_gbk_default_or_blocks_pipe(self):
        logs = []
        code = "import sys, json; assert sys.stdout.encoding.lower() == 'utf-8'; " \
               "[print(f'[{i}/2255] 中.jpg：标注完成 工件 合格') for i in range(2255)]; " \
               f"print('RESULT_JSON:' + json.dumps({RESULT!r}))"
        with patch.dict(os.environ, {'PYTHONIOENCODING': 'gbk'}):
            result = self.supervise(code, logs)
        self.assertEqual(result, RESULT)
        self.assertEqual(len(logs), 2255)
        self.assertIn('中.jpg', logs[0])
        self.assertIn('[2254/2255]', logs[-1])

    def test_bad_third_party_log_bytes_do_not_end_reader(self):
        logs = []
        result = self.supervise("import sys,json; sys.stdout.buffer.write(b'bad-byte:\\xff\\n'); sys.stdout.flush(); "
                                f"print('RESULT_JSON:' + json.dumps({RESULT!r}))", logs)
        self.assertEqual(result, RESULT)
        self.assertIn('\ufffd', logs[0])

    def test_reader_error_is_reported_instead_of_waiting_for_worker_forever(self):
        class Broken:
            def __iter__(self): raise OSError('simulated pipe failure')
            def close(self): pass
        process = Mock(pid=1234, stdout=Broken())
        process.poll.return_value = None
        process.wait.return_value = 0
        with patch.object(worker.subprocess, 'Popen', return_value=process), patch.object(worker, 'stop_worker') as stop:
            with self.assertRaisesRegex(RuntimeError, '日志读取失败.*simulated pipe failure'):
                self.supervise('unused')
            stop.assert_called_once_with(process)

    def test_missing_or_bad_result_packet_is_an_error(self):
        for code in ("print('worker stopped')", "print('RESULT_JSON:not json')", "print('RESULT_JSON:[]')"):
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                self.supervise(code)

    def test_nonzero_exit_reports_worker_error(self):
        with self.assertRaisesRegex(RuntimeError, '7.*模型不兼容'):
            self.supervise("import sys; print('模型不兼容'); sys.exit(7)")

    def test_silent_worker_produces_heartbeat_and_remains_cancellable(self):
        logs = []
        started = time.monotonic()
        def cancel():
            if time.monotonic() - started > .4:
                raise TaskCancelled('manual stop')
        with patch.object(worker, 'HEARTBEAT_SECONDS', .05):
            with self.assertRaises(TaskCancelled):
                self.supervise('import time; time.sleep(60)', logs, cancel)
        self.assertLess(time.monotonic() - started, 5)
        self.assertTrue(any('距上一条日志' in line for line in logs))

    def test_eof_does_not_allow_unbounded_process_wait(self):
        started = time.monotonic()
        with patch.object(worker, 'EXIT_TIMEOUT_SECONDS', .1):
            with self.assertRaisesRegex(RuntimeError, '日志管道已关闭'):
                self.supervise('import os,time; os.close(1); os.close(2); time.sleep(60)')
        self.assertLess(time.monotonic() - started, 5)

    def test_windows_cleanup_uses_terminate_then_kill_without_posix_signals(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('worker', 3), 0]
        with patch.object(worker.os, 'name', 'nt'), patch.object(worker.os, 'killpg', create=True) as killpg:
            worker.stop_worker(process)
        process.terminate.assert_called_once()
        process.kill.assert_called_once()
        killpg.assert_not_called()

    def test_worker_reports_image_start_before_prediction(self):
        source = Path(__file__).resolve().parents[1] / 'backend/auto_annotation_worker.py'
        text = source.read_text(encoding='utf-8')
        self.assertLess(text.index('开始检测：'), text.index('prediction = model.predict'))
        self.assertIn('show=False', text)
