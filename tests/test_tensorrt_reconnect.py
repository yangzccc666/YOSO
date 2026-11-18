"""Persistence and remote reattachment without launching a real SSH/build task."""
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from backend.handlers import RunContext
from backend.task_manager import PlatformTaskManager, TaskCancelled
from backend.tensorrt_build_queue import TensorRTBuildQueue
from backend.remote_tensorrt_build import (
    RemoteBuildDisconnected, RemoteBuildFailed, _follow_job, _run_one_job,
    discover_remote_builds, reconnect_remote_build, validate_remote_job,
)


def job_state(source='/models/a.onnx'):
    return validate_remote_job({'source': source, 'token': 'yolo-trt-123456abcdef',
        'remoteDirectory': f'/home/wel/state/{Path(source).stem}-123456abcdef', 'launcher': 'tmux'})


class PersistentQueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp.name) / 'queue.json'
        self.queues = []
        self.parameters = {'host': 'box-a', 'port': 22, 'username': 'wel', 'password': 'do-not-store',
                           'onnx_files': '/models/a.onnx', 'build_mode': 'FP16', 'trtexec_path': '/custom/trtexec'}

    def tearDown(self):
        for queue in self.queues:
            queue.close()
        for queue in self.queues:
            for thread in [queue._worker, *queue._followers]:
                if thread:
                    thread.join(2)
        self.temp.cleanup()

    def queue(self, runner=None, **kwargs):
        queue = TensorRTBuildQueue(PlatformTaskManager(), runner=runner or Mock(return_value={}),
            storage=self.storage, resolve_connection=kwargs.pop('resolve_connection', lambda raw: {
                'host': raw['host'], 'port': raw.get('port', 22), 'username': raw['username'],
                'password': raw.get('password') or 'vault-secret'}), **kwargs)
        self.queues.append(queue)
        return queue

    def until(self, predicate):
        end = time.monotonic() + 4
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.01)
        self.fail('timed out')

    def save_disconnected(self, queued=False):
        launched = threading.Event()
        def run(context):
            context.checkpoint(job_state())
            context.report('Engine generation in progress')
            launched.set()
            context.detach_event.wait(4)
            raise RemoteBuildDisconnected('monitor detached')
        queue = self.queue(run)
        params = {**self.parameters, 'onnx_files': '/models/a.onnx\n/models/b.onnx' if queued else '/models/a.onnx'}
        submitted = queue.enqueue('remote_tensorrt_build', 'TensorRT', {}, params)
        self.assertTrue(launched.wait(3))
        queue.close()
        queue._worker.join(2)
        return submitted['addedIds']

    def test_restart_retains_manual_pause_configuration_and_no_secrets(self):
        queue = self.queue()
        with queue._condition:
            queue.enqueue('remote_tensorrt_build', 'TRT', {}, {**self.parameters, 'token': 'private-token'})
            queue.set_paused(True)
        queue.close()
        raw = self.storage.read_text()
        self.assertNotIn('do-not-store', raw)
        self.assertNotIn('private-token', raw)
        runner = Mock(return_value={})
        restored = self.queue(runner)
        self.assertTrue(restored.snapshot()['paused'])
        self.assertEqual(restored.snapshot()['queuedCount'], 1)
        self.assertEqual(restored._items[0].parameters['trtexec_path'], '/custom/trtexec')
        runner.assert_not_called()
        restored.set_paused(False, parameters=self.parameters)
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'completed')
        runner.assert_called_once()

    def test_reconnect_uses_original_task_never_normal_runner_and_preserves_start(self):
        ids = self.save_disconnected()
        raw = self.storage.read_text()
        normal, finished = Mock(return_value={}), Mock()
        def reconnect(context, remote):
            self.assertEqual(remote, job_state())
            context.report('restored log line')
            return {'message': 'done', 'outputFiles': ['/models/a.plan']}
        restored = self.queue(normal, reconnect_runner=reconnect, on_finished=finished)
        original = restored.snapshot()['items'][0]['startedAt']
        self.assertEqual(restored.snapshot()['disconnectedCount'], 1)
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: finished.call_count == 1 and not restored.snapshot()['items'])
        normal.assert_not_called()
        self.assertEqual(finished.call_args.args[0].started_at, original)
        self.assertIn('restored log line', finished.call_args.args[1]['logs'])
        self.assertIn('/models/a.plan', finished.call_args.args[1]['outputs'])
        self.assertNotIn('do-not-store', raw)

    def test_unconfirmed_remote_blocks_then_successful_reconnect_automatically_continues(self):
        ids = self.save_disconnected(queued=True)
        normal = Mock(return_value={})
        restored = self.queue(normal, reconnect_runner=lambda *_: {})
        with self.assertRaisesRegex(ValueError, '先重新连接'):
            restored.set_paused(False)
        self.assertEqual(restored.snapshot()['queuedCount'], 1)
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: restored.snapshot()['queuedCount'] == 0 and restored.snapshot()['runningCount'] == 0)
        self.assertFalse(restored.snapshot()['paused'])
        normal.assert_called_once()

    def test_enqueue_does_not_bypass_unknown_remote_work(self):
        self.save_disconnected()
        normal = Mock(return_value={})
        restored = self.queue(normal)
        result = restored.enqueue('remote_tensorrt_build', 'TRT', {}, {**self.parameters, 'onnx_files': '/models/b.onnx'})
        self.assertTrue(result['paused'])
        self.assertIn('尚未执行', result['message'])
        self.assertEqual(result['queuedCount'], 1)
        restored.start_pending()
        normal.assert_not_called()

    def test_reconnect_success_preserves_explicit_manual_pause(self):
        ids = self.save_disconnected(queued=True)
        normal = Mock(return_value={})
        restored = self.queue(normal, reconnect_runner=lambda *_: {})
        restored.set_paused(True)
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'completed')
        self.assertTrue(restored.snapshot()['paused'])
        self.assertEqual(restored.snapshot()['queuedCount'], 1)
        normal.assert_not_called()

    def test_reconnect_password_allows_waiting_tasks_to_continue_without_reentry(self):
        ids = self.save_disconnected(queued=True)
        def resolve(raw):
            if not raw.get('password'):
                raise ValueError('请输入 SSH 密码')
            return {key: raw[key] for key in ('host', 'port', 'username', 'password')}
        normal = Mock(return_value={})
        restored = self.queue(normal, resolve_connection=resolve, reconnect_runner=lambda *_: {})
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: restored.snapshot()['queuedCount'] == 0 and restored.snapshot()['runningCount'] == 0)
        normal.assert_called_once()
        self.assertFalse(restored.snapshot()['paused'])
        self.assertNotIn('do-not-store', self.storage.read_text())

    def test_pending_only_restart_autoruns_on_poll_not_during_construction(self):
        queue = self.queue()
        with queue._condition:
            queue.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
            queue.close()
        normal = Mock(return_value={})
        restored = self.queue(normal)
        self.assertFalse(restored.snapshot()['paused'])
        normal.assert_not_called()
        for _ in range(5):
            restored.start_pending()
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'completed')
        normal.assert_called_once()

    def test_legacy_shutdown_pause_does_not_strand_waiting_tasks(self):
        queue = self.queue()
        with queue._condition:
            queue.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
            queue.close()
        saved = json.loads(self.storage.read_text())
        saved.pop('manualPaused')
        saved.pop('pauseKind')
        self.storage.write_text(json.dumps(saved))
        restored = self.queue()
        self.assertFalse(restored.snapshot()['paused'])

    def test_error_pause_survives_restart_until_explicit_enqueue(self):
        queue = self.queue(Mock(side_effect=RuntimeError('build error')))
        queue.enqueue('remote_tensorrt_build', 'TRT', {}, {**self.parameters, 'onnx_files': '/models/a.onnx\n/models/b.onnx'})
        self.until(lambda: queue.snapshot()['paused'])
        queue.close()
        normal = Mock(return_value={})
        restored = self.queue(normal)
        restored.start_pending()
        self.assertTrue(restored.snapshot()['paused'])
        normal.assert_not_called()
        restored.enqueue('remote_tensorrt_build', 'TRT', {}, {**self.parameters, 'onnx_files': '/models/b.onnx'})
        self.until(lambda: restored.snapshot()['queuedCount'] == 0 and restored.snapshot()['runningCount'] == 0)
        normal.assert_called_once()

    def test_connection_failure_stays_reconnectable_without_finished_history(self):
        ids = self.save_disconnected()
        finished = Mock()
        restored = self.queue(reconnect_runner=Mock(side_effect=OSError('connection dropped')), on_finished=finished)
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: restored.snapshot()['disconnectedCount'] == 1)
        self.assertIn('connection dropped', restored.snapshot()['items'][0]['message'])
        self.assertIsNone(restored.snapshot()['items'][0]['finishedAt'])
        finished.assert_not_called()
        self.assertEqual(restored._tasks.active(), [])

    def test_reconnect_identity_checked_and_duplicate_click_cannot_start_two_followers(self):
        ids = self.save_disconnected()
        entered = threading.Event()
        def reconnect(context, _):
            entered.set()
            context.detach_event.wait(4)
            raise RemoteBuildDisconnected('closing')
        restored = self.queue(reconnect_runner=reconnect)
        with self.assertRaisesRegex(ValueError, 'SSH 设备'):
            restored.reconnect(ids[0], {**self.parameters, 'host': 'other'})
        restored.reconnect(ids[0], self.parameters)
        self.assertTrue(entered.wait(2))
        with self.assertRaisesRegex(ValueError, '不是待重新连接'):
            restored.reconnect(ids[0], self.parameters)
        self.assertEqual(len(restored._tasks.active()), 1)
        with self.assertRaisesRegex(ValueError, '不能移除'):
            restored.remove(ids[0], forget=True)

    def test_explicit_stop_after_reconnect_is_distinct_from_close(self):
        ids = self.save_disconnected()
        entered = threading.Event()
        def reconnect(context, _):
            entered.set()
            context.stop_event.wait(3)
            context.check_cancelled()
        restored = self.queue(reconnect_runner=reconnect)
        restored.reconnect(ids[0], self.parameters)
        self.assertTrue(entered.wait(2))
        restored.stop(ids[0])
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'stopped')

    def test_close_detaches_without_cancelling_and_late_thread_cannot_overwrite_new_state(self):
        ids = self.save_disconnected()
        restored = self.queue()
        with self.assertRaisesRegex(ValueError, '不能移除'):
            restored.remove(ids[0])
        restored.remove(ids[0], forget=True)
        persisted = self.storage.read_text()
        self.queues[0]._save()
        self.assertEqual(self.storage.read_text(), persisted)
        self.assertEqual(restored.snapshot()['items'], [])

    def test_remote_exit_failure_is_terminal_not_retryable_network_failure(self):
        ids = self.save_disconnected()
        finished = Mock()
        restored = self.queue(reconnect_runner=Mock(side_effect=RemoteBuildFailed('exit code 1')), on_finished=finished)
        restored.reconnect(ids[0], self.parameters)
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'failed')
        self.until(lambda: finished.call_count == 1)

    def test_missing_credential_retains_waiting_task_instead_of_losing_it(self):
        queue = self.queue()
        with queue._condition:
            queue.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
            queue.set_paused(True)
        queue.close()
        def resolve(raw):
            if not raw.get('password'):
                raise ValueError('请输入 SSH 密码')
            return {key: raw[key] for key in ('host', 'port', 'username', 'password')}
        runner = Mock(return_value={})
        restored = self.queue(runner, resolve_connection=resolve)
        restored.set_paused(False)
        self.until(lambda: restored.snapshot()['paused'])
        self.assertEqual(restored.snapshot()['queuedCount'], 1)
        runner.assert_not_called()
        restored.set_paused(False, parameters=self.parameters)
        self.until(lambda: restored.snapshot()['items'][0]['status'] == 'completed')
        runner.assert_called_once()

    def test_damaged_storage_is_visible_and_not_replaced(self):
        self.storage.write_text('broken json')
        restored = self.queue()
        self.assertTrue(restored.snapshot()['paused'])
        self.assertIn('无法安全恢复', restored.snapshot()['pauseReason'])
        with self.assertRaisesRegex(ValueError, '无法安全恢复'):
            restored.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
        restored.close()
        self.assertEqual(self.storage.read_text(), 'broken json')

    def test_discovery_imports_only_once_does_not_launch_or_leak_password(self):
        runner = Mock()
        queue = self.queue(runner)
        with patch('backend.tensorrt_build_queue.discover_remote_builds', return_value={'jobs': [job_state()], 'skipped': 2}):
            first = queue.discover('remote_tensorrt_build', 'TRT', self.parameters)
            second = queue.discover('remote_tensorrt_build', 'TRT', self.parameters)
        self.assertEqual(first['disconnectedCount'], 1)
        self.assertEqual(len(second['items']), 1)
        self.assertIn('旧目录', first['message'])
        self.assertNotIn('do-not-store', self.storage.read_text())
        runner.assert_not_called()

    def test_failed_enqueue_save_does_not_leave_a_phantom_task(self):
        queue = self.queue()
        with patch.object(queue, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                queue.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
        self.assertEqual(queue.snapshot()['items'], [])
        self.assertIsNone(queue._worker)

    def test_disk_failure_on_close_does_not_prevent_safe_shutdown(self):
        queue = self.queue()
        with queue._condition:
            queue.enqueue('remote_tensorrt_build', 'TRT', {}, self.parameters)
            queue.set_paused(True)
        saved = self.storage.read_text()
        with patch.object(queue, '_save', side_effect=OSError('disk full')):
            queue.close()
        self.assertTrue(queue._detach.is_set())
        self.assertEqual(self.storage.read_text(), saved)
        self.assertNotIn('password', queue._items[0].parameters)


class RemoteRecoveryTests(unittest.TestCase):
    def test_completed_job_reconnect_only_reads_exit_log_and_checks_plan(self):
        client = MagicMock()
        reports = []
        context = RunContext('trt', {}, {'password': 'secret'}, reports.append)
        with patch('backend.remote_tensorrt_build._connect', return_value=(client, {})), \
             patch('backend.remote_tensorrt_build._remote_capture', side_effect=[(0, 'log line', ''), (0, '0', ''), (0, '4096', '')]) as capture:
            result = reconnect_remote_build(context, job_state())
        commands = '\n'.join(call.args[1] for call in capture.call_args_list)
        self.assertNotIn('new-session', commands)
        self.assertNotIn('--onnx=', commands)
        self.assertNotIn('kill', commands)
        self.assertNotIn('rm ', commands)
        self.assertIn('/models/a.plan', result['outputFiles'])
        self.assertIn('log line', reports)
        self.assertEqual(context.parameters['password'], '')
        client.close.assert_called_once()

    def test_detach_never_sends_kill_or_delete(self):
        event = threading.Event(); event.set()
        context = RunContext('trt', {}, {}, lambda _: None, detach_event=event)
        with patch('backend.remote_tensorrt_build._remote_capture') as capture:
            with self.assertRaises(RemoteBuildDisconnected):
                _follow_job(Mock(), 'token', '/task', '/log', '/plan', 'tmux', context)
        capture.assert_not_called()

    def test_exit_file_race_is_rechecked_before_marking_disconnected(self):
        context = RunContext('trt', {}, {}, lambda _: None)
        with patch('backend.remote_tensorrt_build._remote_capture', side_effect=[(1, '', ''), (1, '', ''), (1, '', ''), (0, '0', '')]):
            _follow_job(Mock(), 'token', '/task', '/log', '/plan', 'tmux', context)

    def test_unknown_state_is_not_success_and_never_auto_relaunches(self):
        context = RunContext('trt', {}, {}, lambda _: None)
        with patch('backend.remote_tensorrt_build._remote_capture', return_value=(1, '', '')) as capture:
            with self.assertRaises(RemoteBuildDisconnected):
                _follow_job(Mock(), 'token', '/task', '/log', '/plan', 'setsid', context)
        self.assertFalse(any('nohup' in call.args[1] or 'new-session' in call.args[1] for call in capture.call_args_list))

    def test_empty_exit_file_during_write_is_not_treated_as_a_failure(self):
        context = RunContext('trt', {}, {}, lambda _: None)
        with patch('backend.remote_tensorrt_build._remote_capture', side_effect=[(1, '', ''), (0, '', ''), (1, '', ''), (0, '0', '')]):
            _follow_job(Mock(), 'token', '/task', '/log', '/plan', 'tmux', context)

    def test_malformed_exit_code_requires_rechecking_not_a_failed_build(self):
        context = RunContext('trt', {}, {}, lambda _: None)
        with patch('backend.remote_tensorrt_build._remote_capture', side_effect=[(1, '', ''), (0, 'corrupted', '')]):
            with self.assertRaises(RemoteBuildDisconnected):
                _follow_job(Mock(), 'token', '/task', '/log', '/plan', 'tmux', context)

    def test_job_paths_and_token_are_validated_before_reconnection(self):
        for patch_values in ({'token': 'invalid'}, {'remoteDirectory': '/'}, {'source': 'relative.onnx'}, {'launcher': 'shell'}):
            with self.subTest(patch_values=patch_values), self.assertRaises(ValueError):
                validate_remote_job({**job_state(), **patch_values})

    def test_metadata_and_local_checkpoint_are_written_before_remote_launch(self):
        client, sftp = MagicMock(), MagicMock()
        client.open_sftp.return_value = sftp
        order = []
        context = RunContext('trt', {}, {}, lambda _: None, checkpoint=lambda state: order.append('checkpoint'))
        source = Path('/models/a.onnx')
        config = {'launcher': 'tmux', 'keep_remote': True, 'overwrite': False, 'trtexec': '/bin/trtexec', 'mode_flags': [], 'benchmark': False}
        def capture(_, command):
            self.assertTrue(command.startswith('tmux new-session'))
            self.assertEqual(order, ['checkpoint'])
            sftp.open.assert_called_once()
            order.append('launch')
            return 0, '', ''
        with patch('backend.remote_tensorrt_build._remote_capture', side_effect=capture), \
             patch('backend.remote_tensorrt_build._follow_existing_job', return_value={'engineSize': 4096}):
            _run_one_job(client, Path('/home/wel/state'), config, {'source': source, 'output_plan': source.with_suffix('.plan'), 'output_log': source.with_suffix('.trtexec.log')}, context, 1, 1)
        self.assertEqual(order, ['checkpoint', 'launch'])

    def test_discover_legacy_directory_requires_original_source_path(self):
        client, sftp = MagicMock(), MagicMock()
        client.open_sftp.return_value = sftp
        sftp.normalize.return_value = '/home/wel'
        sftp.listdir_attr.return_value = [SimpleNamespace(filename='a-123456abcdef', st_mtime=100)]
        sftp.open.side_effect = OSError('no metadata')
        sftp.stat.side_effect = OSError('no pid')
        with patch('backend.remote_tensorrt_build._connect', return_value=(client, {})):
            no_path = discover_remote_builds({})
            with_path = discover_remote_builds({'onnx_files': '/models/a.onnx'})
        self.assertEqual(no_path['jobs'], [])
        self.assertEqual(no_path['skipped'], 1)
        self.assertEqual(with_path['jobs'][0]['source'], '/models/a.onnx')
        client.exec_command.assert_not_called()

    def test_discovery_reads_new_metadata_without_onnx_input(self):
        client, sftp = MagicMock(), MagicMock()
        client.open_sftp.return_value = sftp
        sftp.listdir_attr.return_value = [SimpleNamespace(filename='a-123456abcdef', st_mtime=100)]
        sftp.open.return_value.__enter__.return_value.read.return_value = json.dumps(job_state()).encode()
        parameters = {'remote_workspace': '/home/wel/state'}
        with patch('backend.remote_tensorrt_build._connect', return_value=(client, {})):
            result = discover_remote_builds(parameters)
        self.assertEqual(result['jobs'], [job_state()])
        client.exec_command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
