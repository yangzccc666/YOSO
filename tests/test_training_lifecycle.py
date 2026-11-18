"""Training exit failures must not become endless disconnected sessions."""
import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import yolo_training as training
from backend.task_manager import TaskCancelled


class TrainingExitTests(unittest.TestCase):
    def follow(self, outputs, logs=None, stop=None):
        session = training.TrainingSession(id='oom-test', remote=True, remote_dir='$HOME/state/task')
        for line in logs or []:
            session.add_log(line)
        messages = []
        with patch.object(training, '_remote_capture', side_effect=outputs) as capture:
            training._follow_remote_session(Mock(), session, messages.append, stop or threading.Event())
        return messages, capture

    def test_confirmed_oom_exit_is_failed_with_actionable_message(self):
        with self.assertRaisesRegex(training.RemoteTrainingFailed, 'OOM.*降低 batch'):
            self.follow([(0, 'torch.OutOfMemoryError: CUDA out of memory', ''), (0, '1', '')])

    def test_no_exit_code_but_session_gone_and_oom_log_is_failed(self):
        with self.assertRaisesRegex(training.RemoteTrainingFailed, 'OOM'):
            self.follow([(0, 'CUDA out of memory', ''), (1, '', ''), (1, '', 'no server running'), (1, '', '')])

    def test_recovered_oom_with_zero_exit_is_completed(self):
        messages, capture = self.follow([(0, 'OOM encountered, retrying smaller batch\nTraining complete', ''), (0, '0', '')])
        self.assertEqual(len(messages), 2)
        self.assertFalse(any('kill-session' in call.args[1] for call in capture.call_args_list))

    def test_oom_warning_with_live_session_does_not_fail(self):
        with patch.object(training.time, 'sleep'):
            self.follow([(0, 'CUDA OOM, reducing batch and retrying', ''), (1, '', ''), (0, '', ''),
                         (0, 'epoch 2/200', ''), (0, '0', '')], stop=None)

    def test_exit_file_race_and_empty_file_are_rechecked(self):
        for first_status in (0, 1):
            with self.subTest(first_status=first_status):
                self.follow([(0, 'done', ''), (first_status, '', ''), (1, '', ''), (0, '0', '')])

    def test_exit_137_without_oom_text_does_not_claim_proven_oom(self):
        with self.assertRaisesRegex(training.RemoteTrainingFailed, 'SIGKILL.*不能仅凭退出码确认 OOM'):
            self.follow([(0, 'Killed', ''), (0, '137', '')])

    def test_unreadable_or_corrupt_state_is_disconnected_not_a_proven_failure(self):
        for outputs in ([(0, '', ''), (0, 'bad data', '')],
                        [(0, '', ''), (1, '', ''), (127, '', 'tmux unavailable')],
                        [(0, '', ''), (1, '', ''), (1, '', 'Permission denied')]):
            with self.subTest(outputs=outputs), self.assertRaises(training.RemoteDetached):
                self.follow(outputs)

    def test_stop_does_not_reclassify_already_oom_exited_task(self):
        stop = threading.Event(); stop.set()
        with self.assertRaises(training.RemoteTrainingFailed):
            self.follow([(0, 'CUDA out of memory', ''), (0, '1', '')], stop=stop)

    def test_explicit_stop_of_a_live_remote_task_sends_only_targeted_kill(self):
        stop = threading.Event(); stop.set()
        commands = []
        def capture(_, command):
            commands.append(command)
            if 'exit.code' in command:
                return 1, '', ''
            return 0, '', ''
        session = training.TrainingSession(id='specific-task', remote=True, remote_dir='$HOME/state/task')
        with patch.object(training, '_remote_capture', side_effect=capture), patch.object(training.time, 'sleep'):
            with self.assertRaises(TaskCancelled):
                training._follow_remote_session(Mock(), session, lambda _: None, stop)
        self.assertTrue(any('kill-session -t yolo-training-specific-task' in command for command in commands))
        self.assertFalse(any('rm ' in command for command in commands))

    def test_unconfirmed_kill_does_not_pretend_stopped(self):
        stop = threading.Event(); stop.set()
        with patch.object(training.time, 'sleep'), self.assertRaises(training.RemoteDetached):
            self.follow([(0, '', ''), (1, '', ''), (0, '', ''), (0, '', ''), (1, '', 'failed')], stop=stop)


class TrainingManagerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.storage = Path(self.temp.name) / 'training.json'
        self.manager = training.TrainingManager(self.storage)
        self.session = training.TrainingSession(id='oom-task', remote=True, status='disconnected',
            host='server-a', port=22, username='dell', remote_dir='$HOME/state/oom-task',
            model='yolo26n.pt', output='/runs/model_a', started_at='2026-11-09T09:00:00')
        self.manager._sessions[self.session.id] = self.session
        self.manager._save()
        self.parameters = {'remote_host':'server-a', 'remote_username':'dell', 'remote_password':'never-store-secret'}
        self.paramiko = SimpleNamespace(SSHClient=Mock, AutoAddPolicy=Mock)

    def tearDown(self):
        self.until(self.monitor_finished)
        self.temp.cleanup()

    def monitor_finished(self):
        # monitoring is cleared just before the final checkpoint, under this
        # lock. Wait for the checkpoint too before reading/deleting temp files.
        with self.manager._lock:
            return not self.session.monitoring

    def until(self, predicate):
        end = time.monotonic() + 3
        while time.monotonic() < end:
            if predicate():
                return
            time.sleep(.01)
        self.fail('monitor did not settle')

    def reconnect(self, follower=None, stop=False):
        finished = []
        with patch.object(training.remote_connection_manager, 'resolve_connection', return_value={'password':'never-store-secret'}), \
             patch.object(training.RemoteInferenceManager, '_load_paramiko', return_value=self.paramiko), \
             patch.object(training, '_follow_remote_session', side_effect=follower):
            if stop:
                self.manager.stop(self.session.id, parameters=self.parameters, on_finished=finished.append)
            else:
                self.manager.reconnect(self.session.id, self.parameters, on_finished=finished.append)
            self.until(self.monitor_finished)
            self.until(lambda: len(finished) == 1 if self.session.status in {'completed','failed','stopped'} else True)
        return finished

    def test_oom_reconnect_is_saved_as_failed_and_emits_terminal_history(self):
        def fail(*_):
            raise training.RemoteTrainingFailed('CUDA OOM，显存不足')
        finished = self.reconnect(fail)
        self.assertEqual(self.session.status, 'failed')
        self.assertIn('OOM', self.session.error)
        self.assertIsNotNone(self.session.finished_at)
        self.assertEqual(finished[0]['startedAt'], '2026-11-09T09:00:00')
        self.assertEqual(finished[0]['status'], 'failed')
        self.assertEqual(training.TrainingManager(self.storage).get(self.session.id).status, 'failed')
        self.assertNotIn('never-store-secret', self.storage.read_text())
        self.manager.delete(self.session.id)
        self.assertEqual(self.manager.list(), [])

    def test_network_failure_keeps_record_disconnected_and_not_finished(self):
        def drop(*_):
            raise OSError('SSH disconnected')
        self.assertEqual(self.reconnect(drop), [])
        self.assertEqual(self.session.status, 'disconnected')
        self.assertIsNone(self.session.finished_at)
        self.assertEqual(training.TrainingManager(self.storage).get(self.session.id).status, 'disconnected')

    def test_explicit_forget_needs_no_ssh_does_not_touch_remote_or_outputs(self):
        with patch.object(training.RemoteInferenceManager, '_load_paramiko') as ssh, \
             patch.object(training, '_remote_capture') as commands:
            with self.assertRaises(ValueError):
                self.manager.delete(self.session.id)
            self.manager.delete(self.session.id, forget=True)
        ssh.assert_not_called(); commands.assert_not_called()
        self.assertFalse(self.session.stop_event.is_set())
        self.assertEqual(training.TrainingManager(self.storage).list(), [])

    def test_forget_never_deletes_live_training_or_monitor(self):
        for status in ('running','starting','stopping'):
            self.session.status = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                self.manager.delete(self.session.id, forget=True)
        self.session.status, self.session.monitoring = 'disconnected', True
        with self.assertRaises(ValueError):
            self.manager.delete(self.session.id, forget=True)
        self.session.monitoring = False

    def test_delete_save_failure_restores_record(self):
        with patch.object(self.manager, '_save', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                self.manager.delete(self.session.id, forget=True)
        self.assertIs(self.manager.get(self.session.id), self.session)

    def test_connect_and_stop_has_confirmation_intent_but_detects_old_oom_first(self):
        def fail(_, session, report, event):
            self.assertTrue(event.is_set())
            raise training.RemoteTrainingFailed('OOM failure already happened')
        finished = self.reconnect(fail, stop=True)
        self.assertEqual(finished[0]['status'], 'failed')

    def test_connect_and_stop_live_session(self):
        def stop(_, session, report, event):
            self.assertTrue(event.is_set())
            raise TaskCancelled('stopped')
        finished = self.reconnect(stop, stop=True)
        self.assertEqual(finished[0]['status'], 'stopped')

    def test_reconnect_identity_and_duplicate_monitors_are_guarded(self):
        with self.assertRaisesRegex(ValueError, '服务器不一致'):
            self.manager.reconnect(self.session.id, {**self.parameters,'remote_host':'other'})
        entered, release = threading.Event(), threading.Event()
        def follower(*_):
            entered.set(); release.wait(2)
        with patch.object(training.remote_connection_manager, 'resolve_connection', return_value={'password':'fixture'}), \
             patch.object(training.RemoteInferenceManager, '_load_paramiko', return_value=self.paramiko), \
             patch.object(training, '_follow_remote_session', side_effect=follower):
            self.manager.reconnect(self.session.id, self.parameters)
            self.assertTrue(entered.wait(1))
            try:
                with self.assertRaisesRegex(ValueError, '重复重连'):
                    self.manager.reconnect(self.session.id, self.parameters)
            finally:
                release.set()
                self.until(self.monitor_finished)


class TrainingLifecycleApiTests(unittest.TestCase):
    def request(self, method, path, body=None, activated=True):
        from backend import server
        data = json.dumps(body or {}).encode()
        request = f'{method} {path} HTTP/1.0\r\nContent-Length: {len(data)}\r\n\r\n'.encode() + data
        output = bytearray()
        connection = SimpleNamespace(makefile=lambda *_:io.BytesIO(request), sendall=output.extend)
        with patch.object(server.license_store, 'status', return_value={'activated':activated,'role':'user','message':'required'}), \
             patch.object(server, 'yolo_training_manager') as manager:
            manager.stop.return_value = {'status':'stopping'}
            manager.reconnect.return_value = {'status':'running'}
            server.AppHandler(connection, ('127.0.0.1',1), SimpleNamespace())
        header, body = bytes(output).split(b'\r\n\r\n',1)
        return int(header.split()[1]), json.loads(body), manager

    def test_forget_endpoint_is_explicit_and_licensed(self):
        status, _, manager = self.request('DELETE','/api/yolo-training/sessions/old?forget=true')
        self.assertEqual(status,200); manager.delete.assert_called_once_with('old', forget=True)
        status, _, manager = self.request('DELETE','/api/yolo-training/sessions/old')
        self.assertEqual(status,200); manager.delete.assert_called_once_with('old')
        status, _, manager = self.request('DELETE','/api/yolo-training/sessions/old?forget=true', activated=False)
        self.assertEqual(status,403); manager.delete.assert_not_called()

    def test_stop_passes_credentials_only_when_supplied(self):
        status, _, manager = self.request('POST','/api/yolo-training/old/stop', {'parameters':{'remote_host':'a'}})
        self.assertEqual(status,200)
        self.assertEqual(manager.stop.call_args.kwargs['parameters'],{'remote_host':'a'})
        self.assertTrue(callable(manager.stop.call_args.kwargs['on_finished']))
        status, _, manager = self.request('POST','/api/yolo-training/old/stop')
        self.assertEqual(status,200); manager.stop.assert_called_once_with('old')
        status, _, manager = self.request('POST','/api/yolo-training/old/stop', {'parameters':[]})
        self.assertEqual(status,400); manager.stop.assert_not_called()


if __name__ == '__main__':
    unittest.main()
