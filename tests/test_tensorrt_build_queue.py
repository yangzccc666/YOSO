from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.task_manager import PlatformTaskManager
from backend.tensorrt_build_queue import TensorRTBuildQueue


class TensorRTQueueTests(unittest.TestCase):
    def setUp(self):
        self.tasks = PlatformTaskManager()
        self.queues = []
        self.events = []
        self.parameters = {"host": "box-a", "port": 22, "username": "wel", "password": "secret-A",
                           "onnx_files": "/models/a.onnx", "build_mode": "FP16"}

    def tearDown(self):
        for queue in self.queues:
            queue.close()
        for event in self.events:
            event.set()
        for queue in self.queues:
            if queue._worker:
                queue._worker.join(3)

    def event(self):
        event = threading.Event()
        self.events.append(event)
        return event

    def queue(self, runner, **kwargs):
        def resolve(parameters):
            return {key: parameters[key] for key in ("host", "port", "username", "password")}
        queue = TensorRTBuildQueue(self.tasks, runner=runner, resolve_connection=resolve, **kwargs)
        self.queues.append(queue)
        return queue

    def submit(self, queue, paths="/models/a.onnx", **parameters):
        return queue.enqueue("remote_tensorrt_build", "TensorRT", {}, {**self.parameters, "onnx_files": paths, **parameters})

    def until(self, predicate):
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.01)
        self.fail("Timed out waiting for queue state")

    def test_enqueue_while_running_fifo_config_snapshot_and_no_password_in_api(self):
        release, entered = self.event(), self.event()
        seen = []
        def runner(context):
            seen.append(dict(context.parameters))
            context.report("building " + context.parameters["onnx_files"])
            if len(seen) == 1:
                entered.set()
                release.wait(3)
            return {"message": "done", "outputFiles": [context.parameters["onnx_files"].replace(".onnx", ".plan")]}
        finished = []
        queue = self.queue(runner, on_finished=lambda task, result: finished.append(result))
        self.submit(queue)
        self.assertTrue(entered.wait(3))
        raw = {**self.parameters, "onnx_files": "/models/b.onnx\n/models/c.onnx", "password": "secret-B", "build_mode": "FP32"}
        submitted = queue.enqueue("remote_tensorrt_build", "TensorRT", {}, raw)
        raw.update({"password": "changed", "build_mode": "FP16", "host": "new-host"})
        self.assertEqual(submitted["queuedCount"], 2)
        self.assertNotIn("secret", json.dumps(submitted))
        self.assertNotIn("password", json.dumps(submitted))
        release.set()
        self.until(lambda: len(finished) == 3)
        self.assertEqual([row["onnx_files"] for row in seen], ["/models/a.onnx", "/models/b.onnx", "/models/c.onnx"])
        self.assertEqual([row["build_mode"] for row in seen], ["FP16", "FP32", "FP32"])
        self.assertEqual([row["password"] for row in seen], ["secret-A", "secret-B", "secret-B"])
        self.assertTrue(all(not item.parameters for item in queue._items))
        self.assertEqual(self.tasks.active(), [])

    def test_duplicate_active_or_queued_paths_are_skipped_but_other_device_allowed(self):
        entered, release = self.event(), self.event()
        def runner(_):
            entered.set()
            release.wait(3)
            return {}
        queue = self.queue(runner)
        first = self.submit(queue, "/models/a.onnx\n/models/a.onnx")
        self.assertTrue(entered.wait(3))
        self.assertEqual(len(first["addedIds"]), 1)
        second = self.submit(queue, "/models/a.onnx\n/models/b.onnx")
        self.assertEqual(second["skipped"], ["/models/a.onnx"])
        self.assertEqual(second["queuedCount"], 1)
        third = self.submit(queue, "/models/a.onnx", host="box-b")
        self.assertEqual(third["queuedCount"], 2)

    def test_enqueue_resumes_paused_queue_and_keeps_fifo(self):
        order = []
        queue = self.queue(lambda context: (order.append(context.parameters['onnx_files']) or {}))
        with queue._condition:
            self.submit(queue)
            queue.set_paused(True)
        result = self.submit(queue, '/models/b.onnx')
        self.assertFalse(result['paused'])
        self.assertIn('已开启自动执行', result['message'])
        self.until(lambda: len(order) == 2)
        self.assertEqual(order, ['/models/a.onnx', '/models/b.onnx'])

    def test_resubmitting_waiting_path_resumes_without_duplicate(self):
        seen = []
        queue = self.queue(lambda context: (seen.append(dict(context.parameters)) or {}))
        with queue._condition:
            self.submit(queue)
            queue.set_paused(True)
        result = self.submit(queue, password='new-password')
        self.assertEqual(result['addedIds'], [])
        self.assertEqual(result['skipped'], ['/models/a.onnx'])
        self.assertFalse(result['paused'])
        self.until(lambda: len(seen) == 1)
        self.assertEqual(seen[0]['password'], 'new-password')

    def test_enqueue_after_failure_resumes_existing_waiting_items_first(self):
        order = []
        def runner(context):
            order.append(context.parameters['onnx_files'])
            if len(order) == 1:
                raise RuntimeError('build failed')
            return {}
        queue = self.queue(runner)
        self.submit(queue, '/models/a.onnx\n/models/b.onnx')
        self.until(lambda: queue.snapshot()['paused'])
        self.submit(queue, '/models/c.onnx')
        self.until(lambda: len(order) == 3)
        self.assertEqual(order, ['/models/a.onnx', '/models/b.onnx', '/models/c.onnx'])

    def test_pause_allows_current_to_finish_then_resume(self):
        entered, release = self.event(), self.event()
        order = []
        def runner(context):
            order.append(context.parameters["onnx_files"])
            if len(order) == 1:
                entered.set(); release.wait(3)
            return {}
        queue = self.queue(runner)
        self.submit(queue, "/models/a.onnx\n/models/b.onnx")
        self.assertTrue(entered.wait(3))
        queue.set_paused(True)
        release.set()
        self.until(lambda: queue.snapshot()["runningCount"] == 0)
        self.assertEqual(order, ["/models/a.onnx"])
        self.assertEqual(queue.snapshot()["queuedCount"], 1)
        queue.set_paused(False)
        self.until(lambda: len(order) == 2)

    def test_failure_pauses_next_until_explicit_resume(self):
        order = []
        def runner(context):
            order.append(context.parameters["onnx_files"])
            if len(order) == 1:
                raise RuntimeError("remote connection lost; remote state unknown")
            return {}
        queue = self.queue(runner)
        self.submit(queue, "/models/a.onnx\n/models/b.onnx")
        self.until(lambda: queue.snapshot()["paused"])
        self.assertEqual(len(order), 1)
        self.assertEqual(queue.snapshot()["items"][0]["status"], "failed")
        queue.set_paused(False)
        self.until(lambda: len(order) == 2)

    def test_manual_stop_via_menu_also_pauses_remaining(self):
        entered = self.event()
        def runner(context):
            entered.set()
            while not context.stop_event.wait(.02):
                pass
            context.check_cancelled()
        queue = self.queue(runner)
        self.submit(queue, "/models/a.onnx\n/models/b.onnx")
        self.assertTrue(entered.wait(3))
        self.tasks.stop(queue.snapshot()["items"][0]["taskId"])
        self.until(lambda: queue.snapshot()["paused"])
        self.assertEqual(queue.snapshot()["items"][0]["status"], "stopped")
        self.assertEqual(queue.snapshot()["queuedCount"], 1)

    def test_remove_or_clear_waiting_never_stops_current(self):
        entered, release = self.event(), self.event()
        def runner(context):
            entered.set(); release.wait(3)
            self.assertFalse(context.stop_event.is_set())
            return {}
        queue = self.queue(runner)
        result = self.submit(queue, "/models/a.onnx\n/models/b.onnx\n/models/c.onnx")
        self.assertTrue(entered.wait(3))
        with self.assertRaisesRegex(ValueError, "不能移除"):
            queue.remove(result["addedIds"][0])
        queue.remove(result["addedIds"][1])
        self.assertEqual(queue.snapshot()["queuedCount"], 1)
        queue.clear_waiting()
        self.assertEqual(queue.snapshot()["queuedCount"], 0)
        self.assertEqual(queue.snapshot()["runningCount"], 1)
        release.set()

    def test_explicit_stop_button_pauses_and_stops_active(self):
        entered = self.event()
        def runner(context):
            entered.set()
            context.stop_event.wait(3)
            context.check_cancelled()
        queue = self.queue(runner)
        result = self.submit(queue)
        self.assertTrue(entered.wait(3))
        queue.stop(result["addedIds"][0])
        self.until(lambda: queue.snapshot()["items"][0]["status"] == "stopped")
        self.assertTrue(queue.snapshot()["paused"])

    def test_existing_environment_check_slot_is_waited_not_failed(self):
        external = self.tasks.start("remote_tensorrt_build", "environment check", "remote-build")
        entered = self.event()
        queue = self.queue(lambda _: (entered.set() or {}))
        self.submit(queue)
        self.assertFalse(entered.wait(.1))
        self.assertEqual(queue.snapshot()["queuedCount"], 1)
        self.tasks.finish(external.id)
        self.assertTrue(entered.wait(3))
        self.assertFalse(queue.snapshot()["paused"])

    def test_close_never_kills_started_remote_or_starts_waiting_job(self):
        entered, release = self.event(), self.event()
        order = []
        def runner(context):
            order.append(context.parameters["onnx_files"])
            entered.set(); release.wait(3)
            self.assertFalse(context.stop_event.is_set())
            return {}
        queue = self.queue(runner)
        self.submit(queue, "/models/a.onnx\n/models/b.onnx")
        self.assertTrue(entered.wait(3))
        queue.close()
        self.assertFalse(any(task["status"] == "stopping" for task in self.tasks.active()))
        self.assertFalse(queue._items[1].parameters.get("password"))
        self.assertEqual(queue._items[1].parameters["onnx_files"], "/models/b.onnx")
        release.set()
        queue._worker.join(3)
        self.assertEqual(order, ["/models/a.onnx"])
        with self.assertRaisesRegex(ValueError, "关闭"):
            self.submit(queue)

    def test_pending_not_subject_to_five_finished_limit(self):
        gate, entered = self.event(), self.event()
        count = 0
        def runner(_):
            nonlocal count
            count += 1
            if count == 6:
                entered.set(); gate.wait(3)
            return {}
        queue = self.queue(runner, on_finished=lambda *_: True)
        self.submit(queue, "\n".join(f"/models/m{i}.onnx" for i in range(16)))
        self.assertTrue(entered.wait(3))
        queue.set_paused(True)
        gate.set()
        self.until(lambda: queue.snapshot()["runningCount"] == 0)
        snapshot = queue.snapshot()
        self.assertEqual(snapshot["queuedCount"], 10)
        self.assertEqual(sum(item["status"] == "completed" for item in snapshot["items"]), 0)

    def test_success_is_removed_after_history_saved_with_outputs_logs_and_timing(self):
        from backend import server
        from backend.run_history import RunHistoryStore
        with tempfile.TemporaryDirectory() as scratch:
            store = RunHistoryStore(Path(scratch) / 'history.json')
            def runner(context):
                context.report('built engine successfully')
                return {'message': 'done', 'outputFiles': [context.parameters['onnx_files'].replace('.onnx', '.plan')]}
            with patch.object(server, 'run_history', store), patch.object(server, 'notify_system') as notify:
                queue = self.queue(runner, on_finished=server.record_trt_queue_run, storage=Path(scratch) / 'queue.json')
                self.submit(queue, '/models/a.onnx\n/models/b.onnx')
                self.until(lambda: len(store.list()) == 2 and not queue.snapshot()['items'] and not self.tasks.active())
                records = store.list()
                self.assertEqual({row['details']['ONNX 模型'] for row in records}, {'/models/a.onnx', '/models/b.onnx'})
                self.assertTrue(all(row['outputs'] and 'built engine successfully' in row['logs'] for row in records))
                self.assertTrue(all(row['elapsedSeconds'] is not None for row in records))
                self.assertEqual(notify.call_count, 2)
                self.assertEqual(json.loads((Path(scratch) / 'queue.json').read_text())['items'], [])
                queue.close()
                queue._worker.join(3)

    def test_failed_history_write_preserves_success_and_continues_next_task(self):
        from backend import server
        with patch.object(server.run_history, 'add', side_effect=OSError('disk full')), patch.object(server, 'notify_system'):
            queue = self.queue(lambda _: {'message': 'build succeeded'}, on_finished=server.record_trt_queue_run)
            self.submit(queue, '/models/a.onnx\n/models/b.onnx')
            self.until(lambda: len(queue.snapshot()['items']) == 2 and
                       all('运行历史未成功保存' in item['message'] for item in queue.snapshot()['items']))
            self.assertTrue(all(item['status'] == 'completed' for item in queue.snapshot()['items']))
            self.assertFalse(queue.snapshot()['paused'])

    def test_raised_history_callback_preserves_success(self):
        def broken(*_):
            raise OSError('history unavailable')
        queue = self.queue(lambda _: {}, on_finished=broken)
        self.submit(queue)
        self.until(lambda: any('history unavailable' in item['message'] for item in queue.snapshot()['items']))
        self.assertEqual(queue.snapshot()['items'][0]['status'], 'completed')

    def test_legacy_success_is_cleaned_only_when_history_exists_and_persists_cleanup(self):
        with tempfile.TemporaryDirectory() as scratch:
            storage = Path(scratch) / 'queue.json'
            # No history sink: retain the completed result for recovery.
            queue = self.queue(lambda _: {}, storage=storage)
            self.submit(queue)
            self.until(lambda: not self.tasks.active() and queue.snapshot()['items'][0]['status'] == 'completed')
            task_id = queue.snapshot()['items'][0]['taskId']
            queue.close()
            queue._worker.join(3)
            archived = set()
            restored = self.queue(lambda _: {}, storage=storage, history_contains=lambda key: key in archived)
            restored.start_pending()
            self.assertEqual(len(restored.snapshot()['items']), 1)
            archived.add(task_id)
            restored.start_pending()
            self.assertEqual(restored.snapshot()['items'], [])
            self.assertEqual(json.loads(storage.read_text())['items'], [])
            queue.close()  # A stale, already closed queue must not resurrect the card.
            self.assertEqual(json.loads(storage.read_text())['items'], [])
            restored.close()

    def test_failed_records_remain_in_queue(self):
        def fail(_):
            raise RuntimeError('build failed')
        queue = self.queue(fail, on_finished=lambda *_: True)
        self.submit(queue)
        self.until(lambda: queue.snapshot()['items'][0]['status'] == 'failed' and not self.tasks.active())
        self.assertEqual(len(queue.snapshot()['items']), 1)

    def test_validation_and_capacity_are_atomic(self):
        queue = self.queue(lambda _: {})
        queue.MAX_PENDING = 2
        queue.set_paused(True)
        with self.assertRaisesRegex(ValueError, "最多"):
            self.submit(queue, "/a.onnx\n/b.onnx\n/c.onnx")
        self.assertEqual(queue.snapshot()["queuedCount"], 0)
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            self.submit(queue, "relative.onnx")
        self.assertEqual(queue.snapshot()["queuedCount"], 0)


class TensorRTQueueApiTests(unittest.TestCase):
    def request(self, method, path, payload=None, activated=True, **patches):
        from backend import server
        data = json.dumps(payload or {}).encode()
        raw = f"{method} {path} HTTP/1.0\r\nContent-Length: {len(data)}\r\n\r\n".encode() + data
        output = bytearray()
        connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
        with patch.object(server.license_store, "status", return_value={"activated": activated, "role": "user", "message": "activation required"}), \
             patch.object(server, "trt_build_queue") as queue:
            queue.snapshot.return_value = {"items": []}
            queue.enqueue.return_value = {"message": "queued", "items": [], "addedIds": ["1"]}
            queue.set_paused.return_value = {"paused": True}
            queue.remove.return_value = {"items": []}
            queue.clear_waiting.return_value = {"items": []}
            queue.stop.return_value = {"paused": True}
            queue.reconnect.return_value = {"runningCount": 1}
            queue.discover.return_value = {"message": "found", "items": []}
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
            if not activated:
                queue.snapshot.assert_not_called()
                queue.enqueue.assert_not_called()
                queue.start_pending.assert_not_called()
            elif method == 'GET' and path in {'/api/tasks/active', '/api/tensorrt-build/queue'}:
                queue.start_pending.assert_called_once()
        header, body = bytes(output).split(b"\r\n\r\n", 1)
        return int(header.split()[1]), json.loads(body)

    def test_routes_and_license(self):
        self.assertEqual(self.request("GET", "/api/tensorrt-build/queue", activated=False)[0], 403)
        self.assertEqual(self.request("GET", "/api/tasks/active", activated=False)[0], 403)
        self.assertEqual(self.request("GET", "/api/tasks/active")[0], 200)
        self.assertEqual(self.request("GET", "/api/tensorrt-build/queue"), (200, {"items": []}))
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue", {"parameters": {"onnx_files": "/a.onnx"}})[0], 202)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/pause", {"paused": True})[0], 200)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/clear")[0], 200)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/a/stop")[0], 200)
        self.assertEqual(self.request("DELETE", "/api/tensorrt-build/queue/a")[0], 200)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/a/reconnect", {"parameters": {}}), (200, {"runningCount": 1}))
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/discover", {"parameters": {}})[0], 200)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/a/reconnect", {"parameters": []})[0], 400)
        self.assertEqual(self.request("POST", "/api/tensorrt-build/queue/discover", {"parameters": []})[0], 400)
        self.assertEqual(self.request("DELETE", "/api/tensorrt-build/queue/a?forget=true")[0], 200)

    def test_legacy_run_endpoint_also_submits_instead_of_blocking(self):
        status, response = self.request("POST", "/api/functions/remote_tensorrt_build/run", {"parameters": {"onnx_files": "/a.onnx"}})
        self.assertEqual(status, 202)
        self.assertTrue(response["result"]["queued"])


if __name__ == "__main__":
    unittest.main()
