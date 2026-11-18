from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend import server, yolo_training
from backend.run_history import RunHistoryStore
from backend.run_timing import elapsed_seconds, valid_elapsed
from backend.task_manager import ActiveTask, PlatformTaskManager


class RunTimingTests(unittest.TestCase):
    def test_timestamp_duration_supports_offsets_legacy_and_subseconds(self):
        self.assertEqual(elapsed_seconds("2026-11-06T12:00:00.100+08:00", "2026-11-06T04:01:01.350Z"), 61.25)
        self.assertEqual(elapsed_seconds("2026-11-06T12:00:00", "2026-11-07T13:02:03"), 90123)
        self.assertEqual(elapsed_seconds("2026-11-06T12:00:00", "2026-11-06T12:00:00"), 0)

    def test_unknown_invalid_or_backwards_time_is_not_invented(self):
        for start, end in (("", ""), (None, "2026-11-06"), ("bad", "bad"),
                           ("2026-11-07", "2026-11-06")):
            self.assertIsNone(elapsed_seconds(start, end))
        for value in (None, True, "30", -1, float("nan"), float("inf")):
            self.assertIsNone(valid_elapsed(value))

    def test_live_task_uses_monotonic_clock_even_if_wall_clock_changes(self):
        with patch("backend.task_manager.time.monotonic", return_value=100):
            task = PlatformTaskManager().start("video_frames", "抽帧")
        task.started_at = "2099-01-01T00:00:00"  # Simulate an adjusted system clock.
        with patch("backend.task_manager.time.monotonic", return_value=102.375):
            self.assertEqual(task.snapshot()["elapsedSeconds"], 2.375)

    def test_restored_task_uses_original_start_instead_of_reconnect_time(self):
        task = ActiveTask("restored", "yolo_model_training", "训练", "remote-training", "restored",
                          started_at=datetime.now().astimezone().replace(year=2025).isoformat(),
                          started_monotonic=None)
        self.assertGreater(task.snapshot()["elapsedSeconds"], 300 * 86400)

    def test_history_persists_frozen_duration_for_every_execution_kind_and_outcome(self):
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(Path(directory) / "history.json")
            for kind in ("local", "remote", "remote-build", "remote-training"):
                for status in ("completed", "failed", "stopped"):
                    entry = store.add({"id": f"{kind}-{status}", "functionId": kind, "name": kind,
                                       "kind": kind, "startedAt": "2026-11-06T12:00:00+08:00",
                                       "elapsedSeconds": 0.375}, status=status, message="test")
                    self.assertEqual(entry["elapsedSeconds"], 0.375)
            reloaded = RunHistoryStore(store.path).list()
            self.assertEqual(len(reloaded), 12)
            self.assertTrue(all(record["elapsedSeconds"] == 0.375 for record in reloaded))
            self.assertTrue(all(record["finishedAt"] for record in reloaded))
            self.assertTrue(all(record["elapsedSeconds"] == .375 for record in json.loads(store.path.read_text())))

    def test_old_history_is_derived_without_rewriting_or_deleting_records(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.json"
            old = [{"id": "old", "functionId": "video_frames", "status": "completed",
                    "startedAt": "2026-11-06T12:00:00", "finishedAt": "2026-11-06T12:02:05"},
                   {"id": "unknown", "functionId": "video_clip", "startedAt": "", "finishedAt": ""}]
            text = json.dumps(old)
            path.write_text(text)
            records = RunHistoryStore(path).list()
            self.assertEqual(records[0]["elapsedSeconds"], 125)
            self.assertIsNone(records[1]["elapsedSeconds"])
            self.assertEqual(path.read_text(), text)

    def test_live_history_includes_local_and_remote_functions_without_counting_against_finished_limit(self):
        manager = PlatformTaskManager()
        for number in range(8):
            manager.start(f"function-{number}", "任务", "local" if number % 2 else "remote")
        manager.start("yolo_model_training", "训练", "remote-training")
        training = {"id": "active-training-1", "status": "running"}
        with patch.object(server, "platform_task_manager", manager), \
             patch.object(server, "active_training_history_records", return_value=[training]):
            records = server.active_run_history_records()
        self.assertEqual(len(records), 9)
        self.assertEqual(records[0], training)
        self.assertTrue(all(record["elapsedSeconds"] >= 0 for record in records[1:]))

    def test_disconnect_is_not_saved_as_completed_training(self):
        manager = PlatformTaskManager()
        task = manager.start("yolo_model_training", "训练", "remote-training")
        with patch.object(server, "platform_task_manager", manager), patch.object(server, "record_finished_run") as record:
            server.finish_async_run(task, {"status": "disconnected"}, {})
            record.assert_not_called()
        self.assertEqual(manager.active(), [])

    def test_reconnected_training_emits_finish_callback_with_original_start(self):
        with tempfile.TemporaryDirectory() as directory:
            manager = yolo_training.TrainingManager(Path(directory) / "sessions.json")
            session = yolo_training.TrainingSession(id="reconnect", remote=True, host="test-device",
                                                   username="tester", remote_dir="/tmp/task",
                                                   status="disconnected", started_at="2026-11-05T10:00:00")
            manager._sessions[session.id] = session
            finished, result = threading.Event(), []
            def done(snapshot):
                result.append(snapshot)
                finished.set()
            def resolve(_):
                return {"password": "test-only"}
            fake_paramiko = SimpleNamespace(SSHClient=Mock, AutoAddPolicy=Mock)
            with patch.object(yolo_training.remote_connection_manager, "resolve_connection", side_effect=resolve), \
                 patch.object(yolo_training.RemoteInferenceManager, "_load_paramiko", return_value=fake_paramiko), \
                 patch.object(yolo_training, "_follow_remote_session", return_value=None):
                manager.reconnect(session.id, {"remote_host": "test-device", "remote_username": "tester"}, on_finished=done)
                self.assertTrue(finished.wait(3))
            self.assertEqual(result[0]["startedAt"], "2026-11-05T10:00:00")
            self.assertEqual(result[0]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
