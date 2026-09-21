from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import server
from backend.run_history import RunHistoryStore
from backend.task_manager import PlatformTaskManager


class RunHistoryTests(unittest.TestCase):
    def test_persists_only_five_newest_finished_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_history.json"
            store = RunHistoryStore(path)
            for number in range(6):
                store.add({
                    "id": str(number), "functionId": "video_frames", "name": "视频抽帧",
                    "kind": "local", "startedAt": "2026-09-17T10:00:00",
                }, status="completed", message=f"第 {number} 次完成", logs=["已处理视频"], details={"视频": f"video-{number}.mp4"})
            records = RunHistoryStore(path).list()
            self.assertEqual([record["id"] for record in records], ["5", "4", "3", "2", "1"])
            self.assertEqual(records[0]["details"], {"视频": "video-5.mp4"})
            self.assertEqual(len(records[0]["logs"]), 1)
            with self.assertRaisesRegex(ValueError, "已结束"):
                store.add({"id": "7"}, status="running", message="")

    def test_async_completion_records_history_and_releases_task(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(Path(directory) / "run_history.json")
            manager = PlatformTaskManager()
            task = manager.start("yolo_dataset_split", "YOLO 模型训练", "remote")
            notifications: list[tuple[str, str]] = []
            server.set_system_notifier(lambda title, message: notifications.append((title, message)))
            try:
                with patch.object(server, "run_history", store), patch.object(server, "platform_task_manager", manager):
                    server.finish_async_run(task, {
                        "status": "completed", "message": "训练完成", "logs": ["Epoch 1/2"],
                        "result": {"project": "/data/runs", "name": "light_change"},
                    }, {"数据配置": "/data/data.yaml"})
            finally:
                server.set_system_notifier(None)
            self.assertEqual(manager.active(), [])
            record = store.list()[0]
            self.assertEqual(record["outputs"], ["/data/runs/light_change"])
            self.assertEqual(record["logs"], ["Epoch 1/2"])
            self.assertEqual(record["status"], "completed")
            self.assertEqual(notifications, [("YOLO 模型训练 已完成", "训练完成")])


if __name__ == "__main__":
    unittest.main()
