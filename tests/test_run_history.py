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

    def test_keeps_five_records_for_each_function_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RunHistoryStore(Path(directory) / "run_history.json")
            for function_id in ("video_frames", "video_clip"):
                for number in range(7):
                    store.add({
                        "id": f"{function_id}-{number}",
                        "functionId": function_id,
                        "name": function_id,
                        "kind": "local",
                        "startedAt": "2026-09-22T10:00:00",
                    }, status="completed", message=f"第 {number} 次完成")

            all_records = store.list()
            self.assertEqual(len(all_records), 10)
            self.assertEqual(
                [record["id"] for record in all_records[:5]],
                [f"video_clip-{number}" for number in range(6, 1, -1)],
            )
            self.assertEqual(
                [record["id"] for record in store.list("video_frames")],
                [f"video_frames-{number}" for number in range(6, 1, -1)],
            )
            self.assertEqual(
                [record["id"] for record in store.list("video_clip")],
                [f"video_clip-{number}" for number in range(6, 1, -1)],
            )

    def test_deletes_only_the_requested_history_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run_history.json"
            store = RunHistoryStore(path)
            for record_id, function_id in (("frames-1", "video_frames"), ("clip-1", "video_clip")):
                store.add({
                    "id": record_id,
                    "functionId": function_id,
                    "name": function_id,
                    "kind": "local",
                    "startedAt": "2026-09-22T10:00:00",
                }, status="completed", message="完成")

            store.delete("frames-1")

            self.assertEqual(store.list("video_frames"), [])
            self.assertEqual([record["id"] for record in store.list("video_clip")], ["clip-1"])
            self.assertNotIn("frames-1", path.read_text(encoding="utf-8"))
            with self.assertRaisesRegex(ValueError, "不存在"):
                store.delete("missing")

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
