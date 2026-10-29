from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from backend import bundled_video_frames, server
from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.task_manager import PlatformTaskManager, TaskCancelled
from backend.video_frame_resources import VideoFrameResourcePool
from backend.video_frames import _resolve_videos, normalize_video_folder_paths, run_video_frames


class VideoMultiFolderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        resources = {"cpuCount": 8, "cpuPercent": 5, "availableMemoryMb": 16384, "maxWorkers": 4, "recommendedWorkers": 2, "warning": ""}
        self.resources = patch("backend.video_frame_resources.machine_resources", return_value=resources)
        self.resources.start()
        self.addCleanup(self.resources.stop)
        self.pool_patch = patch("backend.video_frames.resource_pool", VideoFrameResourcePool())
        self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)

    def video(self, relative):
        source = self.root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
        writer.release()
        return source

    def context(self, folders, *, source=None, output=True):
        paths = {"video_folder": folders[0]} if folders else {}
        if source:
            paths["video_file"] = source
        if output:
            paths["output_folder"] = self.root / "output"
        messages = []
        return RunContext("video_frames", paths, {"_video_folder_paths": [str(path) for path in folders], "parallel_workers": 0, "target_fps": 2}, messages.append), messages

    def test_multiline_quotes_spaces_and_normalized_duplicate_paths(self):
        first = self.root / "工位 1"
        second = self.root / "工位2"
        raw = f'  "{second}"\r\n\n\'{first}\'\n{first}/.\n'
        self.assertEqual(normalize_video_folder_paths(raw), sorted([first.resolve(), second.resolve()], key=lambda path: str(path).casefold()))
        self.assertEqual(normalize_video_folder_paths([str(first), str(first / ".")]), [first.resolve()])
        self.assertEqual(normalize_video_folder_paths(""), [])
        for invalid in ({}, [3], 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_video_folder_paths(invalid)

    def test_scan_parent_child_same_folder_and_single_video_once(self):
        first = self.video("A/one.MP4")
        second = self.video("A/nested/deep/two.mp4")
        third = self.video("B/three.mp4")
        context, messages = self.context([self.root / "A/nested", self.root / "A", self.root / "B", self.root / "A"], source=first)
        videos = _resolve_videos(context, bundled_video_frames)
        self.assertEqual(set(videos), {first, second, third})
        self.assertEqual(len(videos), 3)
        self.assertTrue(any("跳过重复扫描" in message for message in messages))

    def test_actual_multi_folder_extraction_shared_output_and_automatic_workers(self):
        self.video("A/deep/工位1.mp4")
        self.video("B/工位1.mp4")
        context, _ = self.context([self.root / "A", self.root / "B"])
        result = run_video_frames(context)
        self.assertEqual(result["videoCount"], 2)
        self.assertEqual(result["imageCount"], 4)
        self.assertEqual(result["parallelWorkers"], 2)
        self.assertEqual(len(set(result["outputFolders"])), 2)
        self.assertTrue(all(Path(folder).parent == self.root / "output" for folder in result["outputFolders"]))

    def test_empty_output_saves_beside_each_source(self):
        self.video("A/one.mp4")
        self.video("B/deep/two.mp4")
        context, _ = self.context([self.root / "A", self.root / "B"], output=False)
        result = run_video_frames(context)
        for detail in result["details"]:
            self.assertEqual(Path(detail["output"]).parent, Path(detail["video"]).parent)
            self.assertEqual(len(list(Path(detail["output"]).glob("*.jpg"))), 2)

    def test_invalid_folder_is_reported_but_valid_folders_continue(self):
        self.video("valid/one.mp4")
        context, messages = self.context([self.root / "missing", self.root / "valid"])
        result = run_video_frames(context)
        self.assertEqual(result["videoCount"], 1)
        self.assertTrue(any("路径不存在或不是文件夹" in message for message in messages))
        context, _ = self.context([self.root / "missing"])
        with self.assertRaisesRegex(ValueError, "没有找到可处理"):
            run_video_frames(context)

    def test_scanning_can_be_cancelled(self):
        folder = self.root / "input"
        folder.mkdir()
        stop = threading.Event()
        context = RunContext("video_frames", {"video_folder": folder}, {}, lambda _: None, stop)
        def walking(*args, **kwargs):
            yield str(folder), [], []
            stop.set()
            yield str(folder / "nested"), [], []
        with patch("backend.bundled_video_frames.os.walk", side_effect=walking), self.assertRaises(TaskCancelled):
            _resolve_videos(context, bundled_video_frames)

    def test_catalog_upgrades_label_preserving_other_fields(self):
        storage = self.root / "functions.json"
        item = FunctionCatalog(storage).get("video_frames")
        item["name"] = "自用抽帧"
        item["pathFields"][0]["label"] = "视频文件夹（递归扫描子文件夹）"
        storage.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
        for _ in range(2):
            migrated = FunctionCatalog(storage).get("video_frames")
            self.assertEqual(migrated["name"], "自用抽帧")
            self.assertIn("每行一个", migrated["pathFields"][0]["label"])
            self.assertEqual(len(migrated["pathFields"]), 3)

    def test_api_normalizes_folder_list_and_blocks_reordered_duplicate_batch(self):
        manager = PlatformTaskManager()
        release = threading.Event()
        finished = threading.Event()
        captured = []
        def execute(_handler, context):
            captured.append(context)
            release.wait(5)
            return {"message": "done", "outputFolders": []}
        def request(folders):
            payload = {"paths": {"video_folder": folders}, "parameters": {"parallel_workers": 0}}
            body = json.dumps(payload).encode()
            raw = f"POST /api/functions/video_frames/run HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
            head, result = bytes(output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(result)
        with patch.object(server, "platform_task_manager", manager), patch.object(server, "execute", side_effect=execute), \
             patch.object(server, "record_finished_run", side_effect=lambda *args, **kwargs: finished.set()), \
             patch.object(server.license_store, "status", return_value={"activated": True, "role": "user"}):
            try:
                status, result = request(f"{self.root}/B\n{self.root}/A\n{self.root}/A")
                self.assertEqual(status, 202)
                self.assertIn("2 个文件夹", result["task"]["name"])
                status, _ = request(f"{self.root}/A\n{self.root}/B")
                self.assertEqual(status, 400)
                self.assertEqual(captured[0].parameters["_video_folder_paths"], [str(self.root / "A"), str(self.root / "B")])
                self.assertEqual(captured[0].paths["video_folder"], self.root / "A")
            finally:
                release.set()
                self.assertTrue(finished.wait(3))
