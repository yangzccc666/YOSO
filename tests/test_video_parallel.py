from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from backend import bundled_video_frames, server
from backend.handlers import RunContext
from backend.task_manager import PlatformTaskManager, TaskCancelled
from backend.video_frame_resources import VideoFrameResourcePool
from backend.video_frames import _unique_output_folder, run_video_frames


SAFE_RESOURCES = {"cpuCount": 8, "cpuPercent": 5, "availableMemoryMb": 16384,
                  "maxWorkers": 4, "recommendedWorkers": 2, "warning": ""}


class VideoParallelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.pool = VideoFrameResourcePool()
        self.resources = patch("backend.video_frame_resources.machine_resources", side_effect=lambda: dict(SAFE_RESOURCES))
        self.resources.start()
        self.pool_patch = patch("backend.video_frames.resource_pool", self.pool)
        self.pool_patch.start()

    def tearDown(self) -> None:
        self.pool_patch.stop()
        self.resources.stop()
        self.temporary.cleanup()

    def video(self, name: str) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        return path

    def context(self, source: Path, scene: str = "scene", stop: threading.Event | None = None, folder: bool = False, workers: int = 2) -> RunContext:
        return RunContext("video_frames", {"video_folder" if folder else "video_file": source, "output_folder": self.root / "output"},
                          {"project_code": "tcl", "scene_code": scene, "parallel_workers": workers, "target_fps": 2}, lambda _: None, stop)

    def test_nested_videos_actually_overlap_and_have_separate_outputs(self) -> None:
        self.video("input/one/sample.mp4")
        self.video("input/two/deep/sample.mp4")
        barrier = threading.Barrier(2)
        lock = threading.Lock()
        active = peak = 0
        original = bundled_video_frames.vedio2img
        original_progress = bundled_video_frames.tqdm
        def tracked(**arguments):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                barrier.wait(timeout=5)
                return original(**arguments)
            finally:
                with lock:
                    active -= 1
        with patch.object(bundled_video_frames, "vedio2img", side_effect=tracked):
            result = run_video_frames(self.context(self.root / "input", folder=True))
        self.assertEqual(peak, 2)
        self.assertEqual(result["imageCount"], 4)
        self.assertEqual(len(set(result["outputFolders"])), 2)
        self.assertIs(bundled_video_frames.tqdm, original_progress)
        self.assertEqual(self.pool.status()["activeWorkers"], 0)

    def test_directory_reservation_is_atomic_between_batches(self) -> None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            paths = list(executor.map(lambda _: _unique_output_folder(self.root, "same"), range(24)))
        self.assertEqual(len(set(paths)), 24)
        self.assertTrue(all(path.is_dir() for path in paths))

    def test_cancel_one_batch_preserves_its_images_and_other_batch(self) -> None:
        source = self.video("input.mp4")
        stop = threading.Event()
        started_a, started_b, finish_b = threading.Event(), threading.Event(), threading.Event()
        def controlled(**arguments):
            output = Path(arguments["images_save_path"])
            (output / "saved.jpg").write_bytes(b"saved-image")
            is_a = "cancel" in arguments["filename_prefix"]
            (started_a if is_a else started_b).set()
            while not finish_b.wait(0.01):
                arguments["check_cancelled"]()
            arguments["check_cancelled"]()
            return {"selected_frame_indices": []}
        with patch.object(bundled_video_frames, "vedio2img", side_effect=controlled):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(run_video_frames, self.context(source, "cancel", stop, workers=1))
                second = executor.submit(run_video_frames, self.context(source, "keep", workers=1))
                try:
                    self.assertTrue(started_a.wait(5))
                    self.assertTrue(started_b.wait(5))
                    stop.set()
                    with self.assertRaises(TaskCancelled):
                        first.result(timeout=5)
                    self.assertFalse(second.done())
                    finish_b.set()
                    self.assertEqual(second.result(timeout=5)["imageCount"], 1)
                finally:
                    stop.set()
                    finish_b.set()
        self.assertEqual(len(list((self.root / "output").rglob("saved.jpg"))), 2)
        self.assertEqual(self.pool.status()["activeWorkers"], 0)

    def test_low_memory_is_reported_before_decoding(self) -> None:
        source = self.video("input.mp4")
        with patch.object(self.pool, "status", return_value={**SAFE_RESOURCES, "maxWorkers": 0, "warning": "可用内存不足"}):
            with patch.object(bundled_video_frames, "vedio2img") as decode:
                with self.assertRaisesRegex(ValueError, "内存不足"):
                    run_video_frames(self.context(source))
                decode.assert_not_called()

    def test_one_bad_video_does_not_discard_successful_video(self) -> None:
        self.video("input/good.mp4")
        (self.root / "input" / "bad.mp4").write_bytes(b"bad-video")
        result = run_video_frames(self.context(self.root / "input", folder=True))
        self.assertEqual(result["imageCount"], 2)
        self.assertEqual(len(result["failedVideos"]), 1)
        self.assertTrue(Path(result["outputFolders"][0]).is_dir())

    def test_cpu_limit_is_shared_across_waiting_batches_and_wait_is_cancellable(self) -> None:
        first_context = self.context(self.root)
        stop = threading.Event()
        second_context = self.context(self.root, stop=stop)
        waiting = threading.Event()
        second_context.report = lambda _: waiting.set()
        with patch("backend.video_frame_resources.machine_resources", return_value={**SAFE_RESOURCES, "cpuCount": 2}):
            with self.pool.acquire(first_context):
                def attempt():
                    with self.pool.acquire(second_context):
                        self.fail("Waiter must not exceed the shared decoder limit")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(attempt)
                    try:
                        self.assertTrue(waiting.wait(5))
                        stop.set()
                        with self.assertRaises(TaskCancelled):
                            future.result(timeout=5)
                    finally:
                        stop.set()
                self.assertEqual(self.pool.status()["activeWorkers"], 1)
        self.assertEqual(self.pool.status()["activeWorkers"], 0)

    def test_http_submission_is_background_and_only_duplicate_config_is_blocked(self) -> None:
        manager = PlatformTaskManager()
        releases = {"one": threading.Event(), "two": threading.Event()}
        finished = {"one": threading.Event(), "two": threading.Event()}
        results = []
        def execute(_handler, context):
            scene = context.parameters["scene_code"]
            releases[scene].wait(5)
            context.check_cancelled()
            return {"message": scene + " completed", "outputFolders": []}
        def record(task, **values):
            results.append((task.name, values["status"]))
            finished["one" if task.name.endswith("one") else "two"].set()
        def request(scene):
            payload = {"paths": {"video_file": str(self.root / "input.mp4")}, "parameters": {"scene_code": scene}}
            body = json.dumps(payload).encode()
            raw = f"POST /api/functions/video_frames/run HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            server.AppHandler(connection, ("127.0.0.1", 1234), SimpleNamespace())
            header, content = bytes(output).split(b"\r\n\r\n", 1)
            return int(header.split(b" ")[1]), json.loads(content)
        with patch.object(server, "platform_task_manager", manager), patch.object(server, "execute", side_effect=execute), \
                patch.object(server, "record_finished_run", side_effect=record), \
                patch.object(server.license_store, "status", return_value={"activated": True}):
            try:
                code, first = request("one")
                self.assertEqual(code, 202)
                self.assertEqual(request("one")[0], 400)
                self.assertEqual(request("two")[0], 202)
                self.assertEqual(len(manager.active()), 2)
                manager.stop(first["task"]["id"])
                releases["one"].set()
                self.assertTrue(finished["one"].wait(5))
                self.assertFalse(finished["two"].is_set())
                releases["two"].set()
                self.assertTrue(finished["two"].wait(5))
                self.assertEqual(sorted(status for _, status in results), ["completed", "stopped"])
            finally:
                for release in releases.values():
                    release.set()


if __name__ == "__main__":
    unittest.main()
