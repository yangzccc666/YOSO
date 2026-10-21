from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.task_manager import TaskCancelled
from backend.video_clip import (
    COPY_MODE, PRECISE_MODE, REMOVE_MODE, _join_remaining, _probe_duration,
    parse_removal_ranges, parse_time, remaining_ranges, run_video_clip,
)


class RemovalRangeTests(unittest.TestCase):
    def test_accepts_lines_semicolons_time_formats_and_open_end(self):
        self.assertEqual(parse_removal_ranges("00:10-00:20\n01:30～02:00；150-; 03:00至结尾"),
                         [(10, 20), (90, 120), (150, None), (180, None)])

    def test_rejects_empty_invalid_backwards_negative_and_nonfinite_times(self):
        for value in ("", " ", "00:10", "20-10", "10-10", "-1-10", "nan-20", "10-inf", "01:99-02:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_removal_ranges(value)
        for value in (float("nan"), float("inf"), "inf"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_time(value, "时间")

    def test_sorts_merges_overlaps_adjacencies_and_clamps_to_duration(self):
        kept, removed, adjusted = remaining_ranges(
            [(40, 60), (10, 20), (15, 25), (25, 30), (90, 120), (150, 180)], 100)
        self.assertEqual(removed, [(10, 30), (40, 60), (90, 100)])
        self.assertEqual(kept, [(0, 10), (30, 40), (60, 90)])
        self.assertTrue(adjusted)

    def test_removes_head_tail_and_uses_original_timeline(self):
        kept, removed, adjusted = remaining_ranges(parse_removal_ranges("0-2;4-5;8-"), 10)
        self.assertEqual(kept, [(2, 4), (5, 8)])
        self.assertEqual(removed, [(0, 2), (4, 5), (8, 10)])
        self.assertFalse(adjusted)

    def test_rejects_deleting_everything_or_no_intersection(self):
        for ranges in ([(0, None)], [(0, 5), (4, 10)], [(12, 20)]):
            with self.subTest(ranges=ranges), self.assertRaises(ValueError):
                remaining_ranges(ranges, 10)

    def test_catalog_upgrade_adds_mode_preserving_user_defaults_and_names(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "functions.json"
            catalog = FunctionCatalog(storage)
            item = next(item for item in catalog.list() if item["id"] == "video_clip")
            item["name"] = "我的裁剪"
            item["parameters"] = [p for p in item["parameters"] if p["id"] != "remove_time_ranges"]
            for parameter in item["parameters"]:
                if parameter["id"] == "clip_mode":
                    parameter["options"] = ["跳过开头和结尾"]
                    parameter["default"] = "跳过开头和结尾"
                if parameter["id"] == "split_train_test":
                    parameter.pop("visibleWhen", None)
            storage.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
            storage.with_name(storage.name + ".video-clip-remove-v3").unlink(missing_ok=True)
            migrated = FunctionCatalog(storage).list()[0]
            parameters = {p["id"]: p for p in migrated["parameters"]}
            self.assertEqual(migrated["name"], "我的裁剪")
            self.assertEqual(parameters["clip_mode"]["default"], "跳过开头和结尾")
            self.assertIn(REMOVE_MODE, parameters["clip_mode"]["options"])
            self.assertIn("remove_time_ranges", parameters)
            self.assertEqual(parameters["split_train_test"]["visibleWhen"]["notEquals"], REMOVE_MODE)
            loaded = FunctionCatalog(storage).list()[0]
            self.assertEqual(sum(p["id"] == "remove_time_ranges" for p in loaded["parameters"]), 1)

    def test_cancel_and_failure_cleanup_staged_parts(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source.mp4"
                source.write_bytes(b"original")
                stop = threading.Event()
                context = RunContext("video_clip", {}, {}, lambda _: None, stop)
                def clip(ffmpeg, video, staging, start, end, encoding_mode, context, label):
                    context.check_cancelled()
                    if start > 0:
                        raise RuntimeError("模拟失败")
                    part = staging / "part.mp4"
                    part.write_bytes(b"part")
                    if cancel:
                        stop.set()
                    return part
                with patch("backend.video_clip._clip_one", side_effect=clip):
                    with self.assertRaises(TaskCancelled if cancel else RuntimeError):
                        _join_remaining("ffmpeg", source, root, [(0, 1), (2, 3)], 3, COPY_MODE, context)
                self.assertEqual(list(root.iterdir()), [source])
                self.assertEqual(source.read_bytes(), b"original")

    def test_precise_cancellation_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.mp4"
            source.write_bytes(b"original")
            context = RunContext("video_clip", {}, {}, lambda _: None)
            def interrupted(command, context):
                Path(command[-1]).write_bytes(b"partial")
                raise TaskCancelled("cancelled")
            with patch("backend.video_clip._has_audio", return_value=False), patch("backend.video_clip._run_process", side_effect=interrupted):
                with self.assertRaises(TaskCancelled):
                    _join_remaining("ffmpeg", source, root, [(0, 1), (2, 3)], 3, PRECISE_MODE, context)
            self.assertEqual(list(root.iterdir()), [source])


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg/ffprobe required")
class RemovalIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="video_remove_")
        self.root = Path(self.temporary.name) / "中文 test's"
        self.root.mkdir()
        self.source = self.root / "源视频.mp4"

    def tearDown(self):
        self.temporary.cleanup()

    def make_source(self, audio=False):
        if audio:
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                       "-f", "lavfi", "-i", "testsrc=size=64x48:rate=10:duration=8",
                       "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=8",
                       "-c:v", "mpeg4", "-g", "1", "-c:a", "aac", "-shortest", str(self.source)]
            completed = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
        else:
            writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(80):
                writer.write(np.full((48, 64, 3), index * 3, dtype=np.uint8))
            writer.release()

    def run_removal(self, mode=PRECISE_MODE, ranges="1-2\n4-6", **extra):
        return run_video_clip(RunContext("video_clip", {"video_file": self.source, "output_folder": self.root / "output"},
                                         {"clip_mode": REMOVE_MODE, "remove_time_ranges": ranges,
                                          "encoding_mode": mode, **extra}, lambda _: None))

    def test_precise_output_keeps_only_requested_frames_in_order_and_source_unchanged(self):
        self.make_source()
        original = self.source.read_bytes()
        result = self.run_removal(split_train_test=True)  # Hidden legacy state must not split.
        self.assertEqual(result["successCount"], 1)
        self.assertEqual(len(result["outputFiles"]), 1)
        output = Path(result["outputFiles"][0])
        self.assertIn("removed_joined", output.name)
        self.assertAlmostEqual(_probe_duration(output), 5, delta=0.12)
        self.assertEqual(result["details"][0]["durationSec"], 5)
        capture = cv2.VideoCapture(str(output))
        values = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                values.append(float(frame.mean()))
        finally:
            capture.release()
        expected = [index * 3 for index in [*range(10), *range(20, 40), *range(60, 80)]]
        self.assertEqual(len(values), len(expected))
        np.testing.assert_allclose(values, expected, atol=10)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse((output.parent / "train").exists())
        self.assertFalse(any(p.name.startswith(".") for p in output.parent.iterdir()))

    def test_precise_preserves_audio_with_expected_duration(self):
        self.make_source(audio=True)
        result = self.run_removal()
        output = Path(result["outputFiles"][0])
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration",
                                "-of", "json", str(output)], capture_output=True, text=True)
        streams = json.loads(probe.stdout)["streams"]
        self.assertEqual({stream["codec_type"] for stream in streams}, {"video", "audio"})
        for stream in streams:
            self.assertAlmostEqual(float(stream["duration"]), 5, delta=0.15)

    def test_stream_copy_preserves_codec_and_handles_chinese_apostrophe_paths(self):
        self.make_source(audio=True)
        result = self.run_removal(mode=COPY_MODE)
        output = Path(result["outputFiles"][0])
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                                "-of", "csv=p=0", str(output)], capture_output=True, text=True)
        self.assertEqual(probe.stdout.strip(), "mpeg4")
        self.assertAlmostEqual(_probe_duration(output), 5, delta=0.5)

    def test_single_remaining_segment_and_delete_to_end(self):
        self.make_source()
        result = self.run_removal(ranges="0-2;4-")
        self.assertAlmostEqual(_probe_duration(Path(result["outputFiles"][0])), 2, delta=0.12)

    def test_no_output_if_all_frames_deleted(self):
        self.make_source()
        with self.assertRaisesRegex(RuntimeError, "覆盖了整个视频"):
            self.run_removal(ranges="0-")
        self.assertEqual(list((self.root / "output").iterdir()), [])

    def test_batch_continues_if_a_shorter_video_has_no_valid_removal(self):
        self.make_source()
        folder = self.root / "batch"
        folder.mkdir()
        self.source.replace(folder / self.source.name)
        short = folder / "short.mp4"
        writer = cv2.VideoWriter(str(short), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for _ in range(10):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        result = run_video_clip(RunContext("video_clip", {"video_folder": folder, "output_folder": self.root / "output"},
                                {"clip_mode": REMOVE_MODE, "remove_time_ranges": "2-4", "encoding_mode": PRECISE_MODE}, lambda _: None))
        self.assertEqual(result["successCount"], 1)
        self.assertEqual(result["failedCount"], 1)
        self.assertIn("时长之外", result["failures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
