from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.video_clip import (
    COPY_MODE, PRECISE_MODE, REMOVE_MODE, _configured_output_name,
    _named_output_stem, _unique_output_file, run_video_clip,
)


class ClipNamingTests(unittest.TestCase):
    def test_parse_optional_extension_chinese_and_spaces(self):
        for value in (None, "", "  "):
            self.assertEqual(_configured_output_name(value), "")
        for value in ("TCL_工位1_光线变化", "TCL_工位1_光线变化.MP4", " TCL_工位1_光线变化 "):
            self.assertEqual(_configured_output_name(value), "TCL_工位1_光线变化")
        self.assertEqual(_configured_output_name("project's test.v2.mp4"), "project's test.v2")

    def test_invalid_filename_rejected_before_processing(self):
        for value in ("../escape", "/tmp/name", r"C:\name", "bad/name", "bad\\name", "bad?name", "bad\nname",
                      "bad|name", 'bad"name', "bad:name", "bad<name>", ".", "..", ".mp4", "ending.",
                      "CON", "prn.mp4", "AUX.test", "LPT1", "COM9", "测" * 61):
            with self.subTest(value=value), self.assertRaises(ValueError), patch("backend.video_clip._find_ffmpeg") as ffmpeg:
                run_video_clip(RunContext("video_clip", {}, {"clip_output_name": value}, lambda _: None))
            ffmpeg.assert_not_called()

    def test_name_roles_batch_and_length(self):
        video = Path("/a/工位1.mp4")
        self.assertEqual(_named_output_stem("光线变化", video, False), "光线变化")
        self.assertEqual(_named_output_stem("光线变化", video, False, "train"), "光线变化_train")
        self.assertEqual(_named_output_stem("光线变化", video, True, "test"), "光线变化_工位1_test")
        self.assertEqual(_named_output_stem("", video, True, "test"), "")
        long = _named_output_stem("光" * 60, Path("工" * 80 + ".mp4"), True, "train")
        self.assertLess(len((long + "_999.mp4").encode("utf-8")), 255)

    def test_collision_safety_preserves_source_and_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "custom.mp4"
            source.write_bytes(b"original")
            (root / "custom_2.mp4").write_bytes(b"existing output")
            chosen = _unique_output_file(root, source, 1, 2, ".mp4", output_name="custom")
            self.assertEqual(chosen.name, "custom_3.mp4")
            self.assertEqual(source.read_bytes(), b"original")
            self.assertIn("custom_clip_000001-000002_", _unique_output_file(root, source, 1, 2, ".mp4").name)

    def test_catalog_upgrades_existing_config_without_duplicate_or_lost_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory) / "functions.json"
            original = FunctionCatalog(storage).list()
            clip = next(item for item in original if item["id"] == "video_clip")
            clip["name"] = "我的裁剪"
            clip["parameters"] = [field for field in clip["parameters"] if field["id"] != "clip_output_name"]
            next(field for field in clip["parameters"] if field["id"] == "range_start_time")["default"] = "1:30"
            storage.write_text(json.dumps(original, ensure_ascii=False), encoding="utf-8")
            for _ in range(2):
                clip = next(item for item in FunctionCatalog(storage).list() if item["id"] == "video_clip")
                self.assertEqual(clip["name"], "我的裁剪")
                self.assertEqual(sum(field["id"] == "clip_output_name" for field in clip["parameters"]), 1)
                self.assertEqual(next(field for field in clip["parameters"] if field["id"] == "range_start_time")["default"], "1:30")


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg required")
class ClipNamingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "源视频.mp4"
        result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                                 "testsrc=size=64x48:rate=10:duration=4", "-c:v", "mpeg4", "-g", "1", str(self.source)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def run_clip(self, paths=None, **parameters):
        return run_video_clip(RunContext("video_clip", paths or {"video_file": self.source, "output_folder": self.root / "out"},
                                        {"clip_output_name": "TCL_工位1_光线变化", "range_start_time": "1", "range_end_time": "3",
                                         **parameters}, lambda _: None))

    def test_single_video_copy_precise_and_repeated_run(self):
        original = self.source.read_bytes()
        first = self.run_clip(encoding_mode=COPY_MODE)
        second = self.run_clip(encoding_mode=PRECISE_MODE)
        self.assertEqual(Path(first["outputFiles"][0]).name, "TCL_工位1_光线变化.mp4")
        self.assertEqual(Path(second["outputFiles"][0]).name, "TCL_工位1_光线变化_2.mp4")
        self.assertTrue(all(Path(path).is_file() for result in (first, second) for path in result["outputFiles"]))
        self.assertEqual(self.source.read_bytes(), original)

    def test_train_test_names(self):
        result = self.run_clip(split_train_test=True)
        self.assertEqual({Path(path).name for path in result["outputFiles"]}, {"TCL_工位1_光线变化_train.mp4", "TCL_工位1_光线变化_test.mp4"})
        self.assertTrue(all(Path(path).is_file() for path in result["outputFiles"]))

    def test_batch_names_preserve_source_identity(self):
        folder = self.root / "batch"
        folder.mkdir()
        for name in ("工位1.mp4", "工位2.mp4"):
            shutil.copyfile(self.source, folder / name)
        result = self.run_clip(paths={"video_folder": folder, "output_folder": self.root / "out"})
        self.assertEqual({Path(path).name for path in result["outputFiles"]}, {"TCL_工位1_光线变化_工位1.mp4", "TCL_工位1_光线变化_工位2.mp4"})

    def test_remove_and_join_uses_name_in_both_encoding_modes(self):
        for mode in (COPY_MODE, PRECISE_MODE):
            result = self.run_clip(clip_mode=REMOVE_MODE, remove_time_ranges="1-2", encoding_mode=mode)
            output = Path(result["outputFiles"][0])
            self.assertEqual(output.name, "TCL_工位1_光线变化.mp4" if mode == COPY_MODE else "TCL_工位1_光线变化_2.mp4")
            self.assertTrue(output.is_file())

    def test_container_extension_follows_actual_encoding(self):
        mkv = self.root / "source.mkv"
        completed = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(self.source), "-c", "copy", str(mkv)], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = self.run_clip(paths={"video_file": mkv, "output_folder": self.root / "out"}, clip_output_name="my_result.mp4")
        self.assertEqual(Path(result["outputFiles"][0]).name, "my_result.mkv")
