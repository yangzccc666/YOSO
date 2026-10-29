from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.video_frame_resources import VideoFrameResourcePool
from backend.video_frames import _english_identifier, _video_filename_identity, run_video_frames


class VideoFrameFilenameFallbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        resources = {"cpuCount": 8, "cpuPercent": 5, "availableMemoryMb": 16384, "maxWorkers": 4, "recommendedWorkers": 2, "warning": ""}
        self.patch_resources = patch("backend.video_frame_resources.machine_resources", return_value=resources)
        self.patch_resources.start()
        self.addCleanup(self.patch_resources.stop)
        self.patch_pool = patch("backend.video_frames.resource_pool", VideoFrameResourcePool())
        self.patch_pool.start()
        self.addCleanup(self.patch_pool.stop)

    def video(self, relative):
        source = self.root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(10):
            writer.write(np.full((48, 64, 3), index * 20, dtype=np.uint8))
        writer.release()
        return source

    def extract(self, source, folder=False, **parameters):
        return run_video_frames(RunContext("video_frames", {"video_folder" if folder else "video_file": source, "output_folder": self.root / "output"},
                                          {"target_fps": 2, "parallel_workers": 2, **parameters}, lambda _: None))

    def test_blank_or_absent_identifiers_use_original_chinese_video_name_once(self):
        source = self.video("工位1_光线变化.mp4")
        original = source.read_bytes()
        for parameters in ({}, {"project_code": "", "scene_code": "  "}):
            result = self.extract(source, **parameters)
            self.assertEqual(result["projectCode"], source.stem)
            self.assertEqual(result["sceneCode"], source.stem)
            folder = Path(result["outputFolders"][0])
            self.assertTrue(folder.name.startswith("工位1_光线变化_video01_fps2_"))
            names = sorted(file.name for file in folder.glob("*.jpg"))
            self.assertEqual(names, ["工位1_光线变化_video01_fps2_t000000000ms_f00000000.jpg", "工位1_光线变化_video01_fps2_t000000500ms_f00000005.jpg"])
            self.assertEqual(result["imageCount"], 2)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(len(list((self.root / "output").iterdir())), 2)

    def test_one_identifier_blank_and_explicit_identifiers_preserve_old_rules(self):
        source = self.video("Line1 工位2.v2.mp4")
        for parameters, expected in (({"project_code": "TCL"}, "tcl_Line1 工位2.v2_video01"),
                                     ({"scene_code": "Station 2"}, "Line1 工位2.v2_station_2_video01"),
                                     ({"project_code": "TCL", "scene_code": "Station 2"}, "tcl_station_2_video01")):
            with self.subTest(parameters=parameters):
                result = self.extract(source, **parameters)
                self.assertEqual(result["details"][0]["englishPrefix"], expected)
                self.assertTrue(all(path.name.startswith(expected + "_fps2_") for path in Path(result["outputFolders"][0]).glob("*.jpg")))

    def test_batch_recursive_same_named_videos_are_not_overwritten(self):
        self.video("input/one/工位1.mp4")
        self.video("input/two/nested/工位1.mp4")
        self.video("input/two/Line2.mp4")
        result = self.extract(self.root / "input", folder=True)
        self.assertEqual(result["videoCount"], 3)
        self.assertEqual(result["imageCount"], 6)
        self.assertEqual(len(set(result["outputFolders"])), 3)
        for detail in result["details"]:
            source = Path(detail["video"])
            self.assertEqual(detail["projectCode"], source.stem)
            self.assertEqual(detail["sceneCode"], source.stem)
            self.assertEqual(detail["englishPrefix"], f"{source.stem}_video{detail['index']:02d}")

    def test_other_sampling_modes_keep_video_name_and_sampling_information(self):
        source = self.video("测试视频.mp4")
        interval = self.extract(source, extraction_mode="按时间间隔", interval_sec=0.5)
        self.assertEqual(interval["details"][0]["samplingTag"], "every0p5s_fps2")
        self.assertTrue(all(path.name.startswith("测试视频_video01_every0p5s_fps2_") for path in Path(interval["outputFolders"][0]).glob("*.jpg")))
        priority = self.extract(source, extraction_mode="工序时间段高密度", priority_time_ranges="0-0.5", priority_fps=10, background_fps=2, priority_max_images=5)
        self.assertEqual(priority["imageCount"], 5)
        self.assertTrue(all(path.name.startswith("测试视频_video01_priority10_bg2_") for path in Path(priority["outputFolders"][0]).glob("*.jpg")))

    def test_optional_but_nonempty_english_identifiers_are_still_validated(self):
        for value in (None, "", "  "):
            self.assertEqual(_english_identifier(value, "标识"), "")
        self.assertEqual(_english_identifier("TCL 191", "标识"), "tcl_191")
        for value in ("中文", "../bad", "a/b", "a" * 49):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _english_identifier(value, "标识")

    def test_source_filename_sanitization_and_long_unicode_names(self):
        _, _, prefix = _video_filename_identity(Path("/tmp/Line 1: test?.mp4"), "", "", 1)
        self.assertEqual(prefix, "Line 1_ test__video01")
        project, scene, prefix = _video_filename_identity(Path("工" * 80 + ".mp4"), "a" * 48, "", 123)
        self.assertEqual(project, "a" * 48)
        self.assertLessEqual(len(scene.encode("utf-8")), 96)
        self.assertLess(len((prefix + "_every0p5s_fps2_t000000500ms_f00000005.jpg").encode("utf-8")), 255)

    def test_catalog_existing_labels_updated_without_changing_defaults(self):
        storage = self.root / "functions.json"
        catalog = FunctionCatalog(storage)
        item = catalog.get("video_frames")
        item["name"] = "我的抽帧"
        for field in item["parameters"]:
            if field["id"] == "project_code":
                field["label"] = "项目英文标识（如 tcl）"
                field["default"] = "factory_a"
            elif field["id"] == "scene_code":
                field["label"] = "场景英文标识（如 line2_station3）"
        storage.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
        for _ in range(2):
            migrated = FunctionCatalog(storage).get("video_frames")
            self.assertEqual(migrated["name"], "我的抽帧")
            fields = {field["id"]: field for field in migrated["parameters"]}
            self.assertEqual(fields["project_code"]["default"], "factory_a")
            self.assertIn("留空使用视频文件名", fields["project_code"]["label"])
            self.assertIn("留空使用视频文件名", fields["scene_code"]["label"])
            self.assertEqual(sum(field["id"] == "project_code" for field in migrated["parameters"]), 1)
