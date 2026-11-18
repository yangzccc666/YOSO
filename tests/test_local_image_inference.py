from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.catalog import FunctionCatalog, LOCAL_PT_INFERENCE_FUNCTION
from backend.local_image_dataset import collect_images, read_image, save_image, validate_image_paths
from backend.local_pt_inference import LocalPTInferenceManager, validate_local_payload


class ImageResult:
    def __init__(self, frame):
        self.frame = frame.copy()

    def plot(self):
        cv2.rectangle(self.frame, (3, 3), (25, 25), (0, 255, 0), 2)
        return self.frame


class ImageModel:
    names = {0: "MARK"}

    def __init__(self, _path):
        pass

    def predict(self, frame, **_options):
        return [ImageResult(frame)]


class LocalImageInferenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "中文图片"
        self.images.mkdir()
        self.output = self.root / "检测结果"
        self.model = self.root / "test.pt"
        self.model.touch()
        self.frame = np.full((48, 64, 3), 60, dtype=np.uint8)
        self.values = {"paths": {"model_file": str(self.model), "image_folder": str(self.images), "output_folder": str(self.output)},
                       "parameters": {"input_mode": "图片文件夹检测", "preview_fps": 0}}

    def write(self, relative):
        path = self.images / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, encoded = cv2.imencode(path.suffix.lower(), self.frame)
        self.assertTrue(ok)
        encoded.tofile(path)
        return path

    def run_images(self, values=None, model_factory=ImageModel):
        complete = threading.Event()
        snapshots = []
        def finished(snapshot):
            snapshots.append(snapshot)
            complete.set()
        manager = LocalPTInferenceManager(model_factory)
        first = manager.start(values or self.values, on_finished=finished)
        self.assertTrue(complete.wait(5), manager.get(first["id"]).snapshot())
        return snapshots[0], manager

    def test_image_mode_needs_no_video_and_requires_safe_output(self):
        config = validate_local_payload(self.values)
        self.assertEqual(config["input_mode"], "images")
        self.assertIsNone(config["video"])
        self.assertTrue(config["recursive_images"])
        self.assertFalse(self.output.exists(), "validation must not create output directories")
        for output in ("", str(self.images), str(self.root)):
            with self.subTest(output=output), self.assertRaises(ValueError):
                validate_image_paths(str(self.images), output)
        output_file = self.root / "not-a-folder"
        output_file.touch()
        with self.assertRaisesRegex(ValueError, "不是文件夹"):
            validate_image_paths(str(self.images), str(output_file))
        with self.assertRaisesRegex(ValueError, "图片文件夹"):
            validate_image_paths(str(self.root / "absent"), str(self.output))
        invalid = {**self.values, "parameters": {"input_mode": "bad"}}
        with self.assertRaisesRegex(ValueError, "输入类型"):
            validate_local_payload(invalid)

    def test_unicode_recursive_and_corrupt_images_keep_originals(self):
        sources = [self.write("中文一.jpg"), self.write("场景A/重复.png"), self.write("场景B/重复.png")]
        (self.images / "bad.jpg").write_bytes(b"not an image")
        (self.images / "ignore.xml").write_text("<annotation />")
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
        snapshot, _ = self.run_images()
        self.assertEqual(snapshot["status"], "completed", snapshot)
        self.assertEqual((snapshot["totalImages"], snapshot["processedImages"], snapshot["savedImages"], snapshot["failedImages"]), (4, 4, 3, 1))
        self.assertEqual(snapshot["progressPercent"], 100)
        self.assertEqual(snapshot["result"]["outputFolders"], [str(self.output)])
        self.assertGreater(snapshot["frameCount"], 0)
        for source in sources:
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before[source])
            output = self.output / source.relative_to(self.images)
            self.assertTrue(output.is_file())
            self.assertEqual(read_image(output).shape, self.frame.shape)
            self.assertFalse(np.array_equal(read_image(source), read_image(output)))
        self.assertFalse((self.output / "bad.jpg").exists())
        self.assertFalse((self.output / "ignore.xml").exists())

    def test_non_recursive_scan_and_old_output_subfolder_exclusion(self):
        self.write("one.JPG")
        self.write("child/two.png")
        output = self.images / "results"
        output.mkdir()
        (output / "old.jpg").write_bytes(b"old result")
        values = {"paths": {**self.values["paths"], "output_folder": str(output)},
                  "parameters": {**self.values["parameters"], "recursive_images": False}}
        snapshot, _ = self.run_images(values)
        self.assertEqual(snapshot["totalImages"], 1)
        self.assertEqual(snapshot["savedImages"], 1)
        self.assertEqual((output / "old.jpg").read_bytes(), b"old result")
        recursive = collect_images(self.images, output, True, threading.Event())
        self.assertEqual(len(recursive), 2)
        self.assertFalse(any(path.is_relative_to(output) for path in recursive))

    def test_second_run_does_not_overwrite_existing_results(self):
        self.write("sub/image.png")
        first, _ = self.run_images()
        original_output = (self.output / "sub/image.png").read_bytes()
        second, _ = self.run_images()
        self.assertEqual(first["status"], "completed")
        self.assertEqual(second["status"], "completed")
        self.assertEqual((self.output / "sub/image.png").read_bytes(), original_output)
        self.assertTrue((self.output / "sub/image_1.png").is_file())

    def test_empty_and_all_corrupt_folders_fail_clearly(self):
        snapshot, _ = self.run_images()
        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("未找到", snapshot["error"])
        self.assertFalse(self.output.exists())
        (self.images / "bad.jpg").write_bytes(b"bad")
        snapshot, _ = self.run_images()
        self.assertEqual(snapshot["status"], "failed")
        self.assertEqual(snapshot["failedImages"], 1)
        self.assertIn("没有成功保存", snapshot["error"])

    def test_model_failure_stops_batch_and_preserves_completed_files(self):
        for name in ("a.png", "b.png", "c.png"):
            self.write(name)
        class BrokenModel(ImageModel):
            count = 0
            def predict(self, frame, **options):
                self.count += 1
                if self.count == 2:
                    raise RuntimeError("inference engine failed")
                return super().predict(frame, **options)
        snapshot, _ = self.run_images(model_factory=BrokenModel)
        self.assertEqual(snapshot["status"], "failed")
        self.assertIn("inference engine failed", snapshot["error"])
        self.assertEqual(snapshot["savedImages"], 1)
        self.assertTrue((self.output / "a.png").is_file())
        self.assertFalse((self.output / "c.png").exists())

    def test_stop_preserves_saved_results_and_rejects_video_controls(self):
        for name in ("a.png", "b.png", "c.png"):
            self.write(name)
        waiting, release, finished = threading.Event(), threading.Event(), threading.Event()
        class BlockingModel(ImageModel):
            count = 0
            def predict(self, frame, **options):
                self.count += 1
                if self.count == 2:
                    waiting.set()
                    release.wait(5)
                return super().predict(frame, **options)
        manager = LocalPTInferenceManager(BlockingModel)
        session_id = manager.start(self.values, on_finished=lambda _: finished.set())["id"]
        try:
            self.assertTrue(waiting.wait(5))
            with self.assertRaisesRegex(ValueError, "图片批量检测"):
                manager.set_paused(session_id, True)
            manager.stop(session_id)
        finally:
            release.set()
        self.assertTrue(finished.wait(5))
        snapshot = manager.get(session_id).snapshot()
        self.assertEqual(snapshot["status"], "stopped")
        self.assertIsNone(snapshot["error"])
        self.assertEqual(snapshot["savedImages"], 1)
        self.assertEqual(len(list(self.output.glob("*.png"))), 1)

    def test_output_symlink_cannot_escape_to_source(self):
        self.output.mkdir()
        (self.output / "child").symlink_to(self.images, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "链接"):
            save_image(self.output / "child" / "new.png", self.frame, self.output)
        self.assertFalse((self.images / "new.png").exists())

    def test_catalog_migrates_old_settings_without_restoring_deleted_function(self):
        catalog_file = self.root / "functions.json"
        old = json.loads(json.dumps(LOCAL_PT_INFERENCE_FUNCTION))
        old["name"] = "本地 PT / ONNX 视频检测"
        old["pathFields"] = [field for field in old["pathFields"] if field["id"] != "image_folder"]
        old["parameters"] = [param for param in old["parameters"] if param["id"] not in {"input_mode", "recursive_images"}]
        catalog_file.write_text(json.dumps([old]), encoding="utf-8")
        catalog = FunctionCatalog(catalog_file)
        migrated = catalog.get("local_pt_inference")
        self.assertIn("image_folder", [field["id"] for field in migrated["pathFields"]])
        self.assertIn("input_mode", [field["id"] for field in migrated["parameters"]])
        self.assertEqual(next(param for param in migrated["parameters"] if param["id"] == "conf")["default"], .5)
        catalog.delete("local_pt_inference")
        self.assertIsNone(FunctionCatalog(catalog_file).get("local_pt_inference"))


if __name__ == "__main__":
    unittest.main()
