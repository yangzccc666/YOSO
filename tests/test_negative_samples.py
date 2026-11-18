from __future__ import annotations

import tempfile
import json
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.negative_samples import run_extract_negative_samples
from backend.task_manager import TaskCancelled


class NegativeSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.images = self.root / "images"
        self.xmls = self.root / "Annotation" / "xmls"
        self.destination = self.root / "negative"
        self.images.mkdir()
        self.xmls.mkdir(parents=True)

    def image(self, name: str) -> Path:
        path = self.images / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
        return path

    def xml(self, name: str, body: str = "<annotation />") -> Path:
        path = self.xmls / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def extract(self, *, recursive=False, destination=None, stop_event=None):
        return run_extract_negative_samples(RunContext(
            function_id="extract_negative_samples",
            paths={"source_folder": self.images, "xml_folder": self.xmls,
                   "negative_folder": destination or self.destination},
            parameters={"recursive": recursive}, report=lambda _: None,
            stop_event=stop_event,
        ))

    def test_separate_folders_move_only_negative_images_and_delete_matching_xml(self):
        empty = self.image("空标注.JPG")
        empty_xml = self.xml("空标注.XML")
        positive = self.image("positive.jpeg")
        positive_xml = self.xml("positive.xml", "<annotation><object><name>part</name></object></annotation>")
        broken = self.image("broken.jpg")
        broken_xml = self.xml("broken.xml", "<annotation>")
        no_xml = self.image("no_xml.jpg")
        missing = self.xml("missing.xml")
        result = self.extract()
        self.assertEqual((result["negativeCount"], result["labeledCount"], result["skippedCount"]), (1, 1, 2))
        self.assertEqual((self.destination / empty.name).read_bytes(), b"image")
        self.assertFalse(empty.exists())
        self.assertFalse(empty_xml.exists())
        for path in (positive, positive_xml, broken, broken_xml, no_xml, missing):
            self.assertTrue(path.exists(), path)

    def test_recursive_separate_folders_prefer_same_relative_subdirectory(self):
        for folder in ("lineA", "lineB"):
            self.image(f"{folder}/same.jpg")
            self.xml(f"{folder}/same.xml")
        result = self.extract(recursive=True)
        self.assertEqual(result["negativeCount"], 2)
        for folder in ("lineA", "lineB"):
            self.assertTrue((self.destination / folder / "same.jpg").is_file())
            self.assertFalse((self.xmls / folder / "same.xml").exists())

    def test_unique_filename_fallback_keeps_image_relative_structure(self):
        image = self.image("lineA/deep/sample.jpeg")
        xml = self.xml("other_layout/sample.xml")
        result = self.extract(recursive=True)
        self.assertEqual(result["negativeCount"], 1)
        self.assertTrue((self.destination / "lineA/deep/sample.jpeg").is_file())
        self.assertFalse(image.exists())
        self.assertFalse(xml.exists())

    def test_recursive_off_does_not_scan_subdirectories(self):
        image = self.image("nested/sample.jpg")
        xml = self.xml("nested/sample.xml")
        result = self.extract()
        self.assertEqual(result["xmlCount"], 0)
        self.assertTrue(image.exists())
        self.assertTrue(xml.exists())

    def test_ambiguous_image_stems_are_skipped(self):
        paths = [self.image("a/sample.jpg"), self.image("b/sample.jpg"), self.xml("sample.xml")]
        result = self.extract(recursive=True)
        self.assertEqual(result["negativeCount"], 0)
        self.assertIn("多个同名", result["skipped"][0]["reason"])
        self.assertTrue(all(path.exists() for path in paths))

    def test_jpg_and_jpeg_with_same_stem_are_not_guessed(self):
        paths = [self.image("sample.jpg"), self.image("sample.jpeg"), self.xml("sample.xml")]
        result = self.extract()
        self.assertEqual(result["negativeCount"], 0)
        self.assertEqual(result["skippedCount"], 1)
        self.assertTrue(all(path.exists() for path in paths))

    def test_multiple_xml_for_one_image_cannot_remove_a_positive_sample(self):
        image = self.image("sample.jpg")
        empty = self.xml("a/sample.xml")
        positive = self.xml("b/sample.xml", "<annotation><object><name>part</name></object></annotation>")
        result = self.extract(recursive=True)
        self.assertEqual(result["negativeCount"], 0)
        self.assertEqual(result["skippedCount"], 2)
        self.assertIn("多个 XML", result["skipped"][0]["reason"])
        self.assertTrue(all(path.exists() for path in (image, empty, positive)))

    def test_separate_folder_destination_collision_preserves_both_inputs(self):
        image = self.image("sample.jpg")
        xml = self.xml("sample.xml")
        self.destination.mkdir()
        existing = self.destination / "sample.jpg"
        existing.write_bytes(b"existing")
        result = self.extract()
        self.assertEqual(result["negativeCount"], 0)
        self.assertTrue(image.exists())
        self.assertTrue(xml.exists())
        self.assertEqual(existing.read_bytes(), b"existing")

    def test_invalid_xml_folder_never_silently_falls_back_to_images(self):
        self.xmls = self.root / "missing"
        image = self.image("sample.jpg")
        (self.images / "sample.xml").write_text("<annotation />", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "XML 标注文件夹"):
            self.extract()
        self.assertTrue(image.exists())
        self.assertFalse(self.destination.exists())

    def test_output_inside_either_input_folder_is_rejected(self):
        for folder in (self.images, self.xmls):
            for destination in (folder, folder / "negative"):
                with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "输入目录"):
                    self.extract(destination=destination)

    def test_xml_deletion_failure_restores_image_to_original_folder(self):
        image = self.image("sample.jpg")
        xml = self.xml("sample.xml")
        with patch.object(Path, "unlink", side_effect=PermissionError("read-only XML")):
            result = self.extract()
        self.assertEqual(result["negativeCount"], 0)
        self.assertEqual(result["skippedCount"], 1)
        self.assertTrue(image.exists())
        self.assertTrue(xml.exists())
        self.assertFalse((self.destination / image.name).exists())

    def test_cancelled_task_does_not_move_or_delete_files(self):
        image = self.image("sample.jpg")
        xml = self.xml("sample.xml")
        stopped = threading.Event()
        stopped.set()
        with self.assertRaises(TaskCancelled):
            self.extract(stop_event=stopped)
        self.assertTrue(image.exists())
        self.assertTrue(xml.exists())

    def test_catalog_exposes_optional_xml_folder_and_migrates_old_config_once(self):
        storage = self.root / "functions.json"
        item = FunctionCatalog(storage).get("extract_negative_samples")
        self.assertEqual([field["id"] for field in item["pathFields"]], ["source_folder", "xml_folder", "negative_folder"])
        item["pathFields"] = [field for field in item["pathFields"] if field["id"] != "xml_folder"]
        item["pathFields"][0]["label"] = "图片和 XML 输入文件夹"
        item["name"] = "我的负样本提取"
        item["parameters"][0]["default"] = True
        item["pathFields"].append({"id": "custom", "label": "保留自定义路径", "mode": "directory"})
        storage.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
        marker = storage.with_name(storage.name + ".negative-sample-separate-folders-v2")
        marker.unlink(missing_ok=True)
        for _ in range(2):
            migrated = FunctionCatalog(storage).get("extract_negative_samples")
            self.assertEqual([field["id"] for field in migrated["pathFields"]], ["source_folder", "xml_folder", "negative_folder", "custom"])
            self.assertEqual(migrated["name"], "我的负样本提取")
            self.assertTrue(migrated["parameters"][0]["default"])
            self.assertIn("混放", migrated["pathFields"][0]["label"])

    def test_explicit_same_folder_keeps_mixed_folder_behavior(self):
        self.xmls = self.images
        image = self.image("nested/sample.jpg")
        xml = self.xml("nested/sample.xml")
        result = self.extract(recursive=True)
        self.assertEqual(result["negativeCount"], 1)
        self.assertTrue((self.destination / "nested/sample.jpg").is_file())
        self.assertFalse(image.exists())
        self.assertFalse(xml.exists())

    def test_moves_only_images_with_valid_empty_xml_and_deletes_that_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "negative"
            source.mkdir()
            (source / "empty.jpg").write_bytes(b"empty-image")
            (source / "empty.xml").write_text("<annotation></annotation>", encoding="utf-8")
            (source / "labeled.jpg").write_bytes(b"labeled-image")
            (source / "labeled.xml").write_text(
                "<annotation><object><name>part</name></object></annotation>", encoding="utf-8"
            )
            (source / "broken.jpg").write_bytes(b"broken-image")
            (source / "broken.xml").write_text("<annotation>", encoding="utf-8")

            messages: list[str] = []
            result = run_extract_negative_samples(RunContext(
                function_id="extract_negative_samples",
                paths={"source_folder": source, "negative_folder": destination},
                parameters={"recursive": False},
                report=messages.append,
            ))

            self.assertEqual(result["negativeCount"], 1)
            self.assertEqual(result["labeledCount"], 1)
            self.assertEqual(result["skippedCount"], 1)
            self.assertFalse((source / "empty.jpg").exists())
            self.assertFalse((source / "empty.xml").exists())
            self.assertEqual((destination / "empty.jpg").read_bytes(), b"empty-image")
            self.assertTrue((source / "labeled.jpg").exists())
            self.assertTrue((source / "labeled.xml").exists())
            self.assertTrue((source / "broken.jpg").exists())
            self.assertTrue((source / "broken.xml").exists())
            self.assertTrue(any("XML 无法解析" in message for message in messages))

    def test_existing_destination_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "negative"
            source.mkdir()
            destination.mkdir()
            (source / "sample.jpg").write_bytes(b"new")
            (source / "sample.xml").write_text("<annotation />", encoding="utf-8")
            (destination / "sample.jpg").write_bytes(b"existing")

            result = run_extract_negative_samples(RunContext(
                function_id="extract_negative_samples",
                paths={"source_folder": source, "negative_folder": destination},
                parameters={},
                report=lambda _message: None,
            ))

            self.assertEqual(result["negativeCount"], 0)
            self.assertEqual(result["skippedCount"], 1)
            self.assertEqual((destination / "sample.jpg").read_bytes(), b"existing")
            self.assertTrue((source / "sample.jpg").exists())
            self.assertTrue((source / "sample.xml").exists())


if __name__ == "__main__":
    unittest.main()
