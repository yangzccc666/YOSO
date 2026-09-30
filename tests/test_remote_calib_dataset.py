from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from backend.catalog import FunctionCatalog
from backend.groups import GroupCatalog
from backend.handlers import RunContext, has_handler
from backend.remote_calib_dataset import (
    _command,
    _configuration,
    is_remote_calib_parameters,
    run_remote_calib_dataset,
)
from backend.remote_calib_worker import run


class CalibrationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.annotations = self.root / "annotations"
        self.negative = self.root / "negative"
        for folder in (self.images, self.annotations, self.negative):
            folder.mkdir()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _args(self, count: int = 3, output: Path | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            image_dirs_json=json.dumps([str(self.images), str(self.negative)]),
            annotation_dirs_json=json.dumps([str(self.annotations)]),
            negative_dirs_json="[]",
            output_dir=str(output or self.root / "calib_dataset"),
            format="auto", count=count, seed=42,
        )

    def test_catalog_is_grouped_and_registered(self) -> None:
        catalog = FunctionCatalog(self.root / "functions.json")
        item = catalog.get("quant_calibration_dataset")
        self.assertIsNotNone(item)
        self.assertTrue(has_handler(item["handlerId"]))
        parameters = {parameter["id"]: parameter for parameter in item["parameters"]}
        self.assertEqual(parameters["execution_location"]["default"], "本地运行")
        self.assertEqual(parameters["remote_host"]["default"], "")
        grouped = GroupCatalog(self.root / "groups.json").decorate(catalog.list())
        item = next(entry for entry in grouped if entry["id"] == "quant_calibration_dataset")
        self.assertEqual(item["groupId"], "dataset_processing")

    def test_xml_selection_converts_images_avoids_name_collisions_and_overwrite(self) -> None:
        Image.new("RGB", (8, 8), color=(20, 20, 20)).save(self.images / "same.jpg")
        Image.new("RGB", (8, 8), color=(240, 240, 240)).save(self.negative / "same.png")
        Image.new("RGB", (8, 8), color=(130, 130, 130)).save(self.images / "middle.bmp")
        (self.annotations / "same.xml").write_text("<annotation><object><name>car</name></object><object><name>person</name></object></annotation>", encoding="utf-8")
        result = run(self._args(count=10))
        output = self.root / "calib_dataset"
        self.assertEqual(result["sampleCount"], 3)
        self.assertEqual(result["availableCount"], 3)
        self.assertEqual(result["classCounts"], {"car": 1, "person": 1, "background": 2})
        self.assertEqual(result["selectedSourceCounts"], {"original": 3, "copy": 0, "negative": 0})
        self.assertEqual(len(list(output.glob("*.jpg"))), 3)
        self.assertTrue((output / "same.jpg").exists())
        self.assertTrue((output / "same_2.jpg").exists())
        with self.assertRaisesRegex(ValueError, "输出目录已包含文件"):
            run(self._args(count=2))

    def test_labelme_json_and_count_limit(self) -> None:
        for index in range(5):
            Image.new("RGB", (8, 8), color=(index * 40, index * 40, index * 40)).save(self.images / f"frame_{index}.jpg")
            (self.annotations / f"frame_{index}.json").write_text(json.dumps({"shapes": [{"label": "defect"}]}), encoding="utf-8")
        result = run(self._args(count=2))
        self.assertEqual(result["annotationFormat"], "json")
        self.assertEqual(result["sampleCount"], 2)
        self.assertEqual(len(list((self.root / "calib_dataset").glob("*.jpg"))), 2)

    def test_keeps_all_originals_then_fills_with_copy_and_negative_images(self) -> None:
        for index in range(3):
            name = f"original_{index}"
            Image.new("RGB", (8, 8), color=(30 + index, 30, 30)).save(self.images / f"{name}.jpg")
            (self.annotations / f"{name}.xml").write_text(
                "<annotation><object><name>part</name></object></annotation>", encoding="utf-8"
            )
        for index in range(8):
            name = f"original_0_COPY_{index}"
            Image.new("RGB", (8, 8), color=(100 + index, 100, 100)).save(self.images / f"{name}.jpg")
            (self.annotations / f"{name}.xml").write_text(
                "<annotation><object><name>part</name></object></annotation>", encoding="utf-8"
            )
        for index in range(4):
            Image.new("RGB", (8, 8), color=(200 + index, 200, 200)).save(self.negative / f"background_{index}.jpg")
        args = SimpleNamespace(
            image_dirs_json=json.dumps([str(self.images)]),
            annotation_dirs_json=json.dumps([str(self.annotations)]),
            negative_dirs_json=json.dumps([str(self.negative)]),
            output_dir=str(self.root / "priority_calib"),
            format="auto", count=8, seed=42,
        )
        result = run(args)
        output_names = {path.name for path in (self.root / "priority_calib").glob("*.jpg")}
        self.assertTrue({f"original_{index}.jpg" for index in range(3)}.issubset(output_names))
        self.assertEqual(result["availableSourceCounts"], {"original": 3, "copy": 8, "negative": 4})
        self.assertEqual(sum(result["selectedSourceCounts"].values()), 8)
        self.assertEqual(result["selectedSourceCounts"]["original"], 3)
        self.assertGreater(result["selectedSourceCounts"]["copy"], 0)
        self.assertGreater(result["selectedSourceCounts"]["negative"], 0)

    def test_does_not_use_supplements_when_originals_fill_target(self) -> None:
        for index in range(4):
            name = f"source_{index}"
            Image.new("RGB", (8, 8), color=(index * 20, 50, 80)).save(self.images / f"{name}.jpg")
            (self.annotations / f"{name}.xml").write_text(
                "<annotation><object><name>item</name></object></annotation>", encoding="utf-8"
            )
        Image.new("RGB", (8, 8)).save(self.images / "source_0_copy.jpg")
        (self.annotations / "source_0_copy.xml").write_text(
            "<annotation><object><name>item</name></object></annotation>", encoding="utf-8"
        )
        result = run(self._args(count=2, output=self.root / "original_only"))
        self.assertEqual(result["selectedSourceCounts"], {"original": 2, "copy": 0, "negative": 0})

    def test_configuration_requires_tested_connection_and_quotes_paths(self) -> None:
        parameters = {
            "execution_location": "SSH 远程服务器",
            "image_dirs": "/srv/data/正样本\n/srv/data/负样本",
            "annotation_dirs": "/srv/data/labels",
            "negative_dirs": "/srv/data/独立负样本",
            "num_samples": 128, "remote_python": "/opt/yolo/bin/python",
        }
        context = RunContext("quant_calibration_dataset", {}, parameters, lambda _line: None)
        with patch("backend.remote_calib_dataset.resolve_tested_yolo_connection", return_value={
            "host": "example", "port": 22, "username": "dell", "password": "secret"
        }):
            config = _configuration(context)
        self.assertEqual(config["output"], "/srv/data/calib_dataset")
        self.assertEqual(config["negatives"], ["/srv/data/独立负样本"])
        command = _command(config, "/tmp/calib-dataset-abc.py", "calib-dataset-abc")
        self.assertIn("正样本", command)
        self.assertNotIn("secret", command)
        self.assertIn("--count 128", command)
        self.assertIn("--negative-dirs-json", command)

    def test_local_mode_does_not_resolve_ssh_and_runs_worker(self) -> None:
        Image.new("RGB", (8, 8), color=(80, 100, 120)).save(self.images / "part.jpg")
        (self.annotations / "part.xml").write_text(
            "<annotation><object><name>part</name></object></annotation>", encoding="utf-8"
        )
        output = self.root / "local_calib"
        parameters = {
            "execution_location": "本地运行",
            "image_dirs": str(self.images),
            "annotation_dirs": str(self.annotations),
            "output_dir": str(output),
            "num_samples": 128,
        }
        messages: list[str] = []
        context = RunContext("quant_calibration_dataset", {}, parameters, messages.append)
        with patch("backend.remote_calib_dataset.resolve_tested_yolo_connection") as resolve_connection:
            result = run_remote_calib_dataset(context)
        resolve_connection.assert_not_called()
        self.assertFalse(is_remote_calib_parameters(parameters))
        self.assertEqual(result["sampleCount"], 1)
        self.assertTrue((output / "part.jpg").is_file())
        self.assertTrue(any("本机 Python" in message for message in messages))

    def test_remote_mode_detection_is_explicit(self) -> None:
        self.assertTrue(is_remote_calib_parameters({"execution_location": "SSH 远程服务器"}))
        self.assertFalse(is_remote_calib_parameters({}))


if __name__ == "__main__":
    unittest.main()
