from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.image_dedup import BACKUP_FOLDER, run_image_dedup


class ImageDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "images"
        self.root.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _scene(kind: int) -> np.ndarray:
        image = np.full((100, 140, 3), 40, dtype=np.uint8)
        if kind == 0:
            cv2.rectangle(image, (15, 20), (60, 75), (180, 90, 30), -1)
        elif kind == 1:
            cv2.circle(image, (105, 50), 25, (20, 220, 210), -1)
        else:
            cv2.rectangle(image, (70, 15), (125, 85), (50, 80, 230), -1)
        return image

    def _write_sequence(self, kinds: list[int], with_labels: bool = False) -> None:
        for index, kind in enumerate(kinds):
            image = self.root / f"process_{index}.jpg"
            self.assertTrue(cv2.imwrite(str(image), self._scene(kind)))
            if with_labels:
                image.with_suffix(".txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    def _run(self, **parameters: object) -> dict[str, object]:
        return run_image_dedup(RunContext(
            function_id="image_dedup",
            paths={"images_folder": self.root},
            parameters={
                "operation": "分析并去重",
                "dedup_strength": "平衡（推荐）",
                "minimum_keep_ratio": 0.2,
                "max_consecutive_skips": 100,
                **parameters,
            },
            report=lambda _message: None,
        ))

    def test_catalog_contains_reversible_dedup(self) -> None:
        item = FunctionCatalog(self.root / "functions.json").get("image_dedup")
        self.assertIsNotNone(item)
        self.assertEqual(item["handlerId"], "image.smart_dedup")
        operation = next(field for field in item["parameters"] if field["id"] == "operation")
        self.assertIn("回滚最近一次", operation["options"])

    def test_dedup_keeps_process_changes_and_rolls_back_images_and_labels(self) -> None:
        self._write_sequence([0] * 10 + [1] * 5 + [2] * 15, with_labels=True)
        result = self._run()
        remaining = sorted(self.root.glob("*.jpg"))
        self.assertEqual(result["imageCount"], 30)
        self.assertEqual(result["keptCount"], 6)
        self.assertEqual(result["removedCount"], 24)
        self.assertEqual(len(remaining), 6)
        self.assertEqual(len(list(self.root.glob("*.txt"))), 6)
        # Every process scene still has at least one representative.
        colors = [cv2.imread(str(path))[50, 105].tolist() for path in remaining]
        self.assertTrue(any(color[1] > 150 for color in colors))
        self.assertTrue(Path(str(result["backupManifest"])).is_file())

        rollback = run_image_dedup(RunContext(
            function_id="image_dedup",
            paths={"images_folder": self.root},
            parameters={"operation": "回滚最近一次"},
            report=lambda _message: None,
        ))
        self.assertEqual(rollback["restoredCount"], 24)
        self.assertEqual(len(list(self.root.glob("*.jpg"))), 30)
        self.assertEqual(len(list(self.root.glob("*.txt"))), 30)

    def test_brief_process_event_is_not_removed(self) -> None:
        self._write_sequence([0] * 10 + [1] + [0] * 10)
        result = self._run(minimum_keep_ratio=0.1)
        self.assertGreater(result["removedCount"], 0)
        self.assertTrue((self.root / "process_10.jpg").is_file())

    def test_different_annotation_classes_protect_identical_images(self) -> None:
        self._write_sequence([0, 0, 0])
        (self.root / "process_0.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (self.root / "process_1.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (self.root / "process_2.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        result = self._run(minimum_keep_ratio=0.1)
        self.assertEqual(result["removedCount"], 0)
        self.assertFalse((self.root.parent / f".{self.root.name}{BACKUP_FOLDER}").exists())

    def test_strength_changes_how_small_brightness_differences_are_treated(self) -> None:
        counts = {}
        base = self._scene(0)
        for strength in ("保守", "平衡（推荐）"):
            folder = self.root / strength
            folder.mkdir()
            for index in range(20):
                image = np.clip(base.astype(np.int16) + (5 if index % 2 else 0), 0, 255).astype(np.uint8)
                self.assertTrue(cv2.imwrite(str(folder / f"frame_{index}.jpg"), image))
            result = run_image_dedup(RunContext(
                function_id="image_dedup",
                paths={"images_folder": folder},
                parameters={
                    "operation": "分析并去重", "dedup_strength": strength,
                    "minimum_keep_ratio": 0.1, "max_consecutive_skips": 100,
                },
                report=lambda _message: None,
            ))
            counts[strength] = result["keptCount"]
        self.assertGreater(counts["保守"], counts["平衡（推荐）"])


if __name__ == "__main__":
    unittest.main()
