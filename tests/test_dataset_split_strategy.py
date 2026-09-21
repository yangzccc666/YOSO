from __future__ import annotations

import tempfile
import unittest
import random
from pathlib import Path
from types import SimpleNamespace

from backend.dataset_split_strategy import split_with_class_coverage
from backend.handlers import RunContext
from backend.remote_yolo_dataset_worker import run as run_remote_worker
from backend.yolo_dataset_split import run_yolo_dataset_split


class DatasetSplitStrategyTests(unittest.TestCase):
    def test_each_class_approximates_seventy_percent(self) -> None:
        # Reproduces the reported counts when classes are on separate images.
        totals = [61, 53, 6, 12, 11, 3]
        labels = [{category} for category, total in enumerate(totals) for _ in range(total)]
        train, val, distribution = split_with_class_coverage(labels, 0.7, 42, len(totals))
        self.assertEqual((len(train), len(val)), (102, 44))
        self.assertEqual([item["train"] for item in distribution], [43, 37, 4, 8, 8, 2])
        self.assertEqual([item["val"] for item in distribution], [18, 16, 2, 4, 3, 1])

    def test_overlapping_classes_still_follow_class_ratio(self) -> None:
        generator = random.Random(8)
        labels = [set(generator.sample(range(6), 2)) for _ in range(100)]
        train, val, distribution = split_with_class_coverage(labels, 0.7, 42, 6)
        self.assertEqual((len(train), len(val)), (70, 30))
        self.assertTrue(all(abs(item["train"] - item["total"] * 0.7) <= 1 for item in distribution))

    def test_rare_class_is_present_in_training_and_validation(self) -> None:
        labels = [{0}, {0}, {0}, {0}, {1}, {1}]
        # A plain seed-2 shuffle puts both rare-class images outside train.
        train, val, distribution = split_with_class_coverage(labels, 0.5, 2, 2)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 3)
        self.assertFalse(set(train) & set(val))
        self.assertEqual(distribution[1], {"total": 2, "train": 1, "val": 1})
        self.assertEqual(
            (train, val, distribution),
            split_with_class_coverage(labels, 0.5, 2, 2),
        )

    def test_multilabel_images_keep_every_observed_class_in_train(self) -> None:
        labels = [{0, 1}, {0, 2}, {1, 2}, {0}, {1}, {2}]
        train, val, distribution = split_with_class_coverage(labels, 0.5, 3, 3)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(val), 3)
        self.assertTrue(all(item["train"] >= 1 for item in distribution))
        self.assertTrue(all(item["val"] >= 1 for item in distribution))

    def test_local_and_remote_worker_preserve_rare_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            labels = root / "labels"
            local_output = root / "local_output"
            remote_output = root / "remote_output"
            for directory in (images, labels, local_output, remote_output):
                directory.mkdir()
            (labels / "classes.txt").write_text("common\nrare\n", encoding="utf-8")
            for index in range(6):
                (images / f"image_{index}.jpg").write_bytes(b"image")
                category = 1 if index >= 4 else 0
                (labels / f"image_{index}.txt").write_text(
                    f"{category} 0.5 0.5 0.2 0.2\n", encoding="utf-8"
                )

            local = run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={"images_folder": images, "labels_folder": labels, "output_folder": local_output},
                parameters={"label_format": "YOLO TXT", "train_ratio": 0.5, "random_seed": 2},
                report=lambda _message: None,
            ))
            remote = run_remote_worker(SimpleNamespace(
                images_folder=str(images), labels_folder=str(labels), negative_folder="",
                output_parent=str(remote_output), dataset_folder_name="yolo_train",
                label_format="yolo_txt", train_ratio=0.5, seed=2,
                missing_policy="skip", session_token="test-rare-class",
            ))
            self.assertEqual(local["classDistribution"], remote["classDistribution"])
            self.assertEqual(local["classDistribution"][1], {
                "name": "rare", "total": 2, "train": 1, "val": 1,
            })


if __name__ == "__main__":
    unittest.main()
