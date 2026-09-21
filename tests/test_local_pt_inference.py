from __future__ import annotations

import tempfile
import time
import unittest
import json
import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.groups import GroupCatalog
from backend.local_pt_inference import LocalPTInferenceManager, _find_yolo_python, validate_local_payload
from backend.local_pt_worker import receive as receive_worker_packet, send as send_worker_packet


class FakeResult:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def plot(self) -> np.ndarray:
        result = self.frame.copy()
        cv2.rectangle(result, (3, 3), (20, 20), (0, 255, 0), 2)
        return result


class FakeModel:
    names = {0: "object"}

    def __init__(self, _path: str) -> None:
        pass

    def predict(self, frame: np.ndarray, **_kwargs: object) -> list[FakeResult]:
        return [FakeResult(frame)]


class SlowFakeModel(FakeModel):
    def predict(self, frame: np.ndarray, **_kwargs: object) -> list[FakeResult]:
        time.sleep(0.24)
        return super().predict(frame, **_kwargs)


class LocalPTInferenceTests(unittest.TestCase):
    def test_worker_frame_protocol_and_python_override(self) -> None:
        stream = io.BytesIO()
        send_worker_packet(stream, {"type": "frame", "shape": [1, 1, 3]}, b"\x01\x02\x03")
        stream.seek(0)
        self.assertEqual(receive_worker_packet(stream), (
            {"type": "frame", "shape": [1, 1, 3]}, b"\x01\x02\x03",
        ))
        with tempfile.TemporaryDirectory() as temporary:
            python_path = Path(temporary) / "python"
            python_path.touch()
            with patch.dict("os.environ", {"PROCESSING_VIEW_YOLO_PYTHON": str(python_path)}):
                self.assertEqual(_find_yolo_python(), python_path.resolve())

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = self.root / "sample.pt"
        self.model.write_bytes(b"fake")
        self.video = self.root / "sample.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(100):
            writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
        writer.release()
        self.values = {"paths": {"model_file": str(self.model), "video_file": str(self.video), "output_folder": ""}, "parameters": {"realtime": True, "replay_grace_seconds": 1}}

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _until(self, predicate, timeout: float = 5) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.03)
        self.fail("Timed out waiting for local inference state")

    def test_catalog_entry_and_validation(self) -> None:
        items = FunctionCatalog(self.root / "functions.json").list()
        entry = next(item for item in items if item["id"] == "local_pt_inference")
        self.assertEqual(entry["handlerId"], "local.pt_inference")
        decorated = GroupCatalog(self.root / "groups.json").decorate(items)
        self.assertEqual(next(item for item in decorated if item["id"] == "local_pt_inference")["groupId"], "ai_inference")
        self.assertEqual(validate_local_payload(self.values)["conf"], 0.5)
        with_output = {**self.values, "paths": {**self.values["paths"], "output_folder": str(self.root)}}
        self.assertEqual(validate_local_payload(with_output)["output"], self.root / "sample_detected.mp4")
        with self.assertRaisesRegex(ValueError, r"\.pt"):
            validate_local_payload({"paths": {"model_file": str(self.video), "video_file": str(self.video)}})

    def test_existing_catalog_adds_local_entry_once(self) -> None:
        catalog_file = self.root / "legacy_functions.json"
        catalog_file.write_text(json.dumps([{"id": "remote_star_inference", "name": "远程推理", "handlerId": "remote.star_inference", "pathFields": [], "parameters": []}]), encoding="utf-8")
        catalog = FunctionCatalog(catalog_file)
        self.assertIsNotNone(catalog.get("local_pt_inference"))
        catalog.delete("local_pt_inference")
        self.assertIsNone(FunctionCatalog(catalog_file).get("local_pt_inference"))

    def test_preview_pause_seek_resume_and_stop(self) -> None:
        manager = LocalPTInferenceManager(FakeModel)
        initial = manager.start(self.values)
        session = manager.get(initial["id"])
        self._until(lambda: session.snapshot()["frameCount"] >= 3)
        self.assertIn("/api/local-pt-inference/", session.snapshot()["streamUrl"])
        self.assertEqual(manager.set_paused(session.id, True)["status"], "paused")
        time.sleep(0.18)
        count = session.snapshot()["frameCount"]
        time.sleep(0.2)
        self.assertEqual(session.snapshot()["frameCount"], count)
        manager.seek(session.id, 4)
        self._until(lambda: session.snapshot()["frameCount"] > count)
        self.assertEqual(session.snapshot()["status"], "paused")
        self.assertAlmostEqual(session.snapshot()["positionSeconds"], 4, delta=0.3)
        self.assertEqual(manager.set_paused(session.id, False)["status"], "running")
        self._until(lambda: session.snapshot()["frameCount"] >= count + 3)
        manager.stop(session.id)
        self._until(lambda: session.snapshot()["status"] == "stopped")

    def test_slow_model_skips_late_frames_instead_of_playing_slow_motion(self) -> None:
        manager = LocalPTInferenceManager(SlowFakeModel)
        started = time.monotonic()
        initial = manager.start(self.values)
        session = manager.get(initial["id"])
        self._until(lambda: session.snapshot()["frameCount"] >= 4, timeout=4)

        snapshot = session.snapshot()
        elapsed = time.monotonic() - started
        self.assertGreater(snapshot["positionSeconds"], elapsed * 0.55)
        self.assertTrue(any("跳过落后的源帧" in line for line in snapshot["logs"]))
        manager.stop(session.id)
        self._until(lambda: session.snapshot()["status"] == "stopped")


if __name__ == "__main__":
    unittest.main()
