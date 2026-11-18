from __future__ import annotations

import tempfile
import time
import unittest
import json
import io
import subprocess
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.groups import GroupCatalog
from backend.local_pt_inference import LocalPTInferenceManager, _find_yolo_python, validate_local_payload, detect_local_model_labels
from backend.local_pt_worker import receive as receive_worker_packet, send as send_worker_packet
from backend.local_pt_worker import inspect_model_file, normalize_names, parse_label_text


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

    def test_onnx_paths_and_labels_are_bound_to_model(self):
        onnx = self.root / "检测模型.onnx"
        onnx.touch()
        values = {"paths": {**self.values["paths"], "model_file": str(onnx)},
                  "parameters": {"labels": "0: MARK\n1：中文类别", "labels_model_file": str(onnx)}}
        config = validate_local_payload(values)
        self.assertEqual(config["model"], onnx.resolve())
        self.assertEqual(config["labels"], ["MARK", "中文类别"])
        values["parameters"]["labels_model_file"] = str(self.model)
        self.assertEqual(validate_local_payload(values)["labels"], [])
        values["parameters"] = {"labels": "1: MARK"}
        with self.assertRaisesRegex(ValueError, "序号"):
            validate_local_payload(values)

    def test_class_ids_are_sorted_and_no_model_text_is_evaluated(self):
        self.assertEqual(normalize_names("{'1': 'NGO', '0': 'MARK'}"), ["MARK", "NGO"])
        self.assertEqual(normalize_names(["OK", "OK"]), ["OK", "OK"])
        self.assertEqual(parse_label_text("0: MARK\r\n1：中文类别"), ["MARK", "中文类别"])
        for names in ["__import__('os').getcwd()", {1: "wrong"}, {0: ""}, {0.5: "bad"}, {False: "bad"}, ["bad\x00"]]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                normalize_names(names)

    def _onnx_graph(self, names="{1: 'NGO', 0: 'MARK'}", task="detect", outputs=1, batch=1):
        dims = [SimpleNamespace(dim_value=value) for value in (batch, 3, 320, 640)]
        tensor = lambda shape: SimpleNamespace(type=SimpleNamespace(tensor_type=SimpleNamespace(shape=SimpleNamespace(dim=shape))))
        image = tensor(dims)
        image.name = "images"
        output = tensor([SimpleNamespace(dim_value=value) for value in (1, 6, 8400)])
        metadata = [SimpleNamespace(key="task", value=task)]
        if names is not None:
            metadata.append(SimpleNamespace(key="names", value=names))
        return SimpleNamespace(metadata_props=metadata, graph=SimpleNamespace(input=[image], initializer=[], output=[output] * outputs))

    def test_onnx_metadata_does_not_fall_back_to_coco_or_load_external_weights(self):
        fake = SimpleNamespace(load=lambda _path, **kwargs: self._onnx_graph())
        with patch.dict("sys.modules", {"onnx": fake}):
            metadata, model = inspect_model_file("fixture.onnx")
        self.assertIsNone(model)
        self.assertEqual(metadata["labels"], ["MARK", "NGO"])
        self.assertEqual(metadata["imageSize"], [320, 640])
        calls = []
        fake.load = lambda _path, **kwargs: (calls.append(kwargs) or self._onnx_graph(names=None))
        with patch.dict("sys.modules", {"onnx": fake}):
            metadata, _ = inspect_model_file("fixture.onnx")
        self.assertFalse(metadata["found"])
        self.assertEqual(metadata["labels"], [])
        self.assertEqual(calls, [{"load_external_data": False}])

    def test_cut_heads_non_detection_and_batch_models_are_rejected(self):
        for options in ({"outputs": 6}, {"task": "segment"}, {"batch": 4}):
            fake = SimpleNamespace(load=lambda *_args, **_kwargs: self._onnx_graph(**options))
            with self.subTest(options=options), patch.dict("sys.modules", {"onnx": fake}), self.assertRaises(ValueError):
                inspect_model_file("fixture.onnx")

    def test_label_detection_worker_protocol_error_and_timeout(self):
        stream = io.BytesIO()
        send_worker_packet(stream, {"type": "metadata", "labels": ["MARK"], "found": True})
        with patch("backend.local_pt_inference.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=stream.getvalue(), stderr=b"")) as run:
            self.assertEqual(detect_local_model_labels(str(self.model))["labels"], ["MARK"])
            self.assertEqual(run.call_args.args[0][-1], "--inspect")
        stream = io.BytesIO()
        send_worker_packet(stream, {"type": "error", "message": "bad model"})
        with patch("backend.local_pt_inference.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout=stream.getvalue(), stderr=b"")), self.assertRaisesRegex(ValueError, "bad model"):
            detect_local_model_labels(str(self.model))
        with patch("backend.local_pt_inference.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 90)), self.assertRaisesRegex(ValueError, "超时"):
            detect_local_model_labels(str(self.model))

    def test_labels_api_is_license_guarded_and_returns_model_metadata(self):
        from backend import server
        def request(activated, result=None, error=None):
            body = json.dumps({"modelFile": str(self.model)}).encode()
            raw = b"POST /api/local-pt-inference/detect-labels HTTP/1.0\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            with patch.object(server.license_store, "status", return_value={"activated": activated, "role": "user", "message": "activation required"}), \
                 patch.object(server, "detect_local_model_labels", return_value=result, side_effect=error) as detect:
                server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
                if not activated:
                    detect.assert_not_called()
            head, payload = bytes(output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(payload)
        self.assertEqual(request(False)[0], 403)
        status, result = request(True, {"labels": ["MARK"], "modelType": "ONNX", "found": True})
        self.assertEqual(status, 200)
        self.assertEqual(result["labels"], ["MARK"])
        status, result = request(True, error=ValueError("ONNX 模型不兼容"))
        self.assertEqual(status, 400)
        self.assertIn("不兼容", result["error"])

    def test_legacy_catalog_updates_labels_but_preserves_custom_name(self):
        catalog_file = self.root / "catalog.json"
        entry = FunctionCatalog(catalog_file).get("local_pt_inference")
        entry["name"] = "本地 PT 视频检测"
        entry["pathFields"][0]["label"] = "本地 PT 权重文件"
        catalog_file.write_text(json.dumps([entry]), encoding="utf-8")
        updated = FunctionCatalog(catalog_file).get("local_pt_inference")
        self.assertIn("ONNX", updated["name"])
        self.assertIn("ONNX", updated["pathFields"][0]["label"])
        updated["name"] = "我的视频检测"
        catalog_file.write_text(json.dumps([updated]), encoding="utf-8")
        self.assertEqual(FunctionCatalog(catalog_file).get("local_pt_inference")["name"], "我的视频检测")

    def test_onnx_uses_same_playback_pipeline(self):
        model = self.root / "fixture.onnx"
        model.touch()
        manager = LocalPTInferenceManager(FakeModel)
        values = {"paths": {**self.values["paths"], "model_file": str(model)}, "parameters": {"realtime": False, "replay_grace_seconds": 0}}
        session = manager.get(manager.start(values)["id"])
        self._until(lambda: session.snapshot()["status"] in {"completed", "failed"})
        self.assertEqual(session.snapshot()["status"], "completed")
        self.assertGreater(session.snapshot()["frameCount"], 0)

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
