from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.jetson_onnx_export import detect_export_labels
from backend.local_pt_worker import send
from backend.model_labels_worker import checkpoint_names, read_pt_labels
from backend.worker_runtime import worker_command


class ExportModelLabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "中文.pt"
        self.model.touch()
        self.python = self.root / "python"
        self.python.touch()
        self.raw = {"modelFile": str(self.model), "parameters": {"python_path": str(self.python)}}

    def response(self, header, code=0):
        stream = io.BytesIO()
        send(stream, header)
        return SimpleNamespace(returncode=code, stdout=stream.getvalue(), stderr=b"")

    def test_worker_reads_id_order_duplicates_and_ema_model_or_checkpoint_names(self):
        self.assertEqual(checkpoint_names({"ema": SimpleNamespace(names={1: "中文", 0: "MARK"}),
                                           "model": SimpleNamespace(names=["WRONG"])}), ["MARK", "中文"])
        self.assertEqual(checkpoint_names({"ema": None, "model": SimpleNamespace(module=SimpleNamespace(names=["X", "X"]))}), ["X", "X"])
        self.assertEqual(checkpoint_names({"names": ["OK", "NG"]}), ["OK", "NG"])
        with self.assertRaisesRegex(ValueError, "没有保存"):
            checkpoint_names({"model": object()})
        with self.assertRaises(ValueError):
            checkpoint_names({"names": {1: "missing_zero"}})

    def test_yolo26_all_tasks_use_actual_model_names_without_detect_restriction(self):
        fake = SimpleNamespace(YOLO=lambda _path: SimpleNamespace(names={1: "B", 0: "A"}, task="segment"))
        with patch.dict(sys.modules, {"ultralytics": fake}):
            metadata = read_pt_labels(str(self.model), "YOLO26")
        self.assertEqual(metadata["labels"], ["A", "B"])
        self.assertEqual(metadata["task"], "segment")

    def repository(self):
        repo = self.root / "v5"
        (repo / "models").mkdir(parents=True)
        (repo / "export.py").touch()
        return repo

    def test_yolov5_uses_matching_repository_and_cpu_checkpoint(self):
        repo = self.repository()
        fake_torch = SimpleNamespace(load=lambda *_args, **_kwargs: {"model": SimpleNamespace(names=["A", "B"])})
        with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(sys, "path", sys.path.copy()):
            metadata = read_pt_labels(str(self.model), "YOLOv5", str(repo))
            self.assertEqual(sys.path[0], str(repo))
        self.assertEqual(metadata["count"], 2)

    def test_endpoint_uses_page_python_without_requiring_any_export_configuration(self):
        result = self.response({"type": "metadata", "labels": ["MARK", "NGO"], "count": 2})
        with patch("backend.jetson_onnx_export.subprocess.run", return_value=result) as run, \
             patch("backend.jetson_onnx_export._run_process") as export:
            self.assertEqual(detect_export_labels(self.raw)["labels"], ["MARK", "NGO"])
            command = run.call_args.args[0]
            self.assertEqual(command[0], str(self.python))
            self.assertIn("model_labels_worker.py", command[3])
            self.assertEqual(command[command.index("--weights") + 1], str(self.model))
            export.assert_not_called()
        self.assertEqual(list(self.root.glob("*.onnx")), [])

    def test_default_runtime_and_yolov5_repository_arguments(self):
        result = self.response({"type": "metadata", "labels": ["A"]})
        with patch("backend.jetson_onnx_export._find_yolo_python", return_value=self.python), patch("backend.jetson_onnx_export.subprocess.run", return_value=result):
            self.assertEqual(detect_export_labels({"modelFile": str(self.model)})["labels"], ["A"])
        repo = self.repository()
        raw = {**self.raw, "parameters": {**self.raw["parameters"], "model_family": "YOLOv5", "yolov5_repo": str(repo)}}
        with patch("backend.jetson_onnx_export.subprocess.run", return_value=result) as run:
            detect_export_labels(raw)
        self.assertIn(str(repo), run.call_args.args[0])
        self.assertIn("YOLOv5", run.call_args.args[0])

    def test_errors_and_timeout_do_not_fall_back_to_any_sidecar(self):
        (self.root / "classes.txt").write_text("WRONG\nOTHER")
        with patch("backend.jetson_onnx_export.subprocess.run", return_value=self.response({"type": "error", "message": "missing names"}, 1)), self.assertRaisesRegex(ValueError, "missing names"):
            detect_export_labels(self.raw)
        with patch("backend.jetson_onnx_export.subprocess.run", side_effect=subprocess.TimeoutExpired("python", 90)), self.assertRaisesRegex(ValueError, "超时"):
            detect_export_labels(self.raw)
        for raw in ({"modelFile": "not-there.pt"}, {**self.raw, "parameters": []},
                    {**self.raw, "parameters": {"model_family": "bad"}},
                    {**self.raw, "parameters": {"model_family": "YOLOv5"}}):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                detect_export_labels(raw)

    def test_new_metadata_worker_is_dispatched_in_frozen_build(self):
        script = Path(__file__).resolve().parents[1] / "backend/model_labels_worker.py"
        with patch.object(sys, "frozen", True, create=True):
            command = worker_command(Path(sys.executable), script, ["--weights", "model.pt"])
        self.assertEqual(command[:3], [sys.executable, "--internal-worker", "model_labels_worker"])

    def test_api_license_guard_success_and_error(self):
        from backend import server
        def request(activated, error=None):
            body = json.dumps(self.raw).encode()
            raw = b"POST /api/jetson-onnx-export/detect-labels HTTP/1.0\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            with patch.object(server.license_store, "status", return_value={"activated": activated, "role": "user", "message": "activation required"}), \
                 patch.object(server._jetson_onnx_export, "detect_export_labels", return_value={"labels": ["MARK"]}, side_effect=error) as detect:
                server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
                if not activated:
                    detect.assert_not_called()
            head, payload = bytes(output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(payload)
        self.assertEqual(request(False)[0], 403)
        self.assertEqual(request(True), (200, {"labels": ["MARK"]}))
        self.assertEqual(request(True, ValueError("bad model"))[0], 400)


if __name__ == "__main__":
    unittest.main()
