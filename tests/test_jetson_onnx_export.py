from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.jetson_onnx_export import _configuration, run_jetson_onnx_export


class JetsonOnnxExportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.weights = self.root / "best.pt"
        self.weights.write_bytes(b"weights")
        self.python = self.root / "python"
        self.python.touch()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _context(self, **parameters: object) -> RunContext:
        return RunContext(
            function_id="jetson_onnx_export",
            paths={"weights_file": self.weights, "output_folder": self.root},
            parameters={"python_path": str(self.python), **parameters},
            report=lambda _message: None,
        )

    def test_catalog_defaults_are_jetson_static_export(self) -> None:
        item = FunctionCatalog(self.root / "functions.json").get("jetson_onnx_export")
        self.assertIsNotNone(item)
        defaults = {field["id"]: field["default"] for field in item["parameters"]}
        self.assertEqual(defaults["task_type"], "目标检测")
        self.assertEqual(defaults["target_platform"], "Jetson / TensorRT（NCHW）")
        self.assertEqual(defaults["imgsz"], 640)
        self.assertEqual(defaults["opset"], 12)

    def test_configuration_generates_cut_name_and_validates_imgsz(self) -> None:
        config = _configuration(self._context())
        self.assertEqual(config["output"], self.root / "best_cut.onnx")
        self.assertEqual(config["task"], "detect")
        with self.assertRaisesRegex(ValueError, "32 的整数倍"):
            _configuration(self._context(imgsz=650))

    def test_handler_moves_verified_temporary_output_to_destination(self) -> None:
        def fake_run(command: list[str], _context: RunContext) -> None:
            output = Path(command[command.index("--output") + 1])
            output.write_bytes(b"valid-onnx")

        with patch("backend.jetson_onnx_export._run_process", side_effect=fake_run):
            result = run_jetson_onnx_export(self._context())
        output = self.root / "best_cut.onnx"
        self.assertEqual(result["outputOnnx"], str(output))
        self.assertEqual(result["onnxQuantDefaults"], {"inputOnnx": str(output)})
        self.assertEqual(output.read_bytes(), b"valid-onnx")
        self.assertEqual(self.weights.read_bytes(), b"weights")


if __name__ == "__main__":
    unittest.main()
