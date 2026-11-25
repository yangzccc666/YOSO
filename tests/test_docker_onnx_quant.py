from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from backend.catalog import FunctionCatalog
from backend.docker_onnx_quant import (
    CUSTOM_EXCLUDE_PRESET,
    DEFAULT_EXCLUDE_PRESET,
    EXCLUDE_MODULE_PRESETS,
    DockerMount,
    _container_path,
    _tool_arguments,
    _validate_export,
    run_docker_onnx_quant,
)
from backend.handlers import RunContext

LEGACY_EXCLUDE = r"^/model\.(13|16|17|23)(?:/|$).*"


class DockerOnnxQuantTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model.onnx"
        self.model.write_bytes(b"onnx")
        self.images = self.root / "images"
        self.images.mkdir()
        (self.images / "sample.jpg").write_bytes(b"jpg")
        self.mounts = [DockerMount(self.root, PurePosixPath("/workspace"), True)]

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_catalog_defaults_match_verified_environment(self) -> None:
        item = FunctionCatalog(self.root / "functions.json").get("docker_onnx_quant")
        self.assertIsNotNone(item)
        defaults = {field["id"]: field["default"] for field in item["parameters"]}
        self.assertEqual(defaults["container_name"], "onnx2quant")
        self.assertEqual(defaults["conda_environment"], "onnxquant")
        self.assertEqual(defaults["limit"], 128)
        self.assertEqual(defaults["exclude_modules"], "4,5,6,10,13,16,17,23")
        field = next(field for field in item["parameters"] if field["id"] == "exclude_modules")
        self.assertEqual(field["options"], [*EXCLUDE_MODULE_PRESETS, CUSTOM_EXCLUDE_PRESET])

    def test_existing_catalog_adds_preset_and_preserves_custom_pattern(self) -> None:
        storage = self.root / "functions.json"
        item = FunctionCatalog(storage).get("docker_onnx_quant")
        item["parameters"] = [p for p in item["parameters"] if p["id"] != "exclude_modules"]
        custom = next(p for p in item["parameters"] if p["id"] == "nodes_to_exclude")
        custom.update(default="my-node", visibleWhen={"fieldId": "operation", "equals": "导出量化 ONNX"})
        storage.write_text(json.dumps([item]), encoding="utf-8")
        for _ in range(2):
            fields = FunctionCatalog(storage).get("docker_onnx_quant")["parameters"]
            self.assertEqual(sum(p["id"] == "exclude_modules" for p in fields), 1)
            custom = next(p for p in fields if p["id"] == "nodes_to_exclude")
            self.assertEqual(custom["default"], "my-node")
            self.assertEqual(custom["visibleWhen"]["equals"], CUSTOM_EXCLUDE_PRESET)

    def test_host_paths_map_to_bind_mount(self) -> None:
        self.assertEqual(_container_path(self.model, self.mounts), "/workspace/model.onnx")
        outside = self.root.parent / "outside.onnx"
        with self.assertRaisesRegex(ValueError, "不在容器的挂载目录"):
            _container_path(outside, self.mounts)

    def test_export_configuration_and_command(self) -> None:
        context = RunContext(
            function_id="docker_onnx_quant",
            paths={"input_onnx": self.model, "calibration_images": self.images},
            parameters={"model_type": "yolo", "limit": 128, "nodes_to_exclude": LEGACY_EXCLUDE},
            report=lambda _message: None,
        )
        config = _validate_export(
            context, self.mounts, "onnx2quant", "onnxquant", "/root/anaconda3/bin/conda"
        )
        arguments = _tool_arguments(config)
        self.assertEqual(config["output_model"], self.root / "model_quant.onnx")
        self.assertEqual(config["container_images"], "/workspace/images")
        self.assertEqual(arguments[arguments.index("--limit") + 1], "128")
        self.assertEqual(arguments[arguments.index("--nodes_to_exclude") + 1], LEGACY_EXCLUDE)

        for preset in EXCLUDE_MODULE_PRESETS:
            with self.subTest(preset=preset):
                context.parameters["exclude_modules"] = preset
                config = _validate_export(context, self.mounts, "onnx2quant", "onnxquant", "/conda")
                arguments = _tool_arguments(config)
                rule = re.compile(arguments[arguments.index("--nodes_to_exclude") + 1])
                self.assertEqual([str(i) for i in range(24) if rule.fullmatch(f"/model.{i}/conv/Conv")], preset.split(","))
                self.assertIsNone(rule.fullmatch("/model.40/conv/Conv"))

        context.parameters.clear()
        config = _validate_export(context, self.mounts, "onnx2quant", "onnxquant", "/conda")
        self.assertEqual(config["nodes_to_exclude"], [rf"^/model\.({DEFAULT_EXCLUDE_PRESET.replace(',', '|')})(?:/|$).*"])
        context.parameters.update(exclude_modules=CUSTOM_EXCLUDE_PRESET, nodes_to_exclude="^one$\n^two$")
        config = _validate_export(context, self.mounts, "onnx2quant", "onnxquant", "/conda")
        self.assertEqual(config["nodes_to_exclude"], ["^one$", "^two$"])

    def test_environment_check_does_not_require_model_paths(self) -> None:
        messages: list[str] = []
        with patch("backend.docker_onnx_quant._ensure_container", return_value=False), patch(
            "backend.docker_onnx_quant._test_environment", return_value="Python=/env/bin/python"
        ):
            result = run_docker_onnx_quant(RunContext(
                function_id="docker_onnx_quant", paths={},
                parameters={"operation": "测试 Docker 环境"}, report=messages.append,
            ))
        self.assertEqual(result["container"], "onnx2quant")
        self.assertTrue(any("量化环境检查成功" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
