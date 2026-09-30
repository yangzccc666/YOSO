from __future__ import annotations

import tempfile
import textwrap
import tomllib
import unittest
from pathlib import Path

from backend.catalog import FunctionCatalog, LOCAL_STAR_PACKAGE_FUNCTION
from backend.handlers import RunContext, has_handler
from backend.local_star_package import _labels, _render_template, detect_plan_labels, run_local_star_package


TEMPLATE = """title = "old-title" # keep title comment

[hardware]
name = "TensorRT"
architecture = "Turing"
driver = "CUDA12.2"

[model]
name = "old-model"
category = "generic"
version = "1"
model_file = "/old/model.plan"
labels = ["old"]
precision = 1

[[model.input_tensor]]
n = 1
c = 3
h = 640
w = 640
color_fmt = 2
"""


class LocalStarPackageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.tool_folder = self.root / "tool"
        self.tool_folder.mkdir()
        self.model_folder = self.root / "models"
        self.model_folder.mkdir()
        self.plan = self.model_folder / "sample_quant.plan"
        self.plan.write_bytes(b"plan")
        self.template = self.tool_folder / "trt.toml"
        self.template.write_text(TEMPLATE, encoding="utf-8")
        self.tool = self.tool_folder / "package_tool"
        self.tool.write_text(textwrap.dedent("""\
            #!/usr/bin/env python3
            import sys
            import tomllib
            from pathlib import Path

            with Path(sys.argv[1]).open("rb") as stream:
                config = tomllib.load(stream)
            output = Path.cwd() / f"{config['title']}.TensorRT.Turing.INT8.test.star"
            output.write_bytes(b"star-model")
            print(f"packaged {config['title']} with {len(config['model']['labels'])} labels")
        """), encoding="utf-8")
        self.tool.chmod(0o755)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def context(self, **overrides: object) -> RunContext:
        parameters: dict[str, object] = {
            "title": "factory-line-a",
            "labels": "screw\nboard，connector,screw",
            "package_tool_path": str(self.tool),
            "template_file": str(self.template),
            "hardware_name": "TensorRT",
            "architecture": "Turing",
            "driver": "CUDA12.2",
            "model_name": "yolo26",
            "category": "generic",
            "version": "1",
            "precision": "INT8 (1)",
            "input_n": 1,
            "input_c": 3,
            "input_h": 640,
            "input_w": 640,
            "color_format": "RGB (2)",
        }
        parameters.update(overrides)
        self.messages: list[str] = []
        return RunContext(
            function_id="local_star_package",
            paths={"model_file": self.plan},
            parameters=parameters,
            report=self.messages.append,
        )

    def test_package_runs_in_tool_folder_and_moves_star_next_to_plan(self) -> None:
        original_template = self.template.read_text(encoding="utf-8")

        result = run_local_star_package(self.context())

        output = Path(result["outputFiles"][0])
        self.assertEqual(output.parent, self.plan.parent)
        self.assertTrue(output.is_file())
        self.assertEqual(output.read_bytes(), b"star-model")
        self.assertFalse((self.tool_folder / output.name).exists())
        self.assertEqual(self.template.read_text(encoding="utf-8"), original_template)
        self.assertEqual(list(self.tool_folder.glob(".yolo-package-*.toml")), [])
        self.assertEqual(result["labelCount"], 3)
        self.assertEqual(result["remoteInferenceDefaults"], {
            "modelFile": str(output),
            "labels": ["screw", "board", "connector"],
        })
        self.assertTrue(any("packaged factory-line-a with 3 labels" in line for line in self.messages))

    def test_template_render_changes_supported_values_and_preserves_comment(self) -> None:
        config = {
            "title": "new-title", "hardware_name": "TensorRT", "architecture": "Ampere",
            "driver": "CUDA12.4", "model_name": "yolo26", "category": "generic",
            "version": "2", "plan": self.plan, "labels": ["a", "中文标签"], "precision": 3,
            "n": 1, "c": 3, "h": 960, "w": 960, "color_fmt": 2,
        }
        rendered = _render_template(TEMPLATE, config)
        parsed = tomllib.loads(rendered)

        self.assertEqual(parsed["title"], "new-title")
        self.assertEqual(parsed["hardware"]["architecture"], "Ampere")
        self.assertEqual(parsed["model"]["model_file"], str(self.plan))
        self.assertEqual(parsed["model"]["labels"], ["a", "中文标签"])
        self.assertEqual(parsed["model"]["input_tensor"][0]["h"], 960)
        self.assertIn("# keep title comment", rendered)

    def test_labels_accept_lines_and_commas_and_remove_duplicates(self) -> None:
        self.assertEqual(_labels("one\ntwo，three,one"), ["one", "two", "three"])
        with self.assertRaisesRegex(ValueError, "至少填写一个类别名"):
            _labels("  \n ， ")

    def test_detect_plan_labels_reads_indexed_classes_from_parent_folder(self) -> None:
        classes = self.root / "classes.txt"
        classes.write_text("0: screw\n1: board\n2: connector\n", encoding="utf-8")

        result = detect_plan_labels(self.plan)

        self.assertEqual(result["labels"], ["screw", "board", "connector"])
        self.assertEqual(result["source"], str(classes.resolve()))
        self.assertEqual(result["count"], 3)

    def test_detect_plan_labels_reads_yaml_names_in_class_index_order(self) -> None:
        data_yaml = self.model_folder / "data.yaml"
        data_yaml.write_text("train: train.txt\nnames:\n  1: defect\n  0: ok\nnc: 2\n", encoding="utf-8")

        result = detect_plan_labels(self.plan)

        self.assertEqual(result["labels"], ["ok", "defect"])
        self.assertEqual(result["source"], str(data_yaml.resolve()))

    def test_detect_plan_labels_explains_that_plan_has_no_semantic_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "PLAN 文件本身通常不保存类别名称.*classes.txt"):
            detect_plan_labels(self.plan)

    def test_catalog_contains_ready_local_package_function(self) -> None:
        catalog = FunctionCatalog(self.root / "functions.json")
        item = catalog.get("local_star_package")
        self.assertIsNotNone(item)
        self.assertEqual(item["handlerId"], "model.package_star")
        self.assertEqual(item["pathFields"][0]["id"], "model_file")
        self.assertEqual(item["name"], "本地 STAR 模型打包")
        self.assertEqual(item["parameters"][0]["id"], "title")
        self.assertTrue(has_handler("model.package_star"))

    def test_existing_catalog_is_migrated_without_losing_items(self) -> None:
        storage = self.root / "runtime" / "functions.json"
        storage.parent.mkdir()
        storage.write_text('[{"id":"custom","name":"自定义","description":"","handlerId":null,"pathFields":[],"parameters":[]}]', encoding="utf-8")

        catalog = FunctionCatalog(storage)

        self.assertIsNotNone(catalog.get("custom"))
        self.assertIsNotNone(catalog.get(LOCAL_STAR_PACKAGE_FUNCTION["id"]))
        self.assertTrue(storage.with_name("functions.json.local-star-package-v1").exists())


if __name__ == "__main__":
    unittest.main()
