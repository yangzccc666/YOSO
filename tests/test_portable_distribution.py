from __future__ import annotations

import json
import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app_settings import APP_DEFINITIONS, AppSettingsStore
from backend.catalog import YOLO_TRAINING_FUNCTION
from backend.distribution import dependency_report
from backend.local_pt_inference import _find_yolo_python
from backend.worker_runtime import restore_worker_stdio, run_worker, worker_command
from backend.yolo_training import DEFAULT_TRAINING_VALUES, _check_remote_training_paths, _remote_training_shell, normalize_training_values


class PortableDistributionTests(unittest.TestCase):
    def test_fresh_training_defaults_have_no_owner_server_or_source(self):
        self.assertEqual(DEFAULT_TRAINING_VALUES["ultralytics_source"], "")
        self.assertEqual(DEFAULT_TRAINING_VALUES["model"], "yolo26n.pt")
        defaults = {p["id"]: p["default"] for p in YOLO_TRAINING_FUNCTION["parameters"]}
        self.assertEqual(defaults["remote_host"], "")
        self.assertEqual(defaults["remote_username"], "")
        for value in APP_DEFINITIONS.values():
            self.assertEqual(value["executable"], "")
            self.assertEqual(value["condaEnvironment"], "")

    def test_saved_training_source_and_model_are_no_longer_overwritten(self):
        values = normalize_training_values({"device": "cpu", "ultralytics_source": "/srv/custom-source", "model": "/srv/weights/custom.pt"})
        self.assertEqual(values["ultralytics_source"], "/srv/custom-source")
        self.assertEqual(values["model"], "/srv/weights/custom.pt")
        shell, _ = _remote_training_shell(["yolo", "detect", "train", "model=custom.pt"], "task", values["ultralytics_source"])
        self.assertIn("/srv/custom-source", shell)

    def test_blank_training_source_uses_installed_package_without_empty_path_check(self):
        values = normalize_training_values({"device": "cpu"})
        commands = []
        with patch("backend.yolo_training._remote_capture", side_effect=lambda _client, command: (commands.append(command) or (0, "", ""))):
            _check_remote_training_paths(object(), values)
        self.assertFalse(any("pyproject.toml" in command for command in commands))
        shell, _ = _remote_training_shell(["yolo", "detect", "train", "model=yolo26n.pt"], "task", "")
        self.assertNotIn("PYTHONPATH=", shell)

    def test_settings_keep_optional_ai_python_without_changing_external_apps(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / "settings.json")
            first = store.update({"externalApps": [], "aiPython": "/opt/custom/python"})
            self.assertEqual(first["aiPython"], "/opt/custom/python")
            self.assertEqual(store.update({"externalApps": []})["aiPython"], "/opt/custom/python")
            self.assertEqual(AppSettingsStore(store.storage_file).payload()["aiPython"], "/opt/custom/python")
            with self.assertRaises(ValueError):
                store.update({"externalApps": [], "aiPython": "bad\npath"})

    def test_ai_runtime_supports_current_python_without_conda(self):
        with patch.dict(os.environ, {"PROCESSING_VIEW_YOLO_PYTHON": ""}), \
             patch("backend.app_settings.settings_store.payload", return_value={"aiPython": ""}), \
             patch("backend.local_pt_inference.importlib.util.find_spec", return_value=object()):
            self.assertEqual(_find_yolo_python(), Path(sys.executable).resolve())

    def test_ai_runtime_accepts_saved_arbitrary_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "not-conda" / "python"
            python.parent.mkdir()
            python.touch()
            with patch.dict(os.environ, {"PROCESSING_VIEW_YOLO_PYTHON": ""}), \
                 patch("backend.app_settings.settings_store.payload", return_value={"aiPython": str(python)}):
                self.assertEqual(_find_yolo_python(), python.resolve())

    def test_source_worker_uses_python_script_but_frozen_worker_uses_app_dispatch(self):
        python = Path(sys.executable).resolve()
        script = Path(__file__).resolve().parents[1] / "backend/bundled_export_onnx_cut.py"
        with patch.object(sys, "frozen", False, create=True):
            self.assertEqual(worker_command(python, script, ["--task", "detect"]), [str(python), "-u", "-B", str(script), "--task", "detect"])
        with patch.object(sys, "frozen", True, create=True):
            self.assertEqual(worker_command(python, script, ["--task", "detect"]), [sys.executable, "--internal-worker", "bundled_export_onnx_cut", "--task", "detect"])
            with self.assertRaises(ValueError):
                worker_command(python, Path("arbitrary.py"), [])

    def test_internal_worker_only_dispatches_whitelisted_modules(self):
        with patch.object(sys, "argv", ["test"]), patch("backend.worker_runtime.runpy.run_module") as run:
            run_worker(["auto_annotation_worker", "config.json"])
            self.assertEqual(sys.argv, ["auto_annotation_worker", "config.json"])
            run.assert_called_once_with("backend.auto_annotation_worker", run_name="__main__")
        with self.assertRaises(ValueError):
            run_worker(["../../other"])

    def test_windowed_worker_restores_redirected_stdio_when_none(self):
        stream = object()
        with patch.object(sys, "stdin", None), patch.object(sys, "stdout", None), patch.object(sys, "stderr", None), \
             patch("builtins.open", return_value=stream) as opened:
            restore_worker_stdio()
            self.assertIs(sys.stdout, stream)
            self.assertIs(sys.stdin, stream)
            self.assertEqual(opened.call_count, 3)

    def test_dependency_report_is_honest_about_missing_and_external_packages(self):
        with patch("backend.distribution.module_present", return_value=False), \
             patch("backend.video_clip._find_ffmpeg", return_value="bundled-ffmpeg"), \
             patch("backend.distribution.shutil.which", return_value=None):
            report = dependency_report()
        checks = {row["name"]: row for row in report["checks"]}
        self.assertEqual(checks["本地 PT / ONNX 推理／反向标注／YOLO26 导出"]["status"], "missing")
        self.assertEqual(checks["STAR 第三方打包工具"]["status"], "external")
        self.assertIn("不自动安装", report["note"])
        json.dumps(report, ensure_ascii=False)

    def test_ai_build_rejects_no_license_acknowledgement_or_missing_packages(self):
        spec = Path(__file__).resolve().parents[1] / "packaging/yolo_data_platform.spec"
        with patch.dict(os.environ, {"YOLO_BUILD_PROFILE": "ai", "YOLO_AI_LICENSE_ACKNOWLEDGED": ""}), self.assertRaisesRegex(SystemExit, "licenses"):
            runpy.run_path(str(spec), init_globals={"SPECPATH": str(spec.parent)})
        with patch.dict(os.environ, {"YOLO_BUILD_PROFILE": "ai", "YOLO_AI_LICENSE_ACKNOWLEDGED": "1"}), \
             patch("importlib.util.find_spec", return_value=None), self.assertRaisesRegex(SystemExit, "missing packages"):
            runpy.run_path(str(spec), init_globals={"SPECPATH": str(spec.parent)})


if __name__ == "__main__":
    unittest.main()
