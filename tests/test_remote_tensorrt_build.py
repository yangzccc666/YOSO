from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import MagicMock, patch

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext, has_handler
from backend.remote_tensorrt_build import (
    DEFAULT_TRTEXEC,
    _configuration,
    _resolved_remote_root,
    _trtexec_arguments,
    _validate_remote_jobs,
    run_remote_tensorrt_build,
)
from backend.task_manager import PlatformTaskManager


class RemoteTensorRTBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.model = self.root / "model_quant.onnx"
        self.model.write_bytes(b"onnx")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _context(self, **parameters: object) -> RunContext:
        return RunContext(
            function_id="remote_tensorrt_build",
            paths={"input_onnx": self.model, "output_folder": self.root},
            parameters={"build_mode": "自动最佳（--best）", **parameters},
            report=lambda _message: None,
        )

    def test_catalog_defaults_and_handler_registration(self) -> None:
        item = FunctionCatalog(self.root / "functions.json").get("remote_tensorrt_build")
        self.assertIsNotNone(item)
        self.assertTrue(has_handler(item["handlerId"]))
        defaults = {field["id"]: field["default"] for field in item["parameters"]}
        self.assertEqual(defaults["trtexec_path"], DEFAULT_TRTEXEC)
        self.assertEqual(defaults["host"], "192.168.1.223")
        self.assertTrue(defaults["remember_password"])
        self.assertEqual(item["pathFields"], [])
        self.assertNotIn("output_filename", defaults)

    def test_configuration_and_trtexec_arguments(self) -> None:
        config = _configuration(self._context(run_benchmark=False), require_paths=True)
        job = config["jobs"][0]
        self.assertEqual(job["output_plan"], self.root / "model_quant.plan")
        self.assertEqual(job["output_log"], self.root / "model_quant.trtexec.log")
        arguments = _trtexec_arguments(config, "/remote/input.onnx", "/remote/output.plan")
        self.assertIn("--best", arguments)
        self.assertIn("--skipInference", arguments)
        self.assertIn("--onnx=/remote/input.onnx", arguments)

    def test_remote_validation_requires_source_and_protects_existing_output(self) -> None:
        config = _configuration(self._context(), require_paths=True)
        source = str(config["jobs"][0]["source"])
        plan = str(config["jobs"][0]["output_plan"])
        sftp = MagicMock()

        def remote_stat(path: str):
            if path == source:
                attributes = MagicMock()
                attributes.st_size = 123
                return attributes
            if path == plan:
                return MagicMock()
            raise OSError(path)

        sftp.stat.side_effect = remote_stat
        client = MagicMock()
        client.open_sftp.return_value = sftp
        with self.assertRaisesRegex(ValueError, "远端输出已存在"):
            _validate_remote_jobs(client, config, lambda _message: None)

        overwrite_config = _configuration(self._context(overwrite_output=True), require_paths=True)
        _validate_remote_jobs(client, overwrite_config, lambda _message: None)

    def test_multiple_models_preserve_selected_order(self) -> None:
        second = self.root / "second_quant.onnx"
        second.write_bytes(b"onnx-2")
        config = _configuration(self._context(
            onnx_files=f"{second}\n{self.model}\n",
        ), require_paths=True)
        self.assertEqual([job["source"] for job in config["jobs"]], [second, self.model])
        self.assertEqual(
            [job["output_plan"].name for job in config["jobs"]],
            ["second_quant.plan", "model_quant.plan"],
        )

    def test_saved_json_model_list_remains_compatible(self) -> None:
        second = self.root / "second_quant.onnx"
        second.write_bytes(b"onnx-2")
        config = _configuration(self._context(
            onnx_files=json.dumps([str(self.model), str(second)]),
        ), require_paths=True)
        self.assertEqual([job["source"] for job in config["jobs"]], [self.model, second])

    def test_remote_model_path_must_be_absolute(self) -> None:
        with self.assertRaisesRegex(ValueError, "远程服务器上的 .onnx 绝对路径"):
            _configuration(self._context(onnx_files="relative/model.onnx"), require_paths=True)

    def test_duplicate_remote_paths_are_only_built_once(self) -> None:
        config = _configuration(self._context(
            onnx_files=f"{self.model}\n{self.model}\n",
        ), require_paths=True)
        self.assertEqual(len(config["jobs"]), 1)

    def test_remote_workspace_expands_against_ssh_home(self) -> None:
        sftp = MagicMock()
        sftp.normalize.return_value = "/home/wel"
        result = _resolved_remote_root(sftp, "~/.local/state/yolo-processing/tensorrt")
        self.assertEqual(result, PurePosixPath("/home/wel/.local/state/yolo-processing/tensorrt"))

    def test_environment_check_operation_does_not_require_model(self) -> None:
        client = MagicMock()
        messages: list[str] = []
        context = RunContext(
            function_id="remote_tensorrt_build", paths={},
            parameters={"operation": "检查远程环境", "password": "saved"},
            report=messages.append,
        )
        with patch("backend.remote_tensorrt_build._connect", return_value=(client, {
            "host": "192.168.1.223", "port": 22, "username": "wel", "password": "",
        })), patch("backend.remote_tensorrt_build._check_environment", return_value="TensorRT 10.3"):
            result = run_remote_tensorrt_build(context)
        self.assertIn("TensorRT 10.3", result["message"])
        self.assertEqual(context.parameters["password"], "")
        client.close.assert_called_once()

    def test_batch_run_executes_each_model_sequentially(self) -> None:
        second = self.root / "second_quant.onnx"
        second.write_bytes(b"onnx-2")
        messages: list[str] = []
        context = self._context(
            operation="构建 TensorRT 引擎",
            onnx_files=f"{second}\n{self.model}",
            password="saved",
        )
        context.report = messages.append
        notifications: list[tuple[str, str]] = []
        context.notify = lambda title, message: notifications.append((title, message))
        client = MagicMock()
        sftp = MagicMock()
        sftp.normalize.return_value = "/home/wel"
        client.open_sftp.return_value = sftp
        order: list[str] = []

        def run_one(_client, _root, _config, job, _context, index, total):
            order.append(job["source"].name)
            return {
                "source": str(job["source"]), "plan": str(job["output_plan"]),
                "log": str(job["output_log"]), "remoteDirectory": "", "remotePlan": "",
                "engineSize": index * 10,
            }

        with patch("backend.remote_tensorrt_build._connect", return_value=(client, {
            "host": "192.168.1.223", "port": 22, "username": "wel", "password": "",
        })), patch("backend.remote_tensorrt_build._check_environment", return_value="TensorRT 10.3"), \
                patch("backend.remote_tensorrt_build._validate_remote_jobs"), \
                patch("backend.remote_tensorrt_build._run_one_job", side_effect=run_one):
            result = run_remote_tensorrt_build(context)
        self.assertEqual(order, ["second_quant.onnx", "model_quant.onnx"])
        self.assertEqual(len(result["jobs"]), 2)
        self.assertIn("2 个模型", result["message"])
        self.assertTrue(any("按填写顺序" in message for message in messages))
        self.assertEqual(len(notifications), 2)
        self.assertIn("second_quant.onnx 转换完成", notifications[0][1])
        self.assertIn("model_quant.onnx 转换完成", notifications[1][1])

    def test_remote_build_survives_platform_shutdown_and_exposes_live_logs(self) -> None:
        manager = PlatformTaskManager()
        task = manager.start("remote_tensorrt_build", "TensorRT 构建", "remote-build")
        task.add_log("Engine generation started")
        self.assertEqual(manager.active()[0]["logs"], ["Engine generation started"])
        manager.stop_all()
        self.assertFalse(task.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
