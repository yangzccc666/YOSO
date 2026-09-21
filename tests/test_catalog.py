from __future__ import annotations

import tempfile
import json
import unittest
import shutil
import os
import shlex
import stat
import sys
import threading
import time
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend import bundled_run_star_video, remote_yolo_dataset, remote_yolo_dataset_worker
from backend.catalog import FunctionCatalog
from backend.credential_store import CredentialStore
from backend.groups import GroupCatalog
from backend.handlers import RunContext, execute, has_handler, register_handler
from backend.remote_inference import BUNDLED_INFERENCE_SCRIPT, REMEMBERED_PASSWORD_MARKER, RemoteInferenceManager, validate_connection_payload, validate_start_payload
from backend.task_manager import PlatformTaskManager, TaskCancelled
from backend.video_clip import COPY_MODE, _configured_range, parse_time, run_video_clip
from backend.video_frames import run_video_frames
from backend.yolo_training import DEFAULT_TRAINING_VALUES, TrainingManager, TrainingProfileStore, TrainingSession, _conda_environment_for_yolo, _follow_remote_session, _remote_output_lines, _remote_training_shell, build_training_command, select_remote_yolo_candidate, training_task_key
from backend.yolo_dataset_split import run_yolo_dataset_split
from desktop import configure_linux_input_method


class CatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.catalog = FunctionCatalog(Path(self.temp.name) / "functions.json")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_default_catalog_contains_video_frames(self) -> None:
        items = self.catalog.list()
        item = items[0]
        self.assertEqual(item["name"], "视频切图")
        self.assertEqual(item["handlerId"], "video.extract_frames")
        max_images = next(parameter for parameter in item["parameters"] if parameter["id"] == "max_images_limit")
        self.assertEqual(max_images["default"], "")
        start_time = next(parameter for parameter in item["parameters"] if parameter["id"] == "range_start_time")
        end_time = next(parameter for parameter in item["parameters"] if parameter["id"] == "range_end_time")
        self.assertEqual(start_time["default"], "00:00:00")
        self.assertEqual(end_time["default"], "")
        strategy = next(parameter for parameter in item["parameters"] if parameter["id"] == "limit_strategy")
        self.assertEqual(strategy["default"], "全程均匀抽取")
        self.assertEqual(items[1]["name"], "YOLO 数据集划分")
        self.assertEqual(items[1]["handlerId"], "yolo.split_dataset")
        task_type = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "dataset_task_type")
        self.assertEqual(task_type["default"], "目标检测")
        self.assertEqual(task_type["options"], ["目标检测", "旋转目标检测", "分割", "分类"])
        label_format = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "label_format")
        self.assertEqual(label_format["default"], "VOC XML")
        dataset_folder_name = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "dataset_folder_name")
        self.assertEqual(dataset_folder_name["default"], "yolo_train")
        parameter_ids = {parameter["id"] for parameter in items[1]["parameters"]}
        remote_host = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "remote_host")
        remote_username = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "remote_username")
        remote_password = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "remote_password")
        remember_yolo_password = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "remember_password")
        self.assertNotIn("execution_location", parameter_ids)
        self.assertEqual(remote_host["default"], "")
        self.assertEqual(remote_username["default"], "")
        self.assertEqual(remote_password["type"], "password")
        self.assertEqual(remote_password["default"], "")
        self.assertTrue(remember_yolo_password["default"])
        self.assertEqual(items[2]["name"], "视频裁剪")
        self.assertEqual(items[2]["handlerId"], "video.clip")
        clip_end_time = next(parameter for parameter in items[2]["parameters"] if parameter["id"] == "range_end_time")
        self.assertEqual(clip_end_time["default"], "")
        self.assertIn("留空则到视频结尾", clip_end_time["label"])
        clip_encoding = next(parameter for parameter in items[2]["parameters"] if parameter["id"] == "encoding_mode")
        self.assertEqual(clip_encoding["default"], COPY_MODE)
        self.assertEqual(items[3]["name"], "远程实时 AI 推理")
        self.assertEqual(items[3]["handlerId"], "remote.star_inference")
        password = next(parameter for parameter in items[3]["parameters"] if parameter["id"] == "password")
        remember_password = next(parameter for parameter in items[3]["parameters"] if parameter["id"] == "remember_password")
        labels = next(parameter for parameter in items[3]["parameters"] if parameter["id"] == "labels")
        self.assertEqual(password["type"], "password")
        self.assertEqual(password["default"], "")
        self.assertEqual(remember_password["type"], "boolean")
        self.assertTrue(remember_password["default"])
        self.assertEqual(labels["default"], "")
        self.assertEqual(
            [field["id"] for field in items[3]["pathFields"]],
            ["local_script_file", "local_model_file", "local_video_file"],
        )
        self.assertNotIn("script_path", {parameter["id"] for parameter in items[3]["parameters"]})

    def test_remote_inference_configuration_and_command_do_not_expose_password(self) -> None:
        model = Path(self.temp.name) / "sample.star"
        video = Path(self.temp.name) / "test video.mp4"
        script = Path(self.temp.name) / "run_star_video.py"
        model.write_bytes(b"model")
        video.write_bytes(b"video")
        script.write_text("# test inference script\n", encoding="utf-8")
        config = validate_start_payload({
            "host": "192.168.1.223",
            "port": 22,
            "username": "wel",
            "password": "temporary-secret",
            "python_path": "/opt/venv/bin/python",
            "labels": "part,defect",
            "conf": 0.2,
            "iou": 0.45,
            "max_det": 100,
            "realtime": True,
        }, {
            "local_script_file": str(script),
            "local_model_file": str(model),
            "local_video_file": str(video),
        })
        config["script_path"] = "/tmp/yolo job/run_star_video.py"
        config["remote_model_file"] = "/tmp/yolo job/model.star"
        config["remote_video_file"] = "/tmp/yolo job/input video.mp4"
        command = RemoteInferenceManager._command(
            config,
            "/tmp/yolo worker.py",
            "yolo-session-test",
        )
        self.assertNotIn("temporary-secret", command)
        self.assertIn("'/tmp/yolo job/run_star_video.py'", command)
        self.assertIn("'/tmp/yolo job/input video.mp4'", command)
        self.assertIn(" -B ", command)
        self.assertIn("--realtime", command)

    def test_remote_connection_can_be_checked_before_selecting_files(self) -> None:
        config = validate_connection_payload({
            "host": "192.168.1.223",
            "port": 22,
            "username": "wel",
            "password": "temporary-secret",
        })
        self.assertEqual(config["host"], "192.168.1.223")
        self.assertEqual(config["username"], "wel")

    def test_remembered_password_is_encrypted_and_can_be_cleared(self) -> None:
        store = CredentialStore(Path(self.temp.name) / "credentials")
        store.set("192.168.1.223", 22, "wel", "temporary-secret")
        stored_text = store.data_file.read_text(encoding="utf-8")
        self.assertNotIn("temporary-secret", stored_text)
        self.assertEqual(stat.S_IMODE(store.key_file.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(store.data_file.stat().st_mode), 0o600)

        config = validate_connection_payload({
            "host": "192.168.1.223",
            "port": 22,
            "username": "wel",
            "password": REMEMBERED_PASSWORD_MARKER,
            "remember_password": True,
        }, store)
        self.assertEqual(config["password"], "temporary-secret")
        self.assertTrue(config["password_from_store"])

        manager = RemoteInferenceManager(store)
        credential_status = manager.credential_status({
            "host": "192.168.1.223",
            "port": 22,
            "username": "wel",
        })
        self.assertTrue(credential_status["remembered"])
        self.assertNotIn("password", credential_status)
        self.assertTrue(store.delete("192.168.1.223", 22, "wel"))
        self.assertIsNone(store.get("192.168.1.223", 22, "wel"))

    def test_remote_connection_errors_have_actionable_messages(self) -> None:
        import paramiko

        auth_message = RemoteInferenceManager._connection_error(
            paramiko.AuthenticationException("bad credentials"),
            paramiko,
            "192.168.1.223",
            22,
        )
        refused_message = RemoteInferenceManager._connection_error(
            paramiko.ssh_exception.NoValidConnectionsError({
                ("192.168.1.223", 22): ConnectionRefusedError("refused"),
            }),
            paramiko,
            "192.168.1.223",
            22,
        )
        self.assertIn("用户名和密码", auth_message)
        self.assertIn("连接被拒绝", refused_message)

    def test_remote_inference_requires_local_model_and_video_files(self) -> None:
        script = Path(self.temp.name) / "run_star_video.py"
        script.write_text("# test inference script\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "请选择本地 STAR 模型文件"):
            validate_start_payload({
                "host": "192.168.1.223",
                "username": "wel",
                "password": "temporary-secret",
                "labels": "person",
            }, {"local_script_file": str(script)})

    def test_remote_inference_uses_bundled_script_when_not_selected(self) -> None:
        model = Path(self.temp.name) / "sample.star"
        video = Path(self.temp.name) / "sample.mp4"
        model.write_bytes(b"model")
        video.write_bytes(b"video")
        config = validate_start_payload({
            "host": "192.168.1.223",
            "username": "wel",
            "password": "temporary-secret",
            "labels": "person",
        }, {
            "local_model_file": str(model),
            "local_video_file": str(video),
        })
        self.assertEqual(config["local_script_file"], BUNDLED_INFERENCE_SCRIPT)

    def test_remote_inference_accepts_multiple_models_with_independent_labels_and_thresholds(self) -> None:
        first = Path(self.temp.name) / "first.star"
        second = Path(self.temp.name) / "second.star"
        video = Path(self.temp.name) / "video.mp4"
        for path in (first, second, video):
            path.write_bytes(b"sample")
        config = validate_start_payload({
            "host": "192.168.1.223", "username": "wel", "password": "temporary-secret",
            "star_models": json.dumps([
                {"path": str(first), "labels": "person", "conf": 0.5},
                {"path": str(second), "labels": "car,bus", "conf": 0.7},
            ]),
        }, {"local_model_file": str(first), "local_video_file": str(video)})
        self.assertEqual(len(config["models"]), 2)
        self.assertEqual(config["models"][1]["labels"], "car,bus")
        self.assertEqual(config["models"][1]["conf"], 0.7)
        config.update(script_path="/tmp/script.py", remote_model_file="/tmp/model_1.star", remote_video_file="/tmp/video.mp4",
                      remote_models=[{"path": "/tmp/model_1.star", "labels": "person", "conf": 0.5},
                                     {"path": "/tmp/model_2.star", "labels": "car,bus", "conf": 0.7}])
        command = RemoteInferenceManager._command(config, "/tmp/worker.py", "token")
        self.assertIn("--models-json", command)
        self.assertIn("model_2.star", command)
        self.assertNotIn("temporary-secret", command)

    def test_star_model_uses_its_category_names_without_a_custom_display_name(self) -> None:
        model = Path(self.temp.name) / "opaque-export-name.star"
        video = Path(self.temp.name) / "video.mp4"
        model.write_bytes(b"sample")
        video.write_bytes(b"sample")
        config = validate_start_payload({
            "host": "device", "username": "wel", "password": "secret",
            "star_models": json.dumps([{"path": str(model), "labels": "checkbox,damaged", "conf": 0.5}]),
        }, {"local_video_file": str(video)})
        self.assertEqual(config["models"][0]["labels"], "checkbox,damaged")
        self.assertNotIn("name", config["models"][0])

    def test_remote_inference_rejects_incomplete_second_model(self) -> None:
        first = Path(self.temp.name) / "first.star"
        video = Path(self.temp.name) / "video.mp4"
        first.write_bytes(b"sample")
        video.write_bytes(b"sample")
        with self.assertRaisesRegex(ValueError, "第 2 个 STAR 模型文件"):
            validate_start_payload({"host": "device", "username": "wel", "password": "secret",
                                    "star_models": json.dumps([{"path": str(first), "labels":"person"},{"path":"","labels":"car"}])},
                                   {"local_model_file": str(first), "local_video_file": str(video)})

    def test_remote_inference_requires_class_names(self) -> None:
        with self.assertRaisesRegex(ValueError, "请填写类别名称"):
            validate_start_payload({
                "host": "192.168.1.223",
                "username": "wel",
                "password": "temporary-secret",
                "labels": "",
            })

    def test_bundled_inference_prefers_dependency_free_cuda_driver(self) -> None:
        class FakeCtypesBackend:
            name = "cuda-driver-ctypes"

        with patch.object(
            bundled_run_star_video,
            "CtypesCudaDriverBackend",
            FakeCtypesBackend,
        ):
            backend = bundled_run_star_video.create_cuda_backend("auto")
        self.assertEqual(backend.name, "cuda-driver-ctypes")

    def test_remote_inference_requires_a_one_time_password(self) -> None:
        model = Path(self.temp.name) / "sample.star"
        video = Path(self.temp.name) / "sample.mp4"
        model.write_bytes(b"model")
        video.write_bytes(b"video")
        with self.assertRaisesRegex(ValueError, "尚未保存.*SSH 密码"):
            validate_start_payload({
                "host": "192.168.1.223",
                "username": "wel",
            }, {
                "local_model_file": str(model),
                "local_video_file": str(video),
            })

    def test_create_update_and_delete(self) -> None:
        item = self.catalog.create({"name": "我的功能", "pathFields": [], "parameters": []})
        updated = self.catalog.update(item["id"], {**item, "name": "重命名后的功能"})
        self.assertEqual(updated["name"], "重命名后的功能")
        self.catalog.delete(item["id"])
        self.assertIsNone(self.catalog.get(item["id"]))

    def test_linux_fcitx_falls_back_to_bundled_xim_plugin(self) -> None:
        environment = {
            "XDG_SESSION_TYPE": "x11",
            "QT_IM_MODULE": "fcitx",
            "XMODIFIERS": "@im=fcitx",
        }
        with patch.dict(os.environ, environment, clear=True), patch.object(sys, "platform", "linux"):
            self.assertEqual(configure_linux_input_method(), "xim")
            self.assertEqual(os.environ["QT_IM_MODULE"], "xim")

    def test_input_method_override_is_respected(self) -> None:
        with patch.dict(os.environ, {"PROCESSING_VIEW_QT_IM_MODULE": "ibus"}, clear=True):
            self.assertEqual(configure_linux_input_method(), "ibus")
            self.assertEqual(os.environ["QT_IM_MODULE"], "ibus")

    def test_groups_support_crud_reorder_and_safe_delete(self) -> None:
        groups = GroupCatalog(Path(self.temp.name) / "groups.json")
        functions = self.catalog.list()
        decorated = groups.decorate(functions)
        self.assertEqual(next(item for item in decorated if item["id"] == "video_frames")["groupId"], "video_processing")

        created = groups.create("图像处理")
        renamed = groups.rename(created["id"], "图像工具")
        self.assertEqual(renamed["name"], "图像工具")

        ordered_ids = [group["id"] for group in groups.list_groups()]
        groups.reorder_groups(list(reversed(ordered_ids)))
        self.assertEqual(groups.list_groups()[0]["id"], created["id"])

        groups.move_function("video_frames", created["id"], functions)
        moved = groups.decorate(functions)
        self.assertEqual(next(item for item in moved if item["id"] == "video_frames")["groupId"], created["id"])

        groups.delete(created["id"], functions)
        after_delete = groups.decorate(functions)
        self.assertIsNone(next(item for item in after_delete if item["id"] == "video_frames")["groupId"])
        self.assertNotIn(created["id"], {group["id"] for group in groups.list_groups()})

    def test_function_order_can_be_changed_within_a_group(self) -> None:
        groups = GroupCatalog(Path(self.temp.name) / "groups.json")
        functions = self.catalog.list()
        groups.move_function("video_clip", "video_processing", functions, position=0)
        video_items = sorted(
            [item for item in groups.decorate(functions) if item["groupId"] == "video_processing"],
            key=lambda item: item["order"],
        )
        self.assertEqual([item["id"] for item in video_items], ["video_clip", "video_frames"])

    def test_handler_interface(self) -> None:
        register_handler("test.handler", lambda context: {"message": str(context.paths["input"])})
        self.assertTrue(has_handler("test.handler"))
        result = execute("test.handler", RunContext("test", {"input": Path("/tmp")}, {}, lambda _message: None))
        self.assertIn("/tmp", result["message"])

    def test_platform_task_manager_runs_independent_tasks_and_stops_only_selected_task(self) -> None:
        manager = PlatformTaskManager()
        callback_calls: list[str] = []
        task = manager.start("video_frames", "视频切图")
        manager.set_stop_callback(task.id, lambda: callback_calls.append("stopped"))

        training = manager.start("yolo_dataset_split", "YOLO 模型训练", "remote", task_key="yolo-training")
        dataset = manager.start("yolo_dataset_split", "YOLO 数据集划分", "remote")
        self.assertEqual(len(manager.active()), 3)

        with self.assertRaisesRegex(ValueError, "视频切图.*正在运行"):
            manager.start("video_frames", "视频切图")
        with self.assertRaisesRegex(ValueError, "模型训练.*正在运行"):
            manager.start("yolo_dataset_split", "YOLO 模型训练", task_key="yolo-training")

        stopped = manager.stop(task.id)
        self.assertEqual(stopped["status"], "stopping")
        self.assertTrue(task.stop_event.is_set())
        self.assertFalse(training.stop_event.is_set())
        self.assertFalse(dataset.stop_event.is_set())
        self.assertEqual(callback_calls, ["stopped"])
        with self.assertRaises(TaskCancelled):
            RunContext("video_frames", {}, {}, lambda _message: None, task.stop_event).check_cancelled()

        manager.finish(task.id)
        self.assertEqual({entry["id"] for entry in manager.active()}, {training.id, dataset.id})
        manager.stop_all()
        self.assertTrue(training.stop_event.is_set())
        self.assertTrue(dataset.stop_event.is_set())

    def test_shutdown_preserves_remote_tmux_task_but_stops_local_tasks(self) -> None:
        tasks = PlatformTaskManager()
        remote = tasks.start("yolo_dataset_split", "远程训练", "remote-training")
        local = tasks.start("video_frames", "本地视频切图")
        tasks.stop_all()
        self.assertFalse(remote.stop_event.is_set())
        self.assertTrue(local.stop_event.is_set())

        manager = TrainingManager(Path(self.temp.name) / "remote_training_sessions.json")
        session = TrainingSession(id="0123456789abcdef", remote=True, host="192.168.21.5", username="dell",
                                  remote_dir="$HOME/.local/state/yolo-processing/training/0123456789abcdef")
        manager._sessions[session.id] = session
        manager.stop_all()
        self.assertFalse(session.stop_event.is_set())
        manager._save()
        loaded = TrainingManager(manager._storage)
        self.assertEqual(loaded.get(session.id).status, "disconnected")
        self.assertEqual(loaded.get(session.id).remote_dir, session.remote_dir)

    def test_remote_tmux_log_follower_observes_exit_without_killing_session(self) -> None:
        session = TrainingSession(id="0123456789abcdef", remote=True, remote_dir="$HOME/job")
        commands: list[str] = []
        def capture(_client, command):
            commands.append(command)
            if "tail -n" in command:
                return 0, "epoch 1/2\nepoch 2/2", ""
            if "exit.code" in command:
                return 0, "0", ""
            raise AssertionError("Completed training must not receive tmux kill commands")
        messages: list[str] = []
        with patch("backend.yolo_training._remote_capture", side_effect=capture):
            _follow_remote_session(object(), session, messages.append, threading.Event())
        self.assertEqual(messages, ["epoch 1/2", "epoch 2/2"])
        self.assertFalse(any("kill-session" in command for command in commands))

    def test_finished_training_record_can_be_deleted_without_touching_remote_output(self) -> None:
        manager = TrainingManager(Path(self.temp.name) / "training_sessions.json")
        output = Path(self.temp.name) / "server_output"
        output.mkdir()
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(b"weights")
        finished = TrainingSession(id="finished", remote=True, output=str(output), status="stopped",
                                   remote_dir="$HOME/job", message="手动停止")
        active = TrainingSession(id="active", remote=True, output="/remote/active", status="disconnected",
                                 remote_dir="$HOME/active")
        manager._sessions = {finished.id: finished, active.id: active}
        manager._save()
        before_delete = TrainingManager(manager._storage)
        self.assertEqual(before_delete.get(finished.id).status, "stopped")
        with self.assertRaisesRegex(ValueError, "不能删除运行中的任务记录"):
            manager.delete(active.id)
        manager.delete(finished.id)
        self.assertTrue(checkpoint.is_file())
        self.assertNotIn(finished.id, [item["id"] for item in manager.list()])
        restarted = TrainingManager(manager._storage)
        self.assertNotIn(finished.id, restarted._sessions)
        self.assertIn(active.id, restarted._sessions)

    def test_video_frame_extraction_honors_stop_signal(self) -> None:
        source = Path(self.temp.name) / "cancel_sample.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()
        stop_event = threading.Event()
        stop_event.set()

        with self.assertRaisesRegex(TaskCancelled, "用户终止"):
            run_video_frames(RunContext(
                function_id="video_frames",
                paths={"video_file": source},
                parameters={"target_fps": 1},
                report=lambda _message: None,
                stop_event=stop_event,
            ))

    def test_video_frame_extraction_adapter(self) -> None:
        source = Path(self.temp.name) / "sample.mp4"
        output = Path(self.temp.name) / "frames"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(10):
            frame = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
        writer.release()

        messages: list[str] = []
        result = run_video_frames(RunContext(
            function_id="video_frames",
            paths={"video_file": source, "output_folder": output},
            parameters={
                "extraction_mode": "每秒抽取张数",
                "target_fps": 2,
                "fallback_step": 24,
                "max_images_limit": "",
                "max_images": 1,
            },
            report=messages.append,
        ))
        self.assertEqual(result["videoCount"], 1)
        self.assertEqual(result["imageCount"], 2)
        self.assertIsNone(result["maxImagesPerVideo"])
        self.assertFalse(result["details"][0]["limitReached"])
        self.assertTrue(Path(result["outputFolders"][0]).exists())
        self.assertTrue(any("已生成 2 张图片" in message for message in messages))

    def test_video_frame_extraction_stops_at_maximum(self) -> None:
        source = Path(self.temp.name) / "long_sample.mp4"
        output = Path(self.temp.name) / "limited_frames"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(30):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        writer.release()

        messages: list[str] = []
        result = run_video_frames(RunContext(
            function_id="video_frames",
            paths={"video_file": source, "output_folder": output},
            parameters={
                "extraction_mode": "每秒抽取张数",
                "target_fps": 10,
                "fallback_step": 24,
                "max_images": 3,
            },
            report=messages.append,
        ))

        self.assertEqual(result["imageCount"], 3)
        self.assertEqual(result["maxImagesPerVideo"], 3)
        self.assertTrue(result["details"][0]["limitReached"])
        self.assertEqual(result["details"][0]["selectedFrameIndices"], [0, 14, 29])
        self.assertEqual(len(list(Path(result["outputFolders"][0]).glob("*.jpg"))), 3)
        self.assertTrue(any("已达到上限" in message for message in messages))

    def test_video_frame_extraction_respects_absolute_time_range(self) -> None:
        source = Path(self.temp.name) / "trimmed_sample.mp4"
        output = Path(self.temp.name) / "trimmed_frames"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(40):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        writer.release()

        result = run_video_frames(RunContext(
            function_id="video_frames",
            paths={"video_file": source, "output_folder": output},
            parameters={
                "extraction_mode": "每秒抽取张数",
                "target_fps": 10,
                "max_images": 4,
                "range_start_time": "00:01",
                "range_end_time": "00:03",
                "limit_strategy": "全程均匀抽取",
            },
            report=lambda _message: None,
        ))

        selected = result["details"][0]["selectedFrameIndices"]
        self.assertEqual(selected, [10, 16, 23, 29])
        self.assertTrue(all(10 <= frame_index < 30 for frame_index in selected))
        self.assertEqual(result["imageCount"], 4)
        self.assertEqual(result["details"][0]["startTimeSec"], 1)
        self.assertEqual(result["details"][0]["endTimeSec"], 3)

    def test_video_frame_extraction_rejects_reversed_time_range(self) -> None:
        source = Path(self.temp.name) / "invalid_range.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for _index in range(10):
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
        writer.release()

        with self.assertRaisesRegex(ValueError, "结束时间必须晚于开始时间"):
            run_video_frames(RunContext(
                function_id="video_frames",
                paths={"video_file": source},
                parameters={"range_start_time": "02:40", "range_end_time": "01:30"},
                report=lambda _message: None,
            ))

    def test_video_clip_time_parser(self) -> None:
        self.assertEqual(parse_time("90", "时间"), 90)
        self.assertEqual(parse_time("01:30.5", "时间"), 90.5)
        self.assertEqual(parse_time("01:02:03", "时间"), 3723)
        with self.assertRaises(ValueError):
            parse_time("01:75", "时间")

    def test_video_clip_range_schemes(self) -> None:
        self.assertEqual(_configured_range({
            "clip_mode": "指定开始和结束时间",
            "range_start_time": "10",
            "range_end_time": "20",
        }, 100), (10, 20, False))
        self.assertEqual(_configured_range({
            "clip_mode": "指定开始和结束时间",
            "range_start_time": "10",
            "range_end_time": "",
        }, 100), (10, 100, False))
        self.assertEqual(_configured_range({
            "clip_mode": "跳过开头和结尾",
            "head_skip_time": "5",
            "tail_skip_time": "10",
        }, 100), (5, 90, False))
        self.assertEqual(_configured_range({
            "clip_mode": "指定开始时间和持续时长",
            "duration_start_time": "90",
            "clip_duration": "20",
        }, 100), (90, 100, True))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the integration test")
    def test_video_clip_by_start_and_duration(self) -> None:
        source = Path(self.temp.name) / "clip_source.mp4"
        output = Path(self.temp.name) / "clips"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(40):
            writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
        writer.release()

        messages: list[str] = []
        result = run_video_clip(RunContext(
            function_id="video_clip",
            paths={"video_file": source, "output_folder": output},
            parameters={
                "clip_mode": "指定开始时间和持续时长",
                "duration_start_time": "00:00:01",
                "clip_duration": "1.2",
                "encoding_mode": "精确裁剪（推荐）",
            },
            report=messages.append,
        ))

        clipped = Path(result["outputFiles"][0])
        capture = cv2.VideoCapture(str(clipped))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            duration = capture.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        finally:
            capture.release()
        self.assertEqual(result["successCount"], 1)
        self.assertTrue(clipped.is_file())
        self.assertAlmostEqual(duration, 1.2, delta=0.2)
        self.assertTrue(any("1.000 秒 → 2.200 秒" in message for message in messages))

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is required for the integration test")
    def test_video_clip_defaults_to_stream_copy_without_size_inflation(self) -> None:
        source = Path(self.temp.name) / "copy_source.mp4"
        output = Path(self.temp.name) / "copy_clips"
        generator = np.random.default_rng(7)
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120))
        self.assertTrue(writer.isOpened())
        for _index in range(100):
            writer.write(generator.integers(0, 256, (120, 160, 3), dtype=np.uint8))
        writer.release()

        messages: list[str] = []
        result = run_video_clip(RunContext(
            function_id="video_clip",
            paths={"video_file": source, "output_folder": output},
            parameters={
                "clip_mode": "指定开始时间和持续时长",
                "duration_start_time": "2",
                "clip_duration": "3",
            },
            report=messages.append,
        ))

        clipped = Path(result["outputFiles"][0])
        self.assertEqual(result["details"][0]["encodingMode"], COPY_MODE)
        self.assertLess(clipped.stat().st_size, source.stat().st_size * 0.6)
        self.assertTrue(any("直接复制原视频编码流" in message for message in messages))

    def test_yolo_txt_split_is_repeatable_and_preserves_sources(self) -> None:
        images = Path(self.temp.name) / "images"
        labels = Path(self.temp.name) / "labels"
        negatives = Path(self.temp.name) / "negatives"
        output = Path(self.temp.name) / "output"
        for folder in (images, labels, negatives, output):
            folder.mkdir()
        for index in range(6):
            (images / f"image_{index}.jpg").write_bytes(b"image")
            (labels / f"image_{index}.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (labels / "classes.txt").write_text("person\n", encoding="utf-8")
        (negatives / "negative.jpg").write_bytes(b"negative")

        messages: list[str] = []
        result = run_yolo_dataset_split(RunContext(
            function_id="yolo_dataset_split",
            paths={
                "images_folder": images,
                "labels_folder": labels,
                "negative_images_folder": negatives,
                "output_folder": output,
            },
            parameters={
                "label_format": "YOLO TXT",
                "train_ratio": 0.7,
                "random_seed": 42,
                "missing_label_policy": "跳过并报告",
            },
            report=messages.append,
        ))

        dataset = Path(result["outputFolders"][0])
        self.assertEqual(result["trainCount"], 4)
        self.assertEqual(result["valCount"], 3)
        self.assertEqual(result["taskType"], "目标检测")
        self.assertEqual(dataset, output / "yolo_train")
        self.assertTrue(dataset.is_dir())
        self.assertEqual(len(list((dataset / "train").glob("*.jpg"))), 4)
        self.assertEqual(len(list((dataset / "val").glob("*.jpg"))), 3)
        self.assertEqual(len(list((dataset / "train").glob("*.txt"))), 4)
        self.assertEqual(len(list((dataset / "val").glob("*.txt"))), 3)
        self.assertFalse((dataset / "classes.txt").exists())
        self.assertEqual((Path(self.temp.name) / "classes.txt").read_text(encoding="utf-8"), "0: person\n")
        self.assertEqual((dataset / "val" / "negative.txt").read_text(encoding="utf-8"), "")
        train_index = (dataset / "train.txt").read_text(encoding="utf-8").splitlines()
        val_index = (dataset / "val.txt").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(train_index), 4)
        self.assertEqual(len(val_index), 3)
        self.assertTrue(all(Path(path).parent == dataset / "train" for path in train_index))
        self.assertTrue(all(Path(path).parent == dataset / "val" for path in val_index))
        self.assertFalse((dataset / "data.yaml").exists())
        data_yaml = Path(result["dataYaml"])
        self.assertEqual(data_yaml, Path(self.temp.name) / "data.yaml")
        self.assertEqual(result["trainingDefaults"], {
            "data": str(data_yaml),
            "project": str(data_yaml.parent),
        })
        yaml_content = data_yaml.read_text(encoding="utf-8")
        self.assertIn(f"train: {(dataset / 'train.txt').resolve()}", yaml_content)
        self.assertIn(f"val: {(dataset / 'val.txt').resolve()}", yaml_content)
        self.assertIn(f"test: {(dataset / 'val.txt').resolve()}", yaml_content)
        self.assertIn("nc: 1  # number of classes", yaml_content)
        self.assertIn("names: ['person']  # class names", yaml_content)
        self.assertEqual(len(list(images.iterdir())), 6)
        self.assertEqual(len(list(labels.glob("*.txt"))), 7)

        with self.assertRaisesRegex(ValueError, "已包含文件"):
            run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={
                    "images_folder": images,
                    "labels_folder": labels,
                    "negative_images_folder": negatives,
                    "output_folder": output,
                },
                parameters={"label_format": "YOLO TXT", "train_ratio": 0.7, "random_seed": 42},
                report=lambda _message: None,
            ))

    def test_unimplemented_yolo_task_type_is_rejected_before_processing(self) -> None:
        with self.assertRaisesRegex(ValueError, "旋转目标检测.*尚未接入"):
            run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={},
                parameters={"dataset_task_type": "旋转目标检测"},
                report=lambda _message: None,
            ))

    def test_yolo_txt_requires_class_names_for_data_yaml(self) -> None:
        images = Path(self.temp.name) / "images_without_classes"
        labels = Path(self.temp.name) / "labels_without_classes"
        output = Path(self.temp.name) / "output_without_classes"
        for folder in (images, labels, output):
            folder.mkdir()
        (images / "sample.jpg").write_bytes(b"image")
        (labels / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "未找到类别名称.*classes.txt"):
            run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={
                    "images_folder": images,
                    "labels_folder": labels,
                    "output_folder": output,
                },
                parameters={"label_format": "YOLO TXT"},
                report=lambda _message: None,
            ))
        self.assertFalse((output / "yolo_train").exists())

    def test_remote_yolo_mode_requires_successful_connection_test(self) -> None:
        with self.assertRaisesRegex(ValueError, "请先测试 SSH 连接"):
            run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={
                    "images_folder": Path("/remote/images"),
                    "labels_folder": Path("/remote/labels"),
                    "output_folder": Path("/remote/dataset"),
                },
                parameters={
                    "remote_host": "192.168.21.5",
                    "remote_username": "dell",
                    "remote_password": "temporary-secret",
                },
                report=lambda _message: None,
            ))

    def test_remote_yolo_mode_is_inferred_and_connection_token_matches_identity(self) -> None:
        self.assertFalse(remote_yolo_dataset.is_remote_yolo_parameters({
            "remote_host": "",
            "remote_username": "",
            "remote_password": "",
        }))
        parameters = {
            "remote_host": "192.168.21.5",
            "remote_port": 22,
            "remote_username": "dell",
            "remote_password": "temporary-secret",
            "remember_password": True,
        }
        self.assertTrue(remote_yolo_dataset.is_remote_yolo_parameters(parameters))
        with patch.object(remote_yolo_dataset.remote_connection_manager, "test_connection", return_value={
            "ok": True,
            "message": "connected",
            "host": "192.168.21.5",
            "port": 22,
            "username": "dell",
            "fingerprint": "00:11",
            "passwordRemembered": True,
            "usedSavedPassword": False,
            "testedAt": "2026-09-16T12:00:00",
        }):
            result = remote_yolo_dataset.test_remote_yolo_connection(parameters)
        parameters["remote_connection_token"] = result["connectionToken"]
        remote_yolo_dataset.require_tested_yolo_connection(parameters)
        with self.assertRaisesRegex(ValueError, "请先测试 SSH 连接"):
            remote_yolo_dataset.require_tested_yolo_connection({
                **parameters,
                "remote_host": "192.168.21.6",
            })

    def test_yolo_training_command_matches_reference_script_defaults(self) -> None:
        command = build_training_command(DEFAULT_TRAINING_VALUES)
        self.assertEqual(command[:3], ["yolo", "detect", "train"])
        self.assertIn("data=/home/dell/yzc_ws/data/Longcheng/shentou2-gw2-state/data.yaml", command)
        self.assertIn("model=/home/dell/yzc_ws/ultralytics/yolo26n.pt", command)
        self.assertIn("epochs=200", command)
        self.assertIn("patience=30", command)
        self.assertIn("device=2", command)
        self.assertIn("name=yolo26n_0915", command)
        self.assertIn("cos_lr=True", command)
        self.assertIn("optimizer=SGD", command)
        self.assertEqual(len(command), 34)

    def test_yolo_training_selects_environment_matching_model_family(self) -> None:
        candidates = [
            "/home/dell/anaconda3/envs/yolov11/bin/yolo",
            "/home/dell/anaconda3/envs/yolo26/bin/yolo",
            "/home/dell/anaconda3/bin/yolo",
        ]
        self.assertEqual(
            select_remote_yolo_candidate(candidates, "yolo26n.pt"),
            "/home/dell/anaconda3/envs/yolo26/bin/yolo",
        )
        self.assertIsNone(select_remote_yolo_candidate(candidates, "custom.pt"))

    def test_yolo_training_activates_matching_conda_environment_and_streams_progress(self) -> None:
        executable = "/home/dell/anaconda3/envs/yolo26/bin/yolo"
        self.assertEqual(
            _conda_environment_for_yolo(executable),
            ("/home/dell/anaconda3/etc/profile.d/conda.sh", "yolo26"),
        )
        shell, environment = _remote_training_shell(
            [executable, "detect", "train", "model=/srv/models/yolo26n.pt", "epochs=2"],
            "yolo-training-test",
        )
        self.assertEqual(environment, "yolo26")
        self.assertIn("conda activate yolo26", shlex.split(shell)[2])
        self.assertIn("PYTHONUNBUFFERED=1", shlex.split(shell)[2])
        self.assertIn("cd /srv/models", shlex.split(shell)[2])
        buffer = bytearray(b"Epoch 1/2\rEpoch 2/2\npartial")
        self.assertEqual(_remote_output_lines(buffer), ["Epoch 1/2", "Epoch 2/2"])
        self.assertEqual(_remote_output_lines(buffer, final=True), ["partial"])

    def test_yolo_training_profiles_support_named_scenario_crud(self) -> None:
        store = TrainingProfileStore(Path(self.temp.name) / "training_profiles.json")
        payload = store.payload()
        self.assertEqual(payload["profiles"][0]["name"], "通用训练")
        self.assertEqual(payload["defaults"]["patience"], 30)
        self.assertEqual(payload["profiles"][0]["values"]["patience"], 30)
        profile = store.create({
            "name": "光线变化",
            "description": "增强旋转和缩放，适合跨时段画面。",
            "values": {**DEFAULT_TRAINING_VALUES, "degrees": 5, "scale": 0.35},
        })
        self.assertEqual(profile["values"]["degrees"], 5.0)
        updated = store.update(profile["id"], {
            **profile,
            "description": "夜间与白天混合数据。",
            "values": {**profile["values"], "epochs": 300},
        })
        self.assertEqual(updated["values"]["epochs"], 300)
        store.delete(profile["id"])
        self.assertNotIn(profile["id"], {item["id"] for item in store.payload()["profiles"]})

    def test_yolo_training_runs_as_background_session(self) -> None:
        manager = TrainingManager(Path(self.temp.name) / "background_training.json")
        parameters = {**DEFAULT_TRAINING_VALUES, "yolo_executable": "/bin/echo"}
        snapshot = manager.start(parameters)
        for _ in range(50):
            current = manager.get(snapshot["id"]).snapshot()
            if current["status"] in {"completed", "failed", "stopped"}:
                break
            time.sleep(0.02)
        self.assertEqual(current["status"], "completed")
        self.assertTrue(any("detect train" in line for line in current["logs"]))

    def test_parallel_training_sessions_keep_outputs_and_controls_independent(self) -> None:
        gate = threading.Event()
        manager = TrainingManager(Path(self.temp.name) / "parallel_training.json")
        tasks = PlatformTaskManager()
        first_values = {**DEFAULT_TRAINING_VALUES, "project": "/tmp/training-tests", "run_name": "model_a", "device": "0"}
        second_values = {**first_values, "run_name": "model_b", "device": "1"}
        first = tasks.start("yolo_dataset_split", "模型 A", task_key=training_task_key(first_values))
        second = tasks.start("yolo_dataset_split", "模型 B", task_key=training_task_key(second_values))
        with self.assertRaisesRegex(ValueError, "正在运行"):
            tasks.start("yolo_dataset_split", "重复输出", task_key=training_task_key(first_values))

        def wait_for_release(context):
            while not gate.wait(0.01):
                context.check_cancelled()
            return {"message": "完成", "project": context.parameters["project"], "name": context.parameters["run_name"]}

        with patch("backend.yolo_training.run_training", side_effect=wait_for_release):
            a = manager.start(first_values)
            b = manager.start(second_values)
            try:
                self.assertNotEqual(a["id"], b["id"])
                self.assertEqual({item["output"] for item in manager.list()}, {"/tmp/training-tests/model_a", "/tmp/training-tests/model_b"})
                manager.stop(a["id"])
                for _ in range(50):
                    if manager.get(a["id"]).status == "stopped":
                        break
                    time.sleep(0.01)
                self.assertEqual(manager.get(a["id"]).status, "stopped")
                self.assertEqual(manager.get(b["id"]).status, "running")
            finally:
                gate.set()
            for _ in range(50):
                if manager.get(b["id"]).status == "completed":
                    break
                time.sleep(0.01)
            self.assertEqual(manager.get(b["id"]).status, "completed")
        tasks.finish(first.id)
        tasks.finish(second.id)

    def test_remote_yolo_worker_generates_dataset_without_project_dependencies(self) -> None:
        root = Path(self.temp.name)
        images = root / "remote_images"
        annotation_folder = root / "Annotation"
        labels = annotation_folder / "xmls"
        output = root / "remote_output"
        for folder in (images, labels, output):
            folder.mkdir(parents=True)
        annotation = """<annotation><size><width>100</width><height>80</height></size>
        <object><name>person</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
        </annotation>"""
        for index in range(4):
            (images / f"remote_{index}.jpg").write_bytes(b"image")
            (labels / f"remote_{index}.xml").write_text(annotation, encoding="utf-8")

        result = remote_yolo_dataset_worker.run(SimpleNamespace(
            images_folder=str(images),
            labels_folder=str(labels),
            negative_folder="",
            output_parent=str(output),
            dataset_folder_name="yolo_train_remote",
            label_format="voc_xml",
            train_ratio=0.5,
            seed=42,
            missing_policy="skip",
            session_token="test-session",
        ))

        dataset = output / "yolo_train_remote"
        self.assertEqual(result["executionLocation"], "SSH远程服务器")
        self.assertEqual(result["trainCount"], 2)
        self.assertEqual(result["valCount"], 2)
        self.assertEqual(Path(result["outputFolders"][0]), dataset)
        self.assertEqual(len(list((dataset / "train").glob("*.jpg"))), 2)
        self.assertEqual(len(list((dataset / "val").glob("*.jpg"))), 2)
        self.assertEqual(len(list((annotation_folder / "txts").glob("*.txt"))), 4)
        self.assertFalse((dataset / "txts").exists())
        self.assertFalse((dataset / "classes.txt").exists())
        self.assertEqual((root / "classes.txt").read_text(encoding="utf-8"), "0: person\n")
        yaml_content = (root / "data.yaml").read_text(encoding="utf-8")
        self.assertEqual(result["trainingDefaults"], {
            "data": str(root / "data.yaml"),
            "project": str(root),
        })
        self.assertIn(f"train: {(dataset / 'train.txt').resolve()}", yaml_content)
        self.assertIn("nc: 1  # number of classes", yaml_content)
        self.assertIn("names: ['person']", yaml_content)

    def test_voc_xml_is_converted_into_both_splits(self) -> None:
        images = Path(self.temp.name) / "xml_images"
        annotation_folder = Path(self.temp.name) / "Annotation"
        labels = annotation_folder / "xmls"
        output = Path(self.temp.name) / "xml_output"
        for folder in (images, labels, output):
            folder.mkdir(parents=True)
        annotation = """<annotation><size><width>100</width><height>80</height></size>
        <object><name>person</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
        </annotation>"""
        for index in range(4):
            (images / f"sample_{index}.png").write_bytes(b"image")
            (labels / f"sample_{index}.xml").write_text(annotation, encoding="utf-8")

        result = run_yolo_dataset_split(RunContext(
            function_id="yolo_dataset_split",
            paths={"images_folder": images, "labels_folder": labels, "output_folder": output},
            parameters={
                "label_format": "VOC XML",
                "dataset_folder_name": "custom_yolo_train",
                "train_ratio": 0.5,
                "random_seed": 7,
            },
            report=lambda _message: None,
        ))

        dataset = Path(result["outputFolders"][0])
        self.assertEqual(dataset, output / "custom_yolo_train")
        self.assertEqual(result["trainCount"], 2)
        self.assertEqual(result["valCount"], 2)
        self.assertEqual(len(list((dataset / "train").glob("*.txt"))), 2)
        self.assertEqual(len(list((dataset / "val").glob("*.txt"))), 2)
        self.assertEqual(len(list((annotation_folder / "txts").glob("*.txt"))), 4)
        self.assertFalse((dataset / "txts").exists())
        self.assertFalse((dataset / "classes.txt").exists())
        self.assertEqual((Path(self.temp.name) / "classes.txt").read_text(encoding="utf-8"), "0: person\n")
        self.assertEqual(Path(result["dataYaml"]), Path(self.temp.name) / "data.yaml")
        self.assertIn("names: ['person']", Path(result["dataYaml"]).read_text(encoding="utf-8"))
        self.assertIn("0 0.30000000 0.50000000 0.40000000 0.50000000", next((dataset / "val").glob("*.txt")).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
