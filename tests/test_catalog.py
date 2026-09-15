from __future__ import annotations

import tempfile
import unittest
import shutil
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend import bundled_run_star_video
from backend.catalog import FunctionCatalog
from backend.credential_store import CredentialStore
from backend.groups import GroupCatalog
from backend.handlers import RunContext, execute, has_handler, register_handler
from backend.remote_inference import BUNDLED_INFERENCE_SCRIPT, RemoteInferenceManager, validate_connection_payload, validate_start_payload
from backend.video_clip import _configured_range, parse_time, run_video_clip
from backend.video_frames import run_video_frames
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
        self.assertEqual(items[1]["name"], "YOLO 数据集分配")
        self.assertEqual(items[1]["handlerId"], "yolo.split_dataset")
        task_type = next(parameter for parameter in items[1]["parameters"] if parameter["id"] == "dataset_task_type")
        self.assertEqual(task_type["default"], "目标检测")
        self.assertEqual(task_type["options"], ["目标检测", "旋转目标检测", "分割", "分类"])
        self.assertEqual(items[2]["name"], "视频裁剪")
        self.assertEqual(items[2]["handlerId"], "video.clip")
        self.assertEqual(items[3]["name"], "远程实时 AI 推理")
        self.assertEqual(items[3]["handlerId"], "remote.star_inference")
        password = next(parameter for parameter in items[3]["parameters"] if parameter["id"] == "password")
        remember_password = next(parameter for parameter in items[3]["parameters"] if parameter["id"] == "remember_password")
        self.assertEqual(password["type"], "password")
        self.assertEqual(password["default"], "")
        self.assertEqual(remember_password["type"], "boolean")
        self.assertFalse(remember_password["default"])
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
            "password": "",
            "remember_password": True,
        }, store)
        self.assertEqual(config["password"], "temporary-secret")
        self.assertTrue(config["password_from_store"])
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
        }, {
            "local_model_file": str(model),
            "local_video_file": str(video),
        })
        self.assertEqual(config["local_script_file"], BUNDLED_INFERENCE_SCRIPT)

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
        with self.assertRaisesRegex(ValueError, "开启.*记住 SSH 密码"):
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
        self.assertEqual(len(list((dataset / "images" / "train").iterdir())), 4)
        self.assertEqual(len(list((dataset / "images" / "val").iterdir())), 3)
        self.assertEqual(len(list((dataset / "labels" / "train").iterdir())), 4)
        self.assertEqual(len(list((dataset / "labels" / "val").iterdir())), 3)
        self.assertTrue((dataset / "data.yaml").is_file())
        self.assertEqual(len(list(images.iterdir())), 6)
        self.assertEqual(len(list(labels.glob("*.txt"))), 7)

    def test_unimplemented_yolo_task_type_is_rejected_before_processing(self) -> None:
        with self.assertRaisesRegex(ValueError, "旋转目标检测.*尚未接入"):
            run_yolo_dataset_split(RunContext(
                function_id="yolo_dataset_split",
                paths={},
                parameters={"dataset_task_type": "旋转目标检测"},
                report=lambda _message: None,
            ))

    def test_voc_xml_is_converted_into_both_splits(self) -> None:
        images = Path(self.temp.name) / "xml_images"
        labels = Path(self.temp.name) / "xml_labels"
        output = Path(self.temp.name) / "xml_output"
        for folder in (images, labels, output):
            folder.mkdir()
        annotation = """<annotation><size><width>100</width><height>80</height></size>
        <object><name>person</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax></bndbox></object>
        </annotation>"""
        for index in range(4):
            (images / f"sample_{index}.png").write_bytes(b"image")
            (labels / f"sample_{index}.xml").write_text(annotation, encoding="utf-8")

        result = run_yolo_dataset_split(RunContext(
            function_id="yolo_dataset_split",
            paths={"images_folder": images, "labels_folder": labels, "output_folder": output},
            parameters={"label_format": "VOC XML", "train_ratio": 0.5, "random_seed": 7},
            report=lambda _message: None,
        ))

        dataset = Path(result["outputFolders"][0])
        self.assertEqual(result["trainCount"], 2)
        self.assertEqual(result["valCount"], 2)
        self.assertEqual(len(list((dataset / "labels" / "train").glob("*.txt"))), 2)
        self.assertEqual(len(list((dataset / "labels" / "val").glob("*.txt"))), 2)
        self.assertEqual((dataset / "classes.txt").read_text(encoding="utf-8"), "person\n")
        self.assertIn("0 0.30000000 0.50000000 0.40000000 0.50000000", next((dataset / "labels" / "val").glob("*.txt")).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
