"""Windows-client regression tests for the SSH dataset split endpoint."""

import io
import json
import shlex
import unittest
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from unittest.mock import patch

from backend import remote_yolo_dataset as remote, server, yolo_dataset_split
from backend.task_manager import PlatformTaskManager


class RemoteYoloPathTests(unittest.TestCase):
    def submit(self, paths, parameters):
        body = json.dumps({"paths": paths, "parameters": parameters}).encode("utf-8")
        raw = f"POST /api/functions/yolo_dataset_split/run HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        response = bytearray()
        connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=response.extend)
        with patch.object(server.license_store, "status", return_value={"activated": True}), \
             patch.object(server, "platform_task_manager", PlatformTaskManager()), \
             patch.object(server, "record_finished_run"):
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
        headers, data = bytes(response).split(b"\r\n\r\n", 1)
        return int(headers.split()[1]), json.loads(data)

    def remote_request(self, paths):
        def prepare_remote_command(context):
            config = remote._configuration(context)
            return {"command": remote._command(config, "/tmp/worker.py", "test")}

        # The local Path constructor follows Windows rules, while SSH still
        # targets a Linux server. Exercise the full API-to-command path.
        with patch.object(server, "Path", PureWindowsPath), \
             patch.object(remote, "run_remote_yolo_dataset", side_effect=prepare_remote_command), \
             patch.object(remote, "resolve_tested_yolo_connection", return_value={
                 "host": "server.example", "port": 22, "username": "test", "password": "fixture",
             }):
            return self.submit(paths, {"remote_host": "server.example"})

    def test_windows_client_preserves_all_remote_paths_in_ssh_arguments(self):
        paths = {
            "images_folder": "/home/test/Data/产线 '1'/image",
            "labels_folder": "/home/test/Data/产线 '1'/xml",
            "negative_images_folder": "datasets/negative examples",
            "output_folder": "~/dataset results",
        }
        status, response = self.remote_request(paths)
        self.assertEqual(status, 200, response)
        command = shlex.split(response["result"]["command"])
        for field, flag in (("images_folder", "--images-folder"), ("labels_folder", "--labels-folder"),
                            ("negative_images_folder", "--negative-folder"), ("output_folder", "--output-parent")):
            self.assertEqual(command[command.index(flag) + 1], paths[field])

    def test_windows_path_entered_in_remote_mode_gets_a_clear_error(self):
        for invalid in (r"\home\test\Data\image", r"C:\Data\image", "D:/Data/image"):
            with self.subTest(path=invalid):
                status, response = self.remote_request({
                    "images_folder": invalid, "labels_folder": "/remote/xml", "output_folder": "/remote/output",
                })
                self.assertEqual(status, 400)
                self.assertIn("远程图片文件夹请使用服务器上的 Linux 路径", response["error"])

    def test_local_mode_still_uses_native_paths(self):
        with patch.object(yolo_dataset_split, "_run_local_yolo_dataset_split", return_value={}) as local:
            status, response = self.submit({"images_folder": "~/images", "output_folder": "~/dataset"}, {})
        self.assertEqual(status, 200, response)
        self.assertEqual(local.call_args.args[0].paths,
                         {"images_folder": Path.home() / "images", "output_folder": Path.home() / "dataset"})
