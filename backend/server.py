"""Dependency-free local API and static-file server."""

from __future__ import annotations

import json
import mimetypes
import os
import subprocess
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

from .catalog import FunctionCatalog
from .groups import GroupCatalog
from .handlers import RunContext, execute, has_handler
from .remote_inference import manager as remote_inference_manager
from .local_pt_inference import manager as local_pt_inference_manager
from .local_star_package import detect_plan_labels
from .run_history import RunHistoryStore
from .remote_yolo_dataset import (
    forget_remote_yolo_password,
    is_remote_yolo_parameters,
    remote_yolo_credential_status,
    test_remote_yolo_connection,
)
from .remote_calib_dataset import is_remote_calib_parameters
from .task_manager import ActiveTask, TaskCancelled, manager as platform_task_manager
from .yolo_training import (
    TrainingProfileStore,
    normalize_training_values,
    training_task_key,
    manager as yolo_training_manager,
)
from . import video_frames as _video_frames  # noqa: F401 - registers built-in handler
from . import video_clip as _video_clip  # noqa: F401 - registers built-in handler
from . import image_dedup as _image_dedup  # noqa: F401 - registers built-in handler
from . import docker_onnx_quant as _docker_onnx_quant  # noqa: F401 - registers built-in handler
from . import jetson_onnx_export as _jetson_onnx_export  # noqa: F401 - registers built-in handler
from . import yolo_dataset_split as _yolo_dataset_split  # noqa: F401 - registers built-in handler
from . import remote_calib_dataset as _remote_calib_dataset  # noqa: F401 - registers built-in handler
from . import remote_tensorrt_build as _remote_tensorrt_build  # noqa: F401 - registers built-in handler
from . import local_star_package as _local_star_package  # noqa: F401 - registers built-in handler


ROOT = Path(__file__).resolve().parents[1]
catalog = FunctionCatalog(ROOT / "runtime" / "functions.json")
group_catalog = GroupCatalog(ROOT / "runtime" / "groups.json")
training_profile_store = TrainingProfileStore(ROOT / "runtime" / "training_profiles.json")
run_history = RunHistoryStore(ROOT / "runtime" / "run_history.json")
_path_chooser: Callable[[str], list[str]] | None = None
_system_notifier: Callable[[str, str], None] | None = None


def set_system_notifier(notifier: Callable[[str, str], None] | None) -> None:
    """Let the desktop shell display non-blocking operating-system notifications."""
    global _system_notifier
    _system_notifier = notifier


def notify_system(title: str, message: str) -> None:
    if _system_notifier is None:
        return
    try:
        _system_notifier(str(title), str(message))
    except Exception as exc:
        # Notification failures must never change processing results.
        print(f"[YOLO数据处理平台] 系统通知显示失败：{exc}", file=sys.stderr)


def record_finished_run(
    task: ActiveTask,
    *,
    status: str,
    message: str,
    logs: list[str] | None = None,
    details: dict[str, str] | None = None,
    outputs: list[str] | None = None,
) -> None:
    try:
        run_history.add(task.snapshot(), status=status, message=message, logs=logs, details=details, outputs=outputs)
    except Exception as exc:
        # A history disk error must not turn a successful data-processing run into a failure.
        print(f"[YOLO数据处理平台] 无法保存运行历史：{exc}", file=sys.stderr)
    if status in {"completed", "finished"}:
        notify_system(f"{task.name} 已完成", message or "任务已成功完成。")


def finish_async_run(task: ActiveTask, snapshot: dict[str, Any], details: dict[str, str]) -> None:
    try:
        result = snapshot.get("result") or {}
        outputs = [str(Path(str(result["project"])) / str(result["name"]))] if result.get("project") and result.get("name") else []
        record_finished_run(
            task,
            status=str(snapshot["status"]),
            message=str(snapshot.get("error") or snapshot.get("message") or ""),
            logs=list(snapshot.get("logs") or []),
            details=details,
            outputs=outputs,
        )
    finally:
        platform_task_manager.finish(task.id)


def active_training_history_records() -> list[dict[str, Any]]:
    """Expose active training sessions alongside, but outside, the five-item finished history."""
    records: list[dict[str, Any]] = []
    terminal = {"completed", "failed", "stopped"}
    for summary in yolo_training_manager.list():
        if str(summary.get("status", "")) in terminal:
            continue
        try:
            session = yolo_training_manager.get(str(summary["id"])).snapshot()
        except (KeyError, ValueError):
            continue
        output = str(session.get("output", "")).strip()
        run_name = Path(output).name if output else str(session.get("id", ""))
        remote = bool(session.get("remote"))
        status = str(session.get("status", "running"))
        if status == "disconnected":
            state_message = "远程训练可能仍在运行，当前等待重新连接。"
        elif status == "stopping":
            state_message = "训练正在终止。"
        elif status == "starting":
            state_message = "训练正在准备启动。"
        else:
            state_message = "模型正在训练中。"
        records.append({
            "id": f"active-training-{session['id']}",
            "functionId": "yolo_dataset_split",
            "name": f"YOLO 训练 · {run_name}",
            "kind": "remote-training" if remote else "local",
            "status": "running",
            "startedAt": str(session.get("startedAt", "")),
            "finishedAt": "",
            "message": str(session.get("message") or state_message),
            "details": {
                "训练状态": state_message,
                "模型权重": str(session.get("model", "")),
                "训练输出": output,
                "设备": f"{session.get('host', '')} · GPU {session.get('device', '')}" if remote
                else f"本地 · GPU {session.get('device', '')}",
            },
            "outputs": [output] if output else [],
            "logs": [str(line) for line in session.get("logs", [])][-500:],
        })
    return records


def workspace_payload() -> dict[str, Any]:
    functions = catalog.list()
    for item in functions:
        item["handlerReady"] = has_handler(item.get("handlerId"))
    return {
        "groups": group_catalog.list_groups(),
        "functions": group_catalog.decorate(functions),
    }


def decorated_function(item: dict[str, Any]) -> dict[str, Any]:
    item["handlerReady"] = has_handler(item.get("handlerId"))
    return next(
        entry for entry in group_catalog.decorate([item]) if entry["id"] == item["id"]
    )


def set_path_chooser(chooser: Callable[[str], list[str]] | None) -> None:
    """Let a desktop shell provide its native file picker."""
    global _path_chooser
    _path_chooser = chooser


def choose_path(mode: str) -> list[str]:
    if _path_chooser is not None:
        return _path_chooser(mode)
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if mode == "directory":
            value = filedialog.askdirectory(title="选择文件夹")
            values = [value] if value else []
        elif mode == "save_directory":
            value = filedialog.askdirectory(title="选择输出文件夹")
            values = [value] if value else []
        elif mode == "file":
            value = filedialog.askopenfilename(title="选择文件")
            values = [value] if value else []
        else:
            values = list(filedialog.askopenfilenames(title="选择一个或多个文件"))
        root.destroy()
        return values
    except Exception as exc:
        raise RuntimeError(f"无法打开系统选择窗口，请直接粘贴完整路径。原因：{exc}") from exc


def open_in_system(path_value: str) -> None:
    path = Path(path_value).expanduser().resolve()
    target = path if path.is_dir() else path.parent
    if not target.exists():
        raise ValueError("结果路径不存在。")
    if sys.platform == "win32":
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


class AppHandler(BaseHTTPRequestHandler):
    static_dir: Path

    def log_message(self, format: str, *args: Any) -> None:
        if self.path.startswith("/api/tasks/active"):
            return
        print(f"[YOLO数据处理平台] {self.address_string()} - {format % args}")

    def send_json(self, data: Any, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_000_000:
            raise ValueError("请求内容过大。")
        body = self.rfile.read(length)
        return json.loads(body.decode("utf-8")) if body else {}

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/health":
            self.send_json({"ok": True, "name": "YOLO数据处理平台"})
            return
        if parsed.path == "/api/workspace":
            self.send_json(workspace_payload())
            return
        if parsed.path == "/api/groups":
            self.send_json({"groups": group_catalog.list_groups()})
            return
        if parsed.path == "/api/functions":
            self.send_json({"functions": workspace_payload()["functions"]})
            return
        if parsed.path == "/api/tasks/active":
            self.send_json({"active": platform_task_manager.active()})
            return
        if parsed.path == "/api/tasks/history":
            try:
                parameters = urllib.parse.parse_qs(parsed.query)
                function_id = str(parameters.get("functionId", [""])[0]).strip()
                active_records = active_training_history_records()
                if function_id:
                    active_records = [record for record in active_records if record["functionId"] == function_id]
                    finished_records = run_history.list(function_id)
                else:
                    finished_records = run_history.list()[: run_history.LIMIT]
                self.send_json({"history": [*active_records, *finished_records]})
            except (ValueError, OSError, json.JSONDecodeError) as exc:
                self.send_json({"error": f"读取运行历史失败：{exc}"}, 500)
            return
        if parsed.path == "/api/yolo-training/profiles":
            self.send_json(training_profile_store.payload())
            return
        if parsed.path == "/api/yolo-training/sessions":
            self.send_json({"sessions": yolo_training_manager.list()})
            return
        if parsed.path.startswith("/api/yolo-training/") and parsed.path.endswith("/status"):
            try:
                session_id = parsed.path.strip("/").split("/")[2]
                self.send_json(yolo_training_manager.get(session_id).snapshot())
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 404)
            return
        if parsed.path.startswith("/api/remote-inference/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "remote-inference"]:
                try:
                    session = remote_inference_manager.get(parts[2])
                    if parts[3] == "status":
                        self.send_json(session.snapshot())
                        return
                    if parts[3] == "stream":
                        self.serve_remote_stream(session)
                        return
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 404)
                    return
        if parsed.path.startswith("/api/local-pt-inference/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) == 4 and parts[:2] == ["api", "local-pt-inference"]:
                try:
                    session = local_pt_inference_manager.get(parts[2])
                    if parts[3] == "status":
                        self.send_json(session.snapshot())
                        return
                    if parts[3] == "stream":
                        self.serve_remote_stream(session)
                        return
                except ValueError as exc:
                    self.send_json({"error": str(exc)}, 404)
                    return
        if parsed.path == "/api/open":
            try:
                params = urllib.parse.parse_qs(parsed.query)
                open_in_system(params.get("path", [""])[0])
                self.send_json({"ok": True})
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
            return
        self.serve_static(parsed.path)

    def do_POST(self) -> None:
        try:
            payload = self.read_json()
            if self.path == "/api/groups":
                group_catalog.create(str(payload.get("name", "")))
                self.send_json(workspace_payload(), 201)
                return
            if self.path == "/api/groups/reorder":
                group_ids = payload.get("groupIds", [])
                if not isinstance(group_ids, list):
                    raise ValueError("分组排序数据格式不正确。")
                group_catalog.reorder_groups([str(group_id) for group_id in group_ids])
                self.send_json(workspace_payload())
                return
            if self.path == "/api/functions":
                item = catalog.create(payload)
                self.send_json(decorated_function(item), 201)
                return
            if self.path == "/api/dialog":
                self.send_json({"paths": choose_path(str(payload.get("mode", "multiple")))})
                return
            if self.path == "/api/local-star-package/detect-labels":
                self.send_json(detect_plan_labels(str(payload.get("modelFile", ""))))
                return
            if self.path == "/api/remote-inference/test-connection":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_inference_manager.test_connection(parameters))
                return
            if self.path == "/api/remote-inference/credential-status":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_inference_manager.credential_status(parameters))
                return
            if self.path == "/api/remote-inference/forget-password":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_inference_manager.forget_password(parameters))
                return
            if self.path == "/api/yolo-dataset/test-connection":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(test_remote_yolo_connection(parameters))
                return
            if self.path == "/api/yolo-dataset/credential-status":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_yolo_credential_status(parameters))
                return
            if self.path == "/api/yolo-dataset/forget-password":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(forget_remote_yolo_password(parameters))
                return
            if self.path == "/api/yolo-training/profiles":
                self.send_json(training_profile_store.create(payload), 201)
                return
            if self.path == "/api/yolo-training/start":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("训练参数格式不正确。")
                is_remote = is_remote_yolo_parameters(parameters)
                values = normalize_training_values(parameters)
                task = platform_task_manager.start(
                    "yolo_dataset_split", f"YOLO 训练 · {values['run_name']} · {values['device']}", "remote-training" if is_remote else "local",
                    task_key=training_task_key(parameters),
                )
                details: dict[str, str] = {}
                try:
                    details = {
                        "数据配置": str(parameters.get("data", "")),
                        "模型权重": str(parameters.get("model", "")),
                        "训练输出": str(Path(str(parameters.get("project", ""))) / str(parameters.get("run_name", ""))) if parameters.get("project") else "",
                        "设备": str(parameters.get("remote_host", "")) if is_remote else "本地",
                    }
                    snapshot = yolo_training_manager.start(
                        parameters,
                        on_finished=lambda finished: finish_async_run(task, finished, details),
                    )
                    session_id = str(snapshot["id"])
                    platform_task_manager.set_stop_callback(
                        task.id,
                        lambda: yolo_training_manager.stop(session_id),
                    )
                except Exception as exc:
                    record_finished_run(task, status="failed", message=str(exc), details=details)
                    platform_task_manager.finish(task.id)
                    raise
                self.send_json(snapshot, 201)
                return
            if self.path.startswith("/api/yolo-training/") and self.path.endswith("/reconnect"):
                session_id = self.path.strip("/").split("/")[2]
                parameters = payload.get("parameters", payload)
                self.send_json(yolo_training_manager.reconnect(session_id, parameters))
                return
            if self.path.startswith("/api/yolo-training/") and self.path.endswith("/stop"):
                session_id = self.path.strip("/").split("/")[2]
                self.send_json(yolo_training_manager.stop(session_id))
                return
            if self.path == "/api/remote-inference/start":
                task = platform_task_manager.start(
                    "remote_star_inference", "远程实时 AI 推理", "remote"
                )
                details = {}
                try:
                    paths = payload.get("paths", {})
                    parameters = payload.get("parameters", {})
                    details = {
                        "模型文件": str(paths.get("local_model_file", "")),
                        "视频文件": str(paths.get("local_video_file", "")),
                        "推理设备": str(parameters.get("host", "")),
                        "远端保存": str(parameters.get("save_path", "")),
                    }
                    snapshot = remote_inference_manager.start(
                        payload,
                        on_finished=lambda finished: finish_async_run(task, finished, details),
                    )
                    session_id = str(snapshot["id"])
                    platform_task_manager.set_stop_callback(
                        task.id,
                        lambda: remote_inference_manager.stop(session_id),
                    )
                except Exception as exc:
                    record_finished_run(task, status="failed", message=str(exc), details=details)
                    platform_task_manager.finish(task.id)
                    raise
                self.send_json(snapshot, 201)
                return
            if self.path == "/api/local-pt-inference/start":
                task = platform_task_manager.start("local_pt_inference", "本地 PT 视频检测", "local")
                paths = payload.get("paths", {})
                details = {"模型文件": str(paths.get("model_file", "")), "视频文件": str(paths.get("video_file", "")), "保存位置": str(paths.get("output_folder", ""))}
                try:
                    snapshot = local_pt_inference_manager.start(
                        payload, on_finished=lambda finished: finish_async_run(task, finished, details),
                    )
                    session_id = str(snapshot["id"])
                    platform_task_manager.set_stop_callback(task.id, lambda: local_pt_inference_manager.stop(session_id))
                except Exception as exc:
                    record_finished_run(task, status="failed", message=str(exc), details=details)
                    platform_task_manager.finish(task.id)
                    raise
                self.send_json(snapshot, 201)
                return
            if self.path.startswith("/api/local-pt-inference/"):
                parts = self.path.strip("/").split("/")
                if len(parts) == 4:
                    session_id, action = parts[2], parts[3]
                    if action == "stop":
                        self.send_json(local_pt_inference_manager.stop(session_id))
                        return
                    if action == "seek":
                        self.send_json(local_pt_inference_manager.seek(session_id, payload.get("seconds")))
                        return
                    if action == "playback":
                        self.send_json(local_pt_inference_manager.set_paused(session_id, payload.get("paused")))
                        return
            if self.path.startswith("/api/remote-inference/") and self.path.endswith("/stop"):
                session_id = self.path.strip("/").split("/")[2]
                self.send_json(remote_inference_manager.stop(session_id))
                return
            if self.path.startswith("/api/remote-inference/") and self.path.endswith("/seek"):
                session_id = self.path.strip("/").split("/")[2]
                self.send_json(remote_inference_manager.seek(session_id, payload.get("seconds")))
                return
            if self.path.startswith("/api/remote-inference/") and self.path.endswith("/playback"):
                session_id = self.path.strip("/").split("/")[2]
                self.send_json(remote_inference_manager.set_paused(session_id, payload.get("paused")))
                return
            if self.path.startswith("/api/tasks/") and self.path.endswith("/stop"):
                task_id = self.path.strip("/").split("/")[2]
                self.send_json({"active": platform_task_manager.stop(task_id)})
                return
            if self.path.startswith("/api/functions/") and self.path.endswith("/run"):
                item_id = self.path.split("/")[3]
                item = catalog.get(item_id)
                if not item:
                    self.send_json({"error": "功能不存在。"}, 404)
                    return
                handler_id = item.get("handlerId")
                if not has_handler(handler_id):
                    self.send_json({
                        "error": f"“{item['name']}”尚未接入处理逻辑。后续绑定处理器后即可直接运行。",
                        "code": "handler_not_bound",
                    }, 409)
                    return
                is_remote_yolo = (
                    handler_id == "yolo.split_dataset"
                    and is_remote_yolo_parameters(payload.get("parameters", {}))
                )
                parameters = payload.get("parameters", {})
                is_remote_calib = handler_id == "calib.select_dataset" and is_remote_calib_parameters(parameters)
                is_remote_trt = handler_id == "remote.build_tensorrt"
                path_values = {
                    key: Path(value) if is_remote_yolo else Path(value).expanduser()
                    for key, value in payload.get("paths", {}).items()
                    if str(value).strip()
                }
                messages: list[str] = []
                task = platform_task_manager.start(
                    item_id, str(item["name"]),
                    "remote-build" if is_remote_trt else "remote" if is_remote_yolo or is_remote_calib else "local",
                )
                details = {
                    str(field["label"]): str(payload.get("paths", {}).get(field["id"], ""))
                    for field in item.get("pathFields", [])
                    if field.get("id")
                }
                if handler_id == "calib.select_dataset":
                    details.update({
                        "图片目录": str(parameters.get("image_dirs", "")),
                        "输出目录": str(parameters.get("output_dir", "")),
                        "执行位置": f"{parameters.get('remote_username', '')}@{parameters.get('remote_host', '')}" if is_remote_calib else "本机",
                    })
                elif is_remote_yolo:
                    details["远程服务器"] = str(payload.get("parameters", {}).get("remote_host", ""))
                elif is_remote_trt:
                    parameters = payload.get("parameters", {})
                    details["远程服务器"] = f"{parameters.get('username', '')}@{parameters.get('host', '')}:{parameters.get('port', 22)}"
                    raw_onnx_files = str(parameters.get("onnx_files", ""))
                    try:
                        onnx_files = json.loads(raw_onnx_files) if raw_onnx_files.strip().startswith("[") else raw_onnx_files.splitlines()
                    except json.JSONDecodeError:
                        onnx_files = []
                    if isinstance(onnx_files, list) and onnx_files:
                        normalized_onnx_files = [str(path).strip() for path in onnx_files if str(path).strip()]
                        details["ONNX 模型"] = "\n".join(normalized_onnx_files)
                        details["模型数量"] = str(len(normalized_onnx_files))
                status = "failed"
                message = ""
                outputs: list[str] = []
                try:
                    def report(message: str) -> None:
                        messages.append(message)
                        task.add_log(message)

                    result = execute(str(handler_id), RunContext(
                        function_id=item_id,
                        paths=path_values,
                        parameters={
                            **{
                                str(parameter.get("id")): parameter.get("default")
                                for parameter in item.get("parameters", [])
                                if parameter.get("id")
                            },
                            **dict(payload.get("parameters", {})),
                        },
                        report=report,
                        stop_event=task.stop_event,
                        notify=notify_system,
                    ))
                    status = "completed"
                    message = str(result.get("message", "处理完成。"))
                    outputs = list(dict.fromkeys(
                        str(path)
                        for path in [*result.get("outputFiles", []), *result.get("outputFolders", [])]
                    ))
                except TaskCancelled as exc:
                    status = "stopped"
                    message = str(exc)
                    self.send_json({
                        "error": str(exc),
                        "code": "task_cancelled",
                        "messages": messages,
                    }, 409)
                    return
                except Exception as exc:
                    message = str(exc)
                    raise
                finally:
                    record_finished_run(task, status=status, message=message, logs=messages, details=details, outputs=outputs)
                    platform_task_manager.finish(task.id)
                self.send_json({"ok": True, "result": result, "messages": messages})
                return
            if self.path.startswith("/api/functions/") and self.path.endswith("/move"):
                item_id = self.path.split("/")[3]
                group_id = payload.get("groupId")
                group_catalog.move_function(
                    item_id,
                    str(group_id) if group_id is not None else None,
                    catalog.list(),
                    payload.get("position"),
                )
                self.send_json(workspace_payload())
                return
            self.send_json({"error": "接口不存在"}, 404)
        except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self.send_json({"error": f"服务异常：{exc}"}, 500)

    def do_PUT(self) -> None:
        try:
            payload = self.read_json()
            if self.path.startswith("/api/groups/"):
                group_id = self.path.rsplit("/", 1)[-1]
                group_catalog.rename(group_id, str(payload.get("name", "")))
                self.send_json(workspace_payload())
                return
            if self.path.startswith("/api/yolo-training/profiles/"):
                profile_id = self.path.rsplit("/", 1)[-1]
                self.send_json(training_profile_store.update(profile_id, payload))
                return
            if self.path.startswith("/api/functions/"):
                item_id = self.path.rsplit("/", 1)[-1]
                item = catalog.update(item_id, payload)
                self.send_json(decorated_function(item))
                return
            self.send_json({"error": "接口不存在"}, 404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 404)
        except Exception as exc:
            self.send_json({"error": f"服务异常：{exc}"}, 500)

    def do_DELETE(self) -> None:
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path.startswith("/api/tasks/history/"):
                record_id = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]).strip()
                run_history.delete(record_id)
                self.send_json({"ok": True})
                return
            if self.path.startswith("/api/yolo-training/sessions/"):
                session_id = self.path.rsplit("/", 1)[-1]
                yolo_training_manager.delete(session_id)
                self.send_json({"ok": True})
                return
            if self.path.startswith("/api/groups/"):
                group_id = self.path.rsplit("/", 1)[-1]
                group_catalog.delete(group_id, catalog.list())
                self.send_json(workspace_payload())
                return
            if self.path.startswith("/api/yolo-training/profiles/"):
                profile_id = self.path.rsplit("/", 1)[-1]
                training_profile_store.delete(profile_id)
                self.send_json({"ok": True})
                return
            if self.path.startswith("/api/functions/"):
                item_id = self.path.rsplit("/", 1)[-1]
                catalog.delete(item_id)
                group_catalog.remove_function(item_id)
                self.send_json({"ok": True})
                return
            self.send_json({"error": "接口不存在"}, 404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, 404)
        except Exception as exc:
            self.send_json({"error": f"服务异常：{exc}"}, 500)

    def serve_static(self, url_path: str) -> None:
        relative = url_path.lstrip("/") or "index.html"
        requested = (self.static_dir / relative).resolve()
        try:
            requested.relative_to(self.static_dir.resolve())
        except ValueError:
            self.send_error(403)
            return
        if not requested.exists() or not requested.is_file():
            requested = self.static_dir / "index.html"
        if not requested.exists():
            self.send_error(404, "界面尚未构建")
            return
        body = requested.read_bytes()
        mime = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_remote_stream(self, session: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=yoloframe")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        version = 0
        try:
            while True:
                next_version, frame, status = session.wait_for_frame(version, timeout=1.0)
                if frame is not None and next_version != version:
                    version = next_version
                    self.wfile.write(b"--yoloframe\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                if status in {"completed", "stopped", "failed"} and next_version == version:
                    break
        except (BrokenPipeError, ConnectionResetError):
            return


class AppServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def server_close(self) -> None:
        platform_task_manager.stop_all()
        remote_inference_manager.stop_all()
        local_pt_inference_manager.stop_all()
        yolo_training_manager.stop_all()
        super().server_close()


def create_server(port: int = 8765, static_dir: Path | None = None) -> ThreadingHTTPServer:
    AppHandler.static_dir = (static_dir or ROOT / "frontend" / "dist").resolve()
    return AppServer(("127.0.0.1", port), AppHandler)


def serve(port: int = 8765, static_dir: Path | None = None) -> None:
    server = create_server(port=port, static_dir=static_dir)
    server.serve_forever()


if __name__ == "__main__":
    serve()
