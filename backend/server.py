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
from . import video_frames as _video_frames  # noqa: F401 - registers built-in handler
from . import video_clip as _video_clip  # noqa: F401 - registers built-in handler
from . import yolo_dataset_split as _yolo_dataset_split  # noqa: F401 - registers built-in handler


ROOT = Path(__file__).resolve().parents[1]
catalog = FunctionCatalog(ROOT / "runtime" / "functions.json")
group_catalog = GroupCatalog(ROOT / "runtime" / "groups.json")
_path_chooser: Callable[[str], list[str]] | None = None


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
            if self.path == "/api/remote-inference/test-connection":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_inference_manager.test_connection(parameters))
                return
            if self.path == "/api/remote-inference/forget-password":
                parameters = payload.get("parameters", payload)
                if not isinstance(parameters, dict):
                    raise ValueError("SSH 连接参数格式不正确。")
                self.send_json(remote_inference_manager.forget_password(parameters))
                return
            if self.path == "/api/remote-inference/start":
                self.send_json(remote_inference_manager.start(payload), 201)
                return
            if self.path.startswith("/api/remote-inference/") and self.path.endswith("/stop"):
                session_id = self.path.strip("/").split("/")[2]
                self.send_json(remote_inference_manager.stop(session_id))
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
                path_values = {key: Path(value).expanduser() for key, value in payload.get("paths", {}).items() if str(value).strip()}
                messages: list[str] = []
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
                    report=messages.append,
                ))
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
            if self.path.startswith("/api/groups/"):
                group_id = self.path.rsplit("/", 1)[-1]
                group_catalog.delete(group_id, catalog.list())
                self.send_json(workspace_payload())
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

    def server_close(self) -> None:
        remote_inference_manager.stop_all()
        super().server_close()


def create_server(port: int = 8765, static_dir: Path | None = None) -> ThreadingHTTPServer:
    AppHandler.static_dir = (static_dir or ROOT / "frontend" / "dist").resolve()
    return AppServer(("127.0.0.1", port), AppHandler)


def serve(port: int = 8765, static_dir: Path | None = None) -> None:
    server = create_server(port=port, static_dir=static_dir)
    server.serve_forever()


if __name__ == "__main__":
    serve()
