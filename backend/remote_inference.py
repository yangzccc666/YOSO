"""SSH-backed live inference sessions for remote NVIDIA devices."""

from __future__ import annotations

import re
import shlex
import socket
import struct
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .credential_store import CredentialStore
from .handlers import RunContext, register_handler


ROOT = Path(__file__).resolve().parents[1]
REMOTE_WORKER = ROOT / "backend" / "remote_star_worker.py"
BUNDLED_INFERENCE_SCRIPT = ROOT / "backend" / "bundled_run_star_video.py"
TERMINAL_STATES = {"completed", "stopped", "failed"}
MAX_FRAME_BYTES = 24 * 1024 * 1024
HOST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,252}$")
MODEL_SUFFIXES = {".star", ".plan", ".engine"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts", ".mts"}
SCRIPT_SUFFIXES = {".py"}


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{label}必须在 {minimum:g} 到 {maximum:g} 之间。")
    return parsed


def _local_file(raw_paths: dict[str, Any], field_id: str, label: str, suffixes: set[str]) -> Path:
    value = str(raw_paths.get(field_id, "")).strip()
    if not value:
        raise ValueError(f"请选择{label}。")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label}不存在或不是文件：{path}")
    if path.suffix.lower() not in suffixes:
        allowed = "、".join(sorted(suffixes))
        raise ValueError(f"{label}格式不支持，请选择 {allowed} 文件。")
    return path


def validate_connection_identity(raw: dict[str, Any]) -> dict[str, Any]:
    host = str(raw.get("host", "")).strip()
    username = str(raw.get("username", "")).strip()
    if not host or not HOST_PATTERN.fullmatch(host):
        raise ValueError("请输入有效的设备 IP 或主机名。")
    if not username:
        raise ValueError("请输入 SSH 用户名。")
    return {
        "host": host,
        "port": int(_number(raw.get("port", 22), "SSH 端口", 1, 65535)),
        "username": username,
    }


def validate_connection_payload(
    raw: dict[str, Any],
    credential_store: CredentialStore | None = None,
) -> dict[str, Any]:
    identity = validate_connection_identity(raw)
    remember_password = bool(raw.get("remember_password", False))
    password = str(raw.get("password", ""))
    password_from_store = False
    if not password and remember_password and credential_store is not None:
        password = credential_store.get(
            identity["host"], identity["port"], identity["username"]
        ) or ""
        password_from_store = bool(password)
    if not password:
        if remember_password:
            raise ValueError("尚未保存此设备的 SSH 密码，请输入一次密码并重新测试连接。")
        raise ValueError("请输入本次 SSH 登录密码，或开启“记住 SSH 密码”。")
    return {
        **identity,
        "password": password,
        "remember_password": remember_password,
        "password_from_store": password_from_store,
    }


def validate_start_payload(
    raw: dict[str, Any],
    raw_paths: dict[str, Any] | None = None,
    credential_store: CredentialStore | None = None,
) -> dict[str, Any]:
    paths = raw_paths or {}
    connection = validate_connection_payload(raw, credential_store)
    python_path = str(raw.get("python_path", "python3")).strip()
    if not python_path:
        raise ValueError("请输入远端 Python 解释器路径。")
    local_script_value = str(paths.get("local_script_file", "")).strip()
    local_script_file = (
        _local_file(paths, "local_script_file", "本地推理脚本", SCRIPT_SUFFIXES)
        if local_script_value
        else BUNDLED_INFERENCE_SCRIPT
    )
    if not local_script_file.is_file():
        raise RuntimeError("平台内置推理脚本缺失，请重新安装或选择本地 run_star_video.py。")
    local_model_file = _local_file(paths, "local_model_file", "本地 STAR 模型文件", MODEL_SUFFIXES)
    local_video_file = _local_file(paths, "local_video_file", "本地视频文件", VIDEO_SUFFIXES)

    return {
        **connection,
        "python_path": python_path,
        "local_script_file": local_script_file,
        "local_model_file": local_model_file,
        "local_video_file": local_video_file,
        "labels": str(raw.get("labels", "class0")).strip() or "class0",
        "conf": _number(raw.get("conf", 0.5), "置信度阈值", 0, 1),
        "iou": _number(raw.get("iou", 0.45), "NMS IoU 阈值", 0, 1),
        "max_det": int(_number(raw.get("max_det", 300), "每帧最大框数", 1, 10000)),
        "save_path": str(raw.get("save_path", "")).strip(),
        "realtime": bool(raw.get("realtime", True)),
        "preview_fps": _number(raw.get("preview_fps", 12), "预览帧率", 0, 60),
        "stream_width": int(_number(raw.get("stream_width", 1280), "预览宽度", 320, 3840)),
        "jpeg_quality": int(_number(raw.get("jpeg_quality", 80), "画面质量", 20, 100)),
        "cuda_backend": str(raw.get("cuda_backend", "auto")),
    }


@dataclass
class RemoteInferenceSession:
    id: str
    host: str
    username: str
    status: str = "connecting"
    message: str = "正在连接远端设备……"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    frame_count: int = 0
    latest_frame: bytes | None = None
    frame_version: int = 0
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=160))
    error: str | None = None
    transfer_stage: str | None = None
    upload_progress: int = 0
    stop_event: threading.Event = field(default_factory=threading.Event)
    condition: threading.Condition = field(default_factory=threading.Condition)
    ssh_client: Any = None
    remote_temp_dir: str | None = None
    remote_files: list[str] = field(default_factory=list)

    def add_log(self, text: str) -> None:
        clean = text.strip()
        if not clean:
            return
        with self.condition:
            self.logs.append(clean)
            self.condition.notify_all()

    def publish_frame(self, frame: bytes) -> None:
        with self.condition:
            self.latest_frame = frame
            self.frame_version += 1
            self.frame_count += 1
            if self.status == "starting":
                self.status = "running"
                self.message = "远端推理运行中，正在接收实时画面。"
            self.condition.notify_all()

    def update_upload(self, stage: str, transferred: int, total: int) -> None:
        progress = min(100, round(transferred * 100 / total)) if total > 0 else 0
        with self.condition:
            self.status = "uploading"
            self.transfer_stage = stage
            self.upload_progress = progress
            self.message = f"正在上传{stage}… {progress}%"
            self.condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {
                "id": self.id,
                "host": self.host,
                "username": self.username,
                "status": self.status,
                "message": self.message,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "frameCount": self.frame_count,
                "logs": list(self.logs),
                "error": self.error,
                "transferStage": self.transfer_stage,
                "uploadProgress": self.upload_progress,
                "streamUrl": f"/api/remote-inference/{self.id}/stream",
            }

    def wait_for_frame(self, version: int, timeout: float = 1.0) -> tuple[int, bytes | None, str]:
        with self.condition:
            if self.frame_version == version and self.status not in TERMINAL_STATES:
                self.condition.wait(timeout)
            return self.frame_version, self.latest_frame, self.status


class RemoteInferenceManager:
    def __init__(self, credential_store: CredentialStore | None = None) -> None:
        self._sessions: dict[str, RemoteInferenceSession] = {}
        self._lock = threading.RLock()
        self._credentials = credential_store or CredentialStore()

    @staticmethod
    def _load_paramiko():
        try:
            import paramiko  # type: ignore

            return paramiko
        except ImportError as exc:
            raise RuntimeError(
                "本机尚未安装 SSH 组件，请先运行：pip install -r requirements.txt"
            ) from exc

    @staticmethod
    def _connection_error(exc: Exception, paramiko: Any, host: str, port: int) -> str:
        if isinstance(exc, paramiko.AuthenticationException):
            return "SSH 身份验证失败：请检查用户名和密码，或确认设备允许密码登录。"
        if isinstance(exc, paramiko.BadHostKeyException):
            return "设备 SSH 指纹与本机已记录的指纹不一致，请确认设备身份和 known_hosts 配置。"
        if isinstance(exc, paramiko.ssh_exception.NoValidConnectionsError):
            nested_errors = list(getattr(exc, "errors", {}).values())
            if any(isinstance(error, ConnectionRefusedError) for error in nested_errors):
                return f"连接被拒绝：{host}:{port} 没有开放 SSH 服务，请检查端口和 sshd 状态。"
            if any(getattr(error, "errno", None) in {101, 113} for error in nested_errors):
                return f"网络不可达：本机无法访问 {host}:{port}，请检查网线、Wi-Fi 和设备 IP。"
            return f"无法连接 {host}:{port}：请确认设备在线、IP 和 SSH 端口正确。"
        if isinstance(exc, (socket.timeout, TimeoutError)):
            return f"连接超时：{host}:{port} 未响应，请检查设备是否在线以及网络是否互通。"
        if isinstance(exc, socket.gaierror):
            return f"无法解析设备地址“{host}”，请检查 IP 或主机名。"
        if isinstance(exc, paramiko.SSHException):
            details = str(exc).strip()
            return f"SSH 握手失败：{details or '设备 SSH 服务返回了异常响应。'}"
        if isinstance(exc, OSError):
            details = str(exc).strip()
            return f"网络连接失败：{details or f'无法访问 {host}:{port}。'}"
        details = str(exc).strip()
        return f"SSH 连接失败：{details or exc.__class__.__name__}"

    def test_connection(self, raw: dict[str, Any]) -> dict[str, Any]:
        config = validate_connection_payload(raw, self._credentials)
        paramiko = self._load_paramiko()
        client = paramiko.SSHClient()
        connected = False
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=config["host"],
                port=config["port"],
                username=config["username"],
                password=config["password"],
                look_for_keys=False,
                allow_agent=False,
                timeout=12,
                banner_timeout=12,
                auth_timeout=12,
            )
            connected = True
            transport = client.get_transport()
            key = transport.get_remote_server_key() if transport is not None else None
            fingerprint = ":".join(f"{byte:02x}" for byte in key.get_fingerprint()) if key else "未知"
            if config["remember_password"]:
                self._credentials.set(
                    config["host"], config["port"], config["username"], config["password"]
                )
            message = (
                "SSH 连接成功，密码已加密保存在本机。"
                if config["remember_password"]
                else "SSH 连接成功，可以开始实时检测。"
            )
            return {
                "ok": True,
                "message": message,
                "host": config["host"],
                "port": config["port"],
                "username": config["username"],
                "fingerprint": fingerprint,
                "passwordRemembered": config["remember_password"],
                "usedSavedPassword": config["password_from_store"],
                "testedAt": datetime.now().isoformat(timespec="seconds"),
            }
        except Exception as exc:
            if connected:
                raise ValueError(f"SSH 连接成功，但本机密码保存失败：{exc}") from exc
            raise ValueError(
                self._connection_error(exc, paramiko, config["host"], config["port"])
            ) from exc
        finally:
            config["password"] = ""
            client.close()

    def forget_password(self, raw: dict[str, Any]) -> dict[str, Any]:
        identity = validate_connection_identity(raw)
        removed = self._credentials.delete(
            identity["host"], identity["port"], identity["username"]
        )
        return {
            "ok": True,
            "removed": removed,
            "message": "已清除该设备保存的 SSH 密码。" if removed else "该设备没有已保存的 SSH 密码。",
        }

    def start(self, raw: dict[str, Any]) -> dict[str, Any]:
        raw_parameters = raw.get("parameters", raw)
        raw_paths = raw.get("paths", {})
        if not isinstance(raw_parameters, dict) or not isinstance(raw_paths, dict):
            raise ValueError("远端推理参数格式不正确。")
        config = validate_start_payload(raw_parameters, raw_paths, self._credentials)
        self._load_paramiko()
        session = RemoteInferenceSession(
            id=uuid.uuid4().hex[:16],
            host=config["host"],
            username=config["username"],
        )
        with self._lock:
            self._sessions[session.id] = session
        threading.Thread(
            target=self._run,
            args=(session, config),
            name=f"remote-inference-{session.id}",
            daemon=True,
        ).start()
        return session.snapshot()

    def get(self, session_id: str) -> RemoteInferenceSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("远端推理任务不存在或已经失效。")
        return session

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        session.stop_event.set()
        with session.condition:
            if session.status not in TERMINAL_STATES:
                session.status = "stopping"
                session.message = "正在停止远端推理……"
            session.condition.notify_all()
        client = session.ssh_client
        if client is not None:
            try:
                pattern = f"[y]olo-session-{session.id}"
                client.exec_command(f"pkill -TERM -f -- {shlex.quote(pattern)}", timeout=3)
            except Exception:
                pass
        return session.snapshot()

    def stop_all(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            try:
                self.stop(session_id)
            except ValueError:
                pass

    @staticmethod
    def _command(config: dict[str, Any], remote_worker: str, token: str) -> str:
        arguments: list[str] = [
            config["python_path"],
            "-B",
            "-u",
            remote_worker,
            "--script",
            config["script_path"],
            "--model",
            config["remote_model_file"],
            "--source",
            config["remote_video_file"],
            "--labels",
            config["labels"],
            "--conf",
            str(config["conf"]),
            "--iou",
            str(config["iou"]),
            "--max-det",
            str(config["max_det"]),
            "--preview-fps",
            str(config["preview_fps"]),
            "--stream-width",
            str(config["stream_width"]),
            "--jpeg-quality",
            str(config["jpeg_quality"]),
            "--cuda-backend",
            config["cuda_backend"],
            "--session-token",
            token,
        ]
        if config["save_path"]:
            arguments.extend(["--save", config["save_path"]])
        if config["realtime"]:
            arguments.append("--realtime")
        return "exec " + " ".join(shlex.quote(str(value)) for value in arguments)

    @staticmethod
    def _drain_stderr(channel: Any, session: RemoteInferenceSession, pending: bytearray) -> None:
        while channel.recv_stderr_ready():
            pending.extend(channel.recv_stderr(65536))
        while b"\n" in pending:
            line, _, remainder = pending.partition(b"\n")
            pending[:] = remainder
            session.add_log(line.decode("utf-8", errors="replace"))

    @staticmethod
    def _upload_file(
        sftp: Any,
        local_path: Path,
        remote_path: str,
        stage: str,
        session: RemoteInferenceSession,
    ) -> None:
        last_reported = -1

        def progress(transferred: int, total: int) -> None:
            nonlocal last_reported
            if session.stop_event.is_set():
                raise RuntimeError("上传已取消。")
            percent = min(100, round(transferred * 100 / total)) if total > 0 else 0
            if percent == 100 or percent >= last_reported + 2:
                last_reported = percent
                session.update_upload(stage, transferred, total)

        session.update_upload(stage, 0, local_path.stat().st_size)
        sftp.put(str(local_path), remote_path, callback=progress)
        session.add_log(f"{stage}上传完成：{local_path.name}")

    def _run(self, session: RemoteInferenceSession, config: dict[str, Any]) -> None:
        client = None
        sftp = None
        channel = None
        stderr_pending = bytearray()
        try:
            paramiko = self._load_paramiko()
            client = paramiko.SSHClient()
            client.load_system_host_keys()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            session.ssh_client = client
            try:
                client.connect(
                    hostname=config["host"],
                    port=config["port"],
                    username=config["username"],
                    password=config["password"],
                    look_for_keys=False,
                    allow_agent=False,
                    timeout=12,
                    banner_timeout=12,
                    auth_timeout=12,
                )
            except Exception as exc:
                raise RuntimeError(
                    self._connection_error(exc, paramiko, config["host"], config["port"])
                ) from exc
            if config["remember_password"]:
                try:
                    self._credentials.set(
                        config["host"], config["port"], config["username"], config["password"]
                    )
                except Exception as exc:
                    session.add_log(f"警告：密码保存失败，本次推理仍会继续：{exc}")
            # Remove the only in-memory copy as soon as authentication finishes.
            config["password"] = ""
            transport = client.get_transport()
            key = transport.get_remote_server_key() if transport is not None else None
            fingerprint = ":".join(f"{byte:02x}" for byte in key.get_fingerprint()) if key else "未知"
            session.add_log(f"SSH 已连接：{config['username']}@{config['host']}:{config['port']}")
            session.add_log(f"设备指纹：{fingerprint}")

            remote_dir = f"/tmp/yolo_remote_inference_{session.id}"
            remote_worker = f"{remote_dir}/worker.py"
            remote_script = f"{remote_dir}/run_star_video.py"
            remote_model = f"{remote_dir}/model{config['local_model_file'].suffix.lower()}"
            remote_video = f"{remote_dir}/input{config['local_video_file'].suffix.lower()}"
            session.remote_temp_dir = remote_dir
            session.remote_files = [remote_worker, remote_script, remote_model, remote_video]
            sftp = client.open_sftp()
            sftp.mkdir(remote_dir)
            sftp.put(str(REMOTE_WORKER), remote_worker)
            sftp.chmod(remote_worker, 0o700)
            self._upload_file(
                sftp,
                config["local_script_file"],
                remote_script,
                "推理脚本",
                session,
            )
            self._upload_file(
                sftp,
                config["local_model_file"],
                remote_model,
                "模型文件",
                session,
            )
            self._upload_file(
                sftp,
                config["local_video_file"],
                remote_video,
                "视频文件",
                session,
            )
            sftp.close()
            sftp = None
            config["script_path"] = remote_script
            config["remote_model_file"] = remote_model
            config["remote_video_file"] = remote_video
            session.add_log("本地推理脚本、模型和视频已上传至任务临时目录。")

            token = f"yolo-session-{session.id}"
            command = self._command(config, remote_worker, token)
            _stdin, stdout, _stderr = client.exec_command(command, get_pty=False)
            channel = stdout.channel
            with session.condition:
                session.status = "starting"
                session.message = "设备已连接，正在加载模型并等待第一帧……"
                session.condition.notify_all()

            stdout_buffer = bytearray()
            expected_size: int | None = None
            while not session.stop_event.is_set():
                received = False
                if channel.recv_ready():
                    chunk = channel.recv(65536)
                    if chunk:
                        stdout_buffer.extend(chunk)
                        received = True
                self._drain_stderr(channel, session, stderr_pending)

                while True:
                    if expected_size is None:
                        if len(stdout_buffer) < 4:
                            break
                        expected_size = struct.unpack(">I", stdout_buffer[:4])[0]
                        del stdout_buffer[:4]
                        if expected_size <= 0 or expected_size > MAX_FRAME_BYTES:
                            raise RuntimeError("远端画面数据格式异常，请确认脚本路径指向兼容版本。")
                    if len(stdout_buffer) < expected_size:
                        break
                    frame = bytes(stdout_buffer[:expected_size])
                    del stdout_buffer[:expected_size]
                    expected_size = None
                    session.publish_frame(frame)

                if channel.exit_status_ready() and not channel.recv_ready():
                    break
                if not received:
                    time.sleep(0.01)

            self._drain_stderr(channel, session, stderr_pending)
            if stderr_pending:
                session.add_log(stderr_pending.decode("utf-8", errors="replace"))
            exit_code = channel.recv_exit_status() if channel.exit_status_ready() else None
            with session.condition:
                if session.stop_event.is_set():
                    session.status = "stopped"
                    session.message = "远端推理已停止。"
                elif exit_code == 0:
                    session.status = "completed"
                    session.message = f"视频处理完成，共接收 {session.frame_count} 帧预览。"
                else:
                    raise RuntimeError(f"远端推理异常结束（退出码 {exit_code}）。")
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.condition.notify_all()
        except Exception as exc:
            config["password"] = ""
            message = str(exc).strip() or exc.__class__.__name__
            with session.condition:
                session.status = "stopped" if session.stop_event.is_set() else "failed"
                session.message = "远端推理已停止。" if session.stop_event.is_set() else "远端推理启动或运行失败。"
                session.error = None if session.stop_event.is_set() else message
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.logs.append(f"错误：{message}")
                session.condition.notify_all()
        finally:
            if channel is not None:
                try:
                    channel.close()
                except Exception:
                    pass
            if sftp is not None:
                try:
                    sftp.close()
                except Exception:
                    pass
            if client is not None:
                remote_temp_dir = session.remote_temp_dir
                if remote_temp_dir:
                    try:
                        cleanup = client.open_sftp()
                        for remote_file in reversed(session.remote_files):
                            try:
                                cleanup.remove(remote_file)
                            except OSError:
                                pass
                        cleanup.rmdir(remote_temp_dir)
                        cleanup.close()
                        session.add_log("远端临时脚本、模型、视频和适配器已清理。")
                    except Exception:
                        pass
                try:
                    client.close()
                except Exception:
                    pass
            session.ssh_client = None


manager = RemoteInferenceManager()


def run_remote_inference_placeholder(_context: RunContext) -> dict[str, Any]:
    raise ValueError("该功能需要使用“连接并开始”实时运行按钮。")


register_handler("remote.star_inference", run_remote_inference_placeholder)
