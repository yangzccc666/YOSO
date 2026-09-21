"""Run YOLO dataset preparation against paths on an SSH server."""

from __future__ import annotations

import json
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from .handlers import RunContext
from .remote_inference import (
    RemoteInferenceManager,
    manager as remote_connection_manager,
)
from .task_manager import TaskCancelled


WORKER_SCRIPT = Path(__file__).with_name("remote_yolo_dataset_worker.py")
STRATEGY_SCRIPT = Path(__file__).with_name("dataset_split_strategy.py")
CONNECTION_TEST_TTL_SECONDS = 30 * 60
_verified_connections: dict[str, tuple[tuple[str, int, str], float]] = {}
_verified_lock = threading.RLock()


def is_remote_yolo_parameters(parameters: dict[str, Any]) -> bool:
    """SSH identity fields opt into remote mode; otherwise processing is local."""
    if str(parameters.get("execution_location", "")).strip() == "SSH 远程服务器":
        return True
    return any((
        str(parameters.get("remote_host", "")).strip(),
        str(parameters.get("remote_username", "")).strip(),
        str(parameters.get("remote_password", "")),
    ))


def _connection_payload(parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "host": parameters.get("remote_host", ""),
        "port": parameters.get("remote_port", 22),
        "username": parameters.get("remote_username", ""),
        "password": parameters.get("remote_password", ""),
        "remember_password": parameters.get("remember_password", True),
    }


def _connection_identity(parameters: dict[str, Any]) -> tuple[str, int, str]:
    raw = _connection_payload(parameters)
    host = str(raw["host"]).strip()
    username = str(raw["username"]).strip()
    try:
        port = int(raw["port"])
    except (TypeError, ValueError):
        port = 0
    return host, port, username


def test_remote_yolo_connection(parameters: dict[str, Any]) -> dict[str, Any]:
    result = remote_connection_manager.test_connection(_connection_payload(parameters))
    token = uuid.uuid4().hex
    identity = (str(result["host"]), int(result["port"]), str(result["username"]))
    now = time.monotonic()
    with _verified_lock:
        expired = [key for key, (_identity, expires_at) in _verified_connections.items() if expires_at <= now]
        for key in expired:
            _verified_connections.pop(key, None)
        _verified_connections[token] = (identity, now + CONNECTION_TEST_TTL_SECONDS)
    result["connectionToken"] = token
    result["message"] = (
        "SSH 连接成功，密码已加密保存在本机，可以执行远程划分。"
        if result["passwordRemembered"]
        else "SSH 连接成功，可以执行远程划分。"
    )
    return result


def remote_yolo_credential_status(parameters: dict[str, Any]) -> dict[str, Any]:
    return remote_connection_manager.credential_status(_connection_payload(parameters))


def forget_remote_yolo_password(parameters: dict[str, Any]) -> dict[str, Any]:
    identity = _connection_identity(parameters)
    result = remote_connection_manager.forget_password(_connection_payload(parameters))
    with _verified_lock:
        for token, (verified_identity, _expires_at) in tuple(_verified_connections.items()):
            if verified_identity == identity:
                _verified_connections.pop(token, None)
    return result


def require_tested_yolo_connection(
    parameters: dict[str, Any],
    action: str = "执行远程数据集划分",
) -> None:
    token = str(parameters.get("remote_connection_token", "")).strip()
    identity = _connection_identity(parameters)
    now = time.monotonic()
    with _verified_lock:
        verified = _verified_connections.get(token)
        if verified is not None and verified[1] > now and verified[0] == identity:
            return
        if token:
            _verified_connections.pop(token, None)
    raise ValueError(f"请先测试 SSH 连接，连接成功后再{action}。")


def resolve_tested_yolo_connection(
    parameters: dict[str, Any],
    action: str = "执行远程数据集划分",
) -> dict[str, Any]:
    require_tested_yolo_connection(parameters, action)
    return remote_connection_manager.resolve_connection(_connection_payload(parameters))


def _integer(value: Any, label: str, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not minimum <= number <= maximum:
        raise ValueError(f"{label}必须在 {minimum} 到 {maximum} 之间。")
    return number


def _float(value: Any, label: str, default: float) -> float:
    try:
        number = float(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc
    return number


def _remote_path(context: RunContext, field_id: str, label: str, required: bool = True) -> str:
    value = str(context.paths.get(field_id, "")).strip()
    if required and not value:
        raise ValueError(f"请填写远程服务器上的{label}。")
    return value


def _configuration(context: RunContext) -> dict[str, Any]:
    connection = resolve_tested_yolo_connection(context.parameters)
    host = str(connection["host"])
    username = str(connection["username"])
    password = str(connection["password"])
    python_path = str(context.parameters.get("remote_python", "python3")).strip()
    if not python_path:
        raise ValueError("请输入远程服务器 Python 解释器。")

    task_type = str(context.parameters.get("dataset_task_type", "目标检测"))
    if task_type != "目标检测":
        raise ValueError(f"“{task_type}”远程处理逻辑尚未接入；当前可使用“目标检测”。")
    label_format = str(context.parameters.get("label_format", "VOC XML"))
    format_map = {"YOLO TXT": "yolo_txt", "VOC XML": "voc_xml", "仅图片": "images_only"}
    if label_format not in format_map:
        raise ValueError("请选择有效的目标检测标签格式。")
    ratio = _float(context.parameters.get("train_ratio"), "训练集比例", 0.7)
    if not 0 < ratio < 1:
        raise ValueError("训练集比例必须大于 0 且小于 1。")
    missing_policy = str(context.parameters.get("missing_label_policy", "跳过并报告"))
    if missing_policy not in {"跳过并报告", "生成空标签"}:
        raise ValueError("请选择有效的缺少标签处理方式。")

    labels_required = label_format != "仅图片"
    return {
        "host": host,
        "port": int(connection["port"]),
        "username": username,
        "password": password,
        "python_path": python_path,
        "images_folder": _remote_path(context, "images_folder", "图片文件夹"),
        "labels_folder": _remote_path(context, "labels_folder", "标签文件夹", labels_required),
        "negative_folder": _remote_path(context, "negative_images_folder", "负样本图片文件夹", False),
        "output_parent": _remote_path(context, "output_folder", "输出位置"),
        "dataset_folder_name": str(context.parameters.get("dataset_folder_name", "yolo_train")).strip() or "yolo_train",
        "label_format": format_map[label_format],
        "train_ratio": ratio,
        "seed": _integer(context.parameters.get("random_seed"), "随机种子", 42, -2_147_483_648, 2_147_483_647),
        "missing_policy": "skip" if missing_policy == "跳过并报告" else "empty",
    }


def _command(config: dict[str, Any], remote_worker: str, token: str) -> str:
    arguments = [
        config["python_path"], "-B", "-u", remote_worker,
        "--images-folder", config["images_folder"],
        "--labels-folder", config["labels_folder"],
        "--negative-folder", config["negative_folder"],
        "--output-parent", config["output_parent"],
        "--dataset-folder-name", config["dataset_folder_name"],
        "--label-format", config["label_format"],
        "--train-ratio", str(config["train_ratio"]),
        "--seed", str(config["seed"]),
        "--missing-policy", config["missing_policy"],
        "--session-token", token,
    ]
    return "exec " + " ".join(shlex.quote(str(value)) for value in arguments)


def _lines_from(buffer: bytearray, final: bool = False) -> list[str]:
    lines: list[str] = []
    while b"\n" in buffer:
        raw, _, remainder = buffer.partition(b"\n")
        buffer[:] = remainder
        lines.append(raw.decode("utf-8", errors="replace").strip())
    if final and buffer:
        lines.append(bytes(buffer).decode("utf-8", errors="replace").strip())
        buffer.clear()
    return [line for line in lines if line]


def run_remote_yolo_dataset(context: RunContext) -> dict[str, Any]:
    config = _configuration(context)
    if not WORKER_SCRIPT.is_file():
        raise RuntimeError("平台内置远程数据集处理脚本缺失，请重新安装平台。")
    paramiko = RemoteInferenceManager._load_paramiko()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    token = f"yolo-dataset-{uuid.uuid4().hex[:16]}"
    remote_dir = f"/tmp/{token}"
    remote_worker = f"{remote_dir}/worker.py"
    remote_strategy = f"{remote_dir}/dataset_split_strategy.py"
    channel = None
    result: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        context.report(f"正在连接远程服务器：{config['username']}@{config['host']}:{config['port']}")
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
            raise ValueError(
                RemoteInferenceManager._connection_error(
                    exc, paramiko, config["host"], config["port"]
                )
            ) from exc
        config["password"] = ""
        context.parameters["remote_password"] = ""
        transport = client.get_transport()
        key = transport.get_remote_server_key() if transport is not None else None
        fingerprint = ":".join(f"{byte:02x}" for byte in key.get_fingerprint()) if key else "未知"
        context.report(f"SSH 已连接，设备指纹：{fingerprint}")
        context.check_cancelled()

        sftp = client.open_sftp()
        try:
            sftp.mkdir(remote_dir)
            sftp.put(str(WORKER_SCRIPT), remote_worker)
            sftp.put(str(STRATEGY_SCRIPT), remote_strategy)
            sftp.chmod(remote_worker, 0o700)
        finally:
            sftp.close()
        context.report("远程数据集处理脚本和类别保护算法上传完成，开始检查路径并划分数据。")

        _stdin, stdout, _stderr = client.exec_command(_command(config, remote_worker, token), get_pty=False)
        channel = stdout.channel
        stdout_buffer = bytearray()
        stderr_buffer = bytearray()
        while True:
            if context.stop_event is not None and context.stop_event.is_set():
                try:
                    client.exec_command(f"pkill -TERM -f -- {shlex.quote(token)}", timeout=3)
                except Exception:
                    pass
                if channel is not None:
                    channel.close()
                raise TaskCancelled("远程 YOLO 数据集划分已被用户终止。已复制完成的文件会保留。")

            received = False
            if channel.recv_ready():
                stdout_buffer.extend(channel.recv(65536))
                received = True
            if channel.recv_stderr_ready():
                stderr_buffer.extend(channel.recv_stderr(65536))
                received = True
            for line in _lines_from(stdout_buffer):
                if line.startswith("RESULT_JSON:"):
                    result = json.loads(line.removeprefix("RESULT_JSON:"))
                else:
                    context.report(line)
            for line in _lines_from(stderr_buffer):
                errors.append(line.removeprefix("ERROR:").strip() if line.startswith("ERROR:") else line)
                context.report(f"远程：{line}")
            if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                break
            if not received:
                time.sleep(0.05)

        for line in _lines_from(stdout_buffer, final=True):
            if line.startswith("RESULT_JSON:"):
                result = json.loads(line.removeprefix("RESULT_JSON:"))
            else:
                context.report(line)
        for line in _lines_from(stderr_buffer, final=True):
            errors.append(line.removeprefix("ERROR:").strip() if line.startswith("ERROR:") else line)
            context.report(f"远程：{line}")
        exit_code = channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(errors[-1] if errors else f"远程处理异常结束（退出码 {exit_code}）。")
        if result is None:
            raise RuntimeError("远程处理已结束，但没有返回结果信息。")
        return result
    finally:
        config["password"] = ""
        context.parameters["remote_password"] = ""
        if channel is not None:
            try:
                channel.close()
            except Exception:
                pass
        try:
            sftp = client.open_sftp()
            try:
                for temporary_file in (remote_worker, remote_strategy):
                    try:
                        sftp.remove(temporary_file)
                    except FileNotFoundError:
                        pass
                try:
                    sftp.rmdir(remote_dir)
                except FileNotFoundError:
                    pass
            finally:
                sftp.close()
            context.report("远程临时处理脚本已清理。")
        except Exception:
            pass
        client.close()
