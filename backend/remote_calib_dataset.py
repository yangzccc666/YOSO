"""Upload and run the calibration selector against server-side image folders."""

from __future__ import annotations

import json
import shlex
import time
import uuid
from types import SimpleNamespace
from pathlib import Path, PurePosixPath
from typing import Any

from .handlers import RunContext, register_handler
from .remote_inference import RemoteInferenceManager
from .remote_yolo_dataset import _lines_from, resolve_tested_yolo_connection
from .remote_calib_worker import run as run_calibration_worker
from .task_manager import TaskCancelled


WORKER_SCRIPT = Path(__file__).with_name("remote_calib_worker.py")


def is_remote_calib_parameters(parameters: dict[str, Any]) -> bool:
    return str(parameters.get("execution_location") or "本地运行").strip() == "SSH 远程服务器"


def _directories(value: Any, label: str, required: bool = False, *, remote: bool) -> list[str]:
    entries = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if required and not entries:
        raise ValueError(f"请至少填写一个{'远程' if remote else '本机'}{label}，每行一个路径。")
    for entry in entries:
        valid = PurePosixPath(entry).is_absolute() if remote else Path(entry).expanduser().is_absolute()
        if not valid or "\x00" in entry:
            raise ValueError(f"{'远程' if remote else '本机'}{label}必须是绝对路径：{entry}")
    return entries


def _integer(value: Any, label: str, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(default if value in (None, "") else value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not minimum <= result <= maximum:
        raise ValueError(f"{label}必须在 {minimum} 到 {maximum} 之间。")
    return result


def _configuration(context: RunContext) -> dict[str, Any]:
    parameters = context.parameters
    remote = is_remote_calib_parameters(parameters)
    images = _directories(parameters.get("image_dirs"), "图片目录", True, remote=remote)
    annotations = _directories(parameters.get("annotation_dirs"), "标注目录", remote=remote)
    negatives = _directories(parameters.get("negative_dirs"), "负样本图片目录", remote=remote)
    if len(annotations) > len(images):
        raise ValueError("标注目录数量不能多于图片目录数量。")
    output = str(parameters.get("output_dir") or "").strip()
    if not output:
        output = str((PurePosixPath(images[0]) if remote else Path(images[0]).expanduser()).parent / "calib_dataset")
    valid_output = PurePosixPath(output).is_absolute() if remote else Path(output).expanduser().is_absolute()
    if not valid_output or output == "/":
        raise ValueError(f"输出目录必须是{'服务器' if remote else '本机'}上的绝对路径，且不能是根目录。")
    annotation_format = str(parameters.get("annotation_format") or "auto")
    if annotation_format not in {"auto", "xml", "json"}:
        raise ValueError("请选择有效的标注格式。")
    config = {
        "remote": remote,
        "images": images,
        "annotations": annotations,
        "negatives": negatives,
        "output": output,
        "format": annotation_format,
        "count": _integer(parameters.get("num_samples"), "图片数量", 128, 1, 100_000),
        "seed": _integer(parameters.get("random_seed"), "随机种子", 42, -2_147_483_648, 2_147_483_647),
    }
    if remote:
        python = str(parameters.get("remote_python") or "python3").strip()
        if not python:
            raise ValueError("请填写远程 Python 解释器。")
        config.update(resolve_tested_yolo_connection(parameters, "制作量化数据集"))
        config["python"] = python
    return config


def _command(config: dict[str, Any], worker: str, token: str) -> str:
    arguments = [
        config["python"], "-B", "-u", worker,
        "--image-dirs-json", json.dumps(config["images"], ensure_ascii=False),
        "--annotation-dirs-json", json.dumps(config["annotations"], ensure_ascii=False),
        "--negative-dirs-json", json.dumps(config["negatives"], ensure_ascii=False),
        "--output-dir", config["output"], "--format", config["format"],
        "--count", str(config["count"]), "--seed", str(config["seed"]),
    ]
    # The token is part of the script path and allows a targeted cancellation.
    assert token in worker
    return "exec " + " ".join(shlex.quote(str(value)) for value in arguments)


def _run_local_calib_dataset(context: RunContext, config: dict[str, Any]) -> dict[str, Any]:
    context.report("正在使用本机 Python 制作量化数据集。")
    args = SimpleNamespace(
        image_dirs_json=json.dumps(config["images"], ensure_ascii=False),
        annotation_dirs_json=json.dumps(config["annotations"], ensure_ascii=False),
        negative_dirs_json=json.dumps(config["negatives"], ensure_ascii=False),
        output_dir=config["output"], format=config["format"],
        count=config["count"], seed=config["seed"],
    )
    return run_calibration_worker(args, report=context.report, check_cancelled=context.check_cancelled)


def run_remote_calib_dataset(context: RunContext) -> dict[str, Any]:
    config = _configuration(context)
    if not config["remote"]:
        return _run_local_calib_dataset(context, config)
    if not WORKER_SCRIPT.is_file():
        raise RuntimeError("平台内置量化数据集处理脚本缺失，请重新安装平台。")
    paramiko = RemoteInferenceManager._load_paramiko()
    client = paramiko.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    token = f"calib-dataset-{uuid.uuid4().hex[:16]}"
    remote_worker = f"/tmp/{token}.py"
    channel = None
    uploaded = False
    result: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        context.report(f"正在连接远程服务器：{config['username']}@{config['host']}:{config['port']}")
        try:
            client.connect(
                hostname=config["host"], port=int(config["port"]),
                username=config["username"], password=config["password"],
                look_for_keys=False, allow_agent=False,
                timeout=12, banner_timeout=12, auth_timeout=12,
            )
        except Exception as exc:
            raise ValueError(RemoteInferenceManager._connection_error(exc, paramiko, config["host"], int(config["port"]))) from exc
        config["password"] = ""
        context.parameters["remote_password"] = ""
        context.check_cancelled()
        sftp = client.open_sftp()
        try:
            sftp.put(str(WORKER_SCRIPT), remote_worker)
            uploaded = True
        finally:
            sftp.close()
        context.report("内置量化数据集脚本已上传；开始读取服务器图片与标注。")
        _stdin, stdout, _stderr = client.exec_command(_command(config, remote_worker, token), get_pty=False)
        channel = stdout.channel
        stdout_buffer, stderr_buffer = bytearray(), bytearray()
        while True:
            if context.stop_event is not None and context.stop_event.is_set():
                try:
                    client.exec_command(f"pkill -TERM -f -- {shlex.quote(token)}", timeout=3)
                except Exception:
                    pass
                channel.close()
                raise TaskCancelled("远程量化数据集制作已终止；已生成的图片会保留。")
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
            if any("No module named 'PIL'" in line or 'No module named "PIL"' in line for line in errors):
                raise RuntimeError(f"远程 Python 缺少 Pillow，请在 {config['python']} 对应环境安装 Pillow，或更换解释器。")
            raise RuntimeError(errors[-1] if errors else f"远程制作异常结束（退出码 {exit_code}）。")
        if result is None:
            raise RuntimeError("远程脚本已结束，但没有返回结果信息。")
        return result
    finally:
        config["password"] = ""
        context.parameters["remote_password"] = ""
        if channel is not None:
            channel.close()
        if uploaded:
            try:
                sftp = client.open_sftp()
                try:
                    sftp.remove(remote_worker)
                finally:
                    sftp.close()
                context.report("远程临时脚本已清理。")
            except Exception:
                pass
        client.close()


register_handler("calib.select_dataset", run_remote_calib_dataset)
