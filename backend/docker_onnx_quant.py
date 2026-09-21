"""Run EasyONNXQuant in an existing local Docker container."""

from __future__ import annotations

import json
import math
import os
import queue
import re
import shlex
import shutil
import subprocess
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "docker.quantize_onnx"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_CONTAINER = "onnx2quant"
DEFAULT_ENVIRONMENT = "onnxquant"
DEFAULT_CONDA = "/root/anaconda3/bin/conda"
DEFAULT_EXCLUDE = r"^/model\.(13|16|17|23)(?:/|$).*"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass(frozen=True)
class DockerMount:
    source: Path
    destination: PurePosixPath
    writable: bool


def _safe_name(raw: Any, label: str, default: str) -> str:
    value = str(raw or default).strip()
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label}只能包含字母、数字、点、下划线和短横线。")
    return value


def _positive_integer(raw: Any, label: str, default: int) -> int:
    try:
        value = float(default if raw in (None, "") else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not math.isfinite(value) or not value.is_integer() or not 1 <= value <= 1_000_000:
        raise ValueError(f"{label}应为 1 到 1000000 之间的整数。")
    return int(value)


def _docker_capture(arguments: list[str], timeout: float = 30) -> str:
    if not shutil.which("docker"):
        raise RuntimeError("本机未找到 Docker 命令，请先安装 Docker。")
    completed = subprocess.run(
        ["docker", *arguments], capture_output=True, text=True, timeout=timeout, check=False
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"Docker 命令执行失败（{completed.returncode}）。")
    return completed.stdout.strip()


def _ensure_container(container: str, report: Any) -> bool:
    try:
        running = _docker_capture(["inspect", "--format", "{{.State.Running}}", container]).lower() == "true"
    except RuntimeError as exc:
        raise RuntimeError(f"找不到或无法访问 Docker 容器 {container}：{exc}") from exc
    if running:
        report(f"Docker 容器 {container} 已在运行。")
        return False
    report(f"Docker 容器 {container} 当前已停止，正在自动启动……")
    _docker_capture(["start", container], timeout=60)
    report(f"Docker 容器 {container} 已启动。")
    return True


def _container_mounts(container: str) -> list[DockerMount]:
    raw = _docker_capture(["inspect", "--format", "{{json .Mounts}}", container])
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("无法解析 Docker 容器挂载信息。") from exc
    mounts = []
    for item in values if isinstance(values, list) else []:
        if not isinstance(item, dict) or item.get("Type") != "bind":
            continue
        source = Path(str(item.get("Source", ""))).expanduser().resolve()
        destination = PurePosixPath(str(item.get("Destination", "")))
        if str(source) and destination.is_absolute():
            mounts.append(DockerMount(source, destination, bool(item.get("RW", False))))
    return mounts


def _container_path(host_path: Path, mounts: list[DockerMount], *, writable: bool = False) -> str:
    resolved = host_path.expanduser().resolve()
    candidates = [
        mount for mount in mounts
        if (resolved == mount.source or mount.source in resolved.parents) and (mount.writable or not writable)
    ]
    if not candidates:
        mode = "可写挂载目录" if writable else "挂载目录"
        available = "、".join(str(mount.source) for mount in mounts if mount.writable or not writable) or "无"
        raise ValueError(f"路径不在容器的{mode}内：{resolved}。当前可用：{available}")
    mount = max(candidates, key=lambda item: len(item.source.parts))
    relative = resolved.relative_to(mount.source)
    return str(mount.destination.joinpath(*relative.parts))


def _test_environment(container: str, conda_path: str, environment: str) -> str:
    code = (
        "import sys, easyonnxquant; "
        "print('Python=' + sys.executable); "
        "print('easyonnxquant=' + str(getattr(easyonnxquant, '__version__', '已安装')))"
    )
    return _docker_capture([
        "exec", container, conda_path, "run", "--no-capture-output", "-n", environment,
        "python", "-c", code,
    ], timeout=60)


def _tool_arguments(config: dict[str, Any]) -> list[str]:
    arguments = [
        config["conda_path"], "run", "--no-capture-output", "-n", config["environment"],
        "python", "-u", "-m", "easyonnxquant.tool",
        "--input", config["container_input"],
        "--img_dir", config["container_images"],
        "--type", config["model_type"],
        "--limit", str(config["limit"]),
        "--output", config["container_output"],
    ]
    if config["method"]:
        arguments.extend(["--method", config["method"]])
    if config["nodes_to_exclude"]:
        arguments.append("--nodes_to_exclude")
        arguments.extend(config["nodes_to_exclude"])
    return arguments


def _validate_export(context: RunContext, mounts: list[DockerMount], container: str,
                     environment: str, conda_path: str) -> dict[str, Any]:
    input_model = context.paths.get("input_onnx")
    image_folder = context.paths.get("calibration_images")
    if not input_model or input_model.suffix.lower() != ".onnx" or not input_model.is_file():
        raise ValueError("请选择存在的输入 ONNX 模型。")
    if not image_folder or not image_folder.is_dir():
        raise ValueError("请选择存在的量化图片文件夹。")
    image_count = sum(1 for path in image_folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if image_count == 0:
        raise ValueError("量化图片文件夹中没有 JPG、PNG、BMP 或 WEBP 图片。")
    output_folder = context.paths.get("output_folder") or input_model.parent
    if not output_folder.is_dir():
        raise ValueError("输出文件夹不存在。")
    raw_output_name = str(context.parameters.get("output_filename", "")).strip()
    output_name = raw_output_name or f"{input_model.stem}_quant.onnx"
    if Path(output_name).name != output_name or not output_name.lower().endswith(".onnx"):
        raise ValueError("输出文件名只能填写一个以 .onnx 结尾的文件名。")
    output_model = (output_folder / output_name).resolve()
    if output_model == input_model.resolve():
        raise ValueError("量化输出不能覆盖输入模型。")
    if output_model.exists():
        raise ValueError(f"量化输出已存在：{output_model}。请修改输出文件名，避免覆盖。")
    model_type = str(context.parameters.get("model_type", "yolo")).strip().lower()
    if model_type not in {"yolo", "mobilenet", "custom"}:
        raise ValueError("模型类型只能选择 yolo、mobilenet 或 custom。")
    method_raw = str(context.parameters.get("method", "默认")).strip().lower()
    method = "" if method_raw in {"", "默认"} else method_raw
    if method not in {"", "minmax", "entropy"}:
        raise ValueError("量化方法只能选择默认、minmax 或 entropy。")
    raw_excluded = str(context.parameters.get("nodes_to_exclude", DEFAULT_EXCLUDE)).strip()
    excluded = [line.strip() for line in raw_excluded.splitlines() if line.strip()]
    return {
        "container": container,
        "environment": environment,
        "conda_path": conda_path,
        "input_model": input_model.resolve(),
        "image_folder": image_folder.resolve(),
        "output_model": output_model,
        "container_input": _container_path(input_model, mounts),
        "container_images": _container_path(image_folder, mounts),
        "container_output": _container_path(output_model, mounts, writable=True),
        "model_type": model_type,
        "limit": _positive_integer(context.parameters.get("limit"), "量化图片上限", 128),
        "method": method,
        "nodes_to_exclude": excluded,
        "image_count": image_count,
    }


def _terminate_container_process(container: str, pid_file: str) -> None:
    script = f"if test -f {shlex.quote(pid_file)}; then kill -TERM -- -$(cat {shlex.quote(pid_file)}) 2>/dev/null || true; fi"
    try:
        subprocess.run(["docker", "exec", container, "bash", "-lc", script],
                       capture_output=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        pass


def _run_export(config: dict[str, Any], context: RunContext) -> None:
    token = f"processing-view-onnx-{uuid.uuid4().hex[:12]}"
    pid_file = f"/tmp/{token}.pid"
    command = shlex.join(_tool_arguments(config))
    script = (
        "set -o pipefail; "
        f"trap 'rm -f {shlex.quote(pid_file)}' EXIT; "
        f"setsid {command} & child=$!; printf '%s\\n' \"$child\" > {shlex.quote(pid_file)}; wait \"$child\""
    )
    process = subprocess.Popen(
        ["docker", "exec", config["container"], "bash", "-lc", script],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    threading.Thread(target=read_output, name=f"onnx-quant-log-{token}", daemon=True).start()
    stream_closed = False
    try:
        while not stream_closed or process.poll() is None:
            context.check_cancelled()
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                stream_closed = True
            elif line:
                context.report(line)
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Docker ONNX 量化异常结束（退出码 {return_code}）。")
    except TaskCancelled:
        _terminate_container_process(config["container"], pid_file)
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        config["output_model"].unlink(missing_ok=True)
        raise


def run_docker_onnx_quant(context: RunContext) -> dict[str, Any]:
    container = _safe_name(context.parameters.get("container_name"), "容器名称", DEFAULT_CONTAINER)
    environment = _safe_name(context.parameters.get("conda_environment"), "Conda 环境名称", DEFAULT_ENVIRONMENT)
    conda_path = str(context.parameters.get("conda_path", DEFAULT_CONDA)).strip() or DEFAULT_CONDA
    if not conda_path.startswith("/") or "\x00" in conda_path:
        raise ValueError("容器内 Conda 路径必须是绝对路径。")
    started = _ensure_container(container, context.report)
    context.check_cancelled()
    environment_info = _test_environment(container, conda_path, environment)
    context.report(f"量化环境检查成功：{environment_info.replace(os.linesep, '；')}")
    operation = str(context.parameters.get("operation", "导出量化 ONNX"))
    if operation not in {"导出量化 ONNX", "测试 Docker 环境"}:
        raise ValueError("请选择有效的操作方式。")
    if operation == "测试 Docker 环境":
        return {
            "message": f"Docker 量化环境可用：容器 {container}，Conda 环境 {environment}。",
            "container": container, "environment": environment, "containerStarted": started,
        }

    mounts = _container_mounts(container)
    config = _validate_export(context, mounts, container, environment, conda_path)
    context.report(
        f"输入模型：{config['input_model']} → {config['container_input']}\n"
        f"量化图片：{config['image_folder']} → {config['container_images']}（找到 {config['image_count']} 张，上限 {config['limit']}）\n"
        f"输出模型：{config['output_model']} → {config['container_output']}"
    )
    context.report(f"开始量化：type={config['model_type']}，method={config['method'] or '工具默认值'}")
    try:
        _run_export(config, context)
    except Exception:
        if config["output_model"].exists():
            config["output_model"].unlink(missing_ok=True)
        raise
    if not config["output_model"].is_file() or config["output_model"].stat().st_size == 0:
        raise RuntimeError("量化命令已结束，但没有生成有效的 ONNX 输出文件。")
    size_mb = config["output_model"].stat().st_size / 1024 / 1024
    context.report(f"量化完成：{config['output_model']}（{size_mb:.1f} MB）")
    return {
        "message": f"量化 ONNX 导出完成：{config['output_model'].name}。",
        "outputFiles": [str(config["output_model"])],
        "outputFolders": [str(config["output_model"].parent)],
        "container": container,
        "environment": environment,
        "imageCount": config["image_count"],
        "limit": config["limit"],
        "modelType": config["model_type"],
    }


register_handler(HANDLER_ID, run_docker_onnx_quant)
