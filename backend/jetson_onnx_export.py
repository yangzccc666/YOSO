"""Export local YOLO26 PT weights to raw-head ONNX for Jetson/TensorRT."""

from __future__ import annotations

import math
import os
import queue
import shutil
import signal
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .local_pt_inference import _find_yolo_python
from .task_manager import TaskCancelled


HANDLER_ID = "model.export_jetson_onnx"
WORKER_SCRIPT = Path(__file__).with_name("bundled_export_onnx_cut.py")
TASKS = {
    "目标检测": "detect",
    "目标分割": "segment",
    "旋转目标检测": "obb",
    "姿态估计": "pose",
    "图像分类": "classify",
}
PLATFORMS = {"Jetson / TensorRT（NCHW）": "jetson", "RKNN（NHWC）": "rknn"}
MODEL_FAMILIES = {"YOLO26", "YOLOv5"}


def _integer(raw: Any, label: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = float(default if raw in (None, "") else raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是整数。") from exc
    if not math.isfinite(value) or not value.is_integer() or not minimum <= value <= maximum:
        raise ValueError(f"{label}应为 {minimum} 到 {maximum} 之间的整数。")
    return int(value)


def _resolve_python(raw: Any) -> Path:
    text = str(raw or "").strip()
    if not text:
        return _find_yolo_python()
    path = Path(text).expanduser()
    if not path.is_file():
        raise ValueError(f"YOLO Python 不存在：{path}")
    return path.resolve()


def _configuration(context: RunContext) -> dict[str, Any]:
    weights = context.paths.get("weights_file")
    if not weights or weights.suffix.lower() != ".pt" or not weights.is_file():
        raise ValueError("请选择存在的本地 PT 权重文件。")
    output_folder = context.paths.get("output_folder") or weights.parent
    if not output_folder.is_dir():
        raise ValueError("输出文件夹不存在。")
    output_name = str(context.parameters.get("output_filename", "")).strip() or f"{weights.stem}_cut.onnx"
    if Path(output_name).name != output_name or not output_name.lower().endswith(".onnx"):
        raise ValueError("输出文件名只能填写一个以 .onnx 结尾的文件名。")
    output = (output_folder / output_name).resolve()
    if output.exists():
        raise ValueError(f"输出 ONNX 已存在：{output}。请修改输出文件名，避免覆盖。")
    task_name = str(context.parameters.get("task_type", "目标检测"))
    if task_name not in TASKS:
        raise ValueError("请选择有效的模型任务类型。")
    platform_name = str(context.parameters.get("target_platform", "Jetson / TensorRT（NCHW）"))
    if platform_name not in PLATFORMS:
        raise ValueError("请选择有效的目标平台。")
    image_size = _integer(context.parameters.get("imgsz"), "输入尺寸 imgsz", 640, 32, 4096)
    if image_size % 32:
        raise ValueError("输入尺寸 imgsz 必须是 32 的整数倍。")
    model_family = str(context.parameters.get("model_family") or "YOLO26")
    if model_family not in MODEL_FAMILIES:
        raise ValueError("请选择有效的模型版本。")
    yolov5_repo: Path | None = None
    if model_family == "YOLOv5":
        if task_name != "目标检测":
            raise ValueError("当前 YOLOv5 导出仅支持目标检测模型。")
        raw_repo = str(context.parameters.get("yolov5_repo") or "").strip()
        if not raw_repo:
            raise ValueError("YOLOv5 模型需要填写源码仓库目录；该目录应包含 export.py 和 models 文件夹。")
        yolov5_repo = Path(raw_repo).expanduser().resolve()
        if not (yolov5_repo / "export.py").is_file() or not (yolov5_repo / "models").is_dir():
            raise ValueError(f"YOLOv5 仓库不完整：{yolov5_repo}。需要包含 export.py 和 models 文件夹。")
    return {
        "weights": weights.expanduser().resolve(),
        "output": output,
        "python": _resolve_python(context.parameters.get("python_path")),
        "task": TASKS[task_name],
        "task_name": task_name,
        "platform": PLATFORMS[platform_name],
        "platform_name": platform_name,
        "imgsz": image_size,
        "opset": _integer(context.parameters.get("opset"), "ONNX opset", 12, 11, 20),
        "model_family": model_family,
        "yolov5_repo": yolov5_repo,
    }


def _run_process(command: list[str], context: RunContext) -> None:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    output_queue: queue.Queue[str | None] = queue.Queue()
    recent: list[str] = []

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            output_queue.put(line.rstrip())
        output_queue.put(None)

    threading.Thread(target=read_output, name="jetson-onnx-export-log", daemon=True).start()
    closed = False
    try:
        while not closed or process.poll() is None:
            context.check_cancelled()
            try:
                line = output_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                closed = True
            elif line:
                recent.append(line)
                del recent[:-20]
                context.report(line)
        return_code = process.wait()
        if return_code != 0:
            detail = recent[-1] if recent else f"退出码 {return_code}"
            raise RuntimeError(f"Jetson ONNX 导出失败：{detail}")
    except TaskCancelled:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        raise


def run_jetson_onnx_export(context: RunContext) -> dict[str, Any]:
    config = _configuration(context)
    context.report(f"使用 YOLO 环境：{config['python']}")
    context.report(
        f"输入权重：{config['weights']}\n输出 ONNX：{config['output']}\n"
        f"模型版本：{config['model_family']}；任务：{config['task_name']}；平台：{config['platform_name']}；"
        f"imgsz={config['imgsz']}；opset={config['opset']}"
    )
    context.check_cancelled()
    with tempfile.TemporaryDirectory(prefix=".processing-view-export-", dir=config["output"].parent) as temporary:
        work = Path(temporary)
        temporary_weights = work / "weights.pt"
        temporary_output = work / "export.onnx"
        shutil.copy2(config["weights"], temporary_weights)
        if config["model_family"] == "YOLOv5":
            context.report(f"使用 YOLOv5 仓库：{config['yolov5_repo']}")
            command = [
                str(config["python"]), "-u", "-B", str(config["yolov5_repo"] / "export.py"),
                "--weights", str(temporary_weights), "--include", "onnx",
                "--imgsz", str(config["imgsz"]), "--opset", str(config["opset"]),
                "--batch-size", "1", "--device", "cpu", "--simplify",
            ]
        else:
            command = [
                str(config["python"]), "-u", "-B", str(WORKER_SCRIPT),
                "--weights", str(temporary_weights),
                "--output", str(temporary_output),
                "--imgsz", str(config["imgsz"]),
                "--opset", str(config["opset"]),
                "--task", config["task"],
                "--platform", config["platform"],
            ]
        _run_process(command, context)
        if config["model_family"] == "YOLOv5":
            exported = temporary_weights.with_suffix(".onnx")
            if exported.is_file():
                exported.replace(temporary_output)
        if not temporary_output.is_file() or temporary_output.stat().st_size == 0:
            raise RuntimeError("导出进程已结束，但没有生成有效的 ONNX 文件。")
        temporary_output.replace(config["output"])
    size_mb = config["output"].stat().st_size / 1024 / 1024
    context.report(f"Jetson 适配 ONNX 导出完成：{config['output']}（{size_mb:.1f} MB）")
    return {
        "message": f"Jetson 适配 ONNX 导出完成：{config['output'].name}。",
        "outputFiles": [str(config["output"])],
        "outputFolders": [str(config["output"].parent)],
        "inputWeights": str(config["weights"]),
        "outputOnnx": str(config["output"]),
        "taskType": config["task_name"],
        "targetPlatform": config["platform_name"],
        "imgsz": config["imgsz"],
        "opset": config["opset"],
        "modelFamily": config["model_family"],
        "onnxQuantDefaults": {"inputOnnx": str(config["output"])},
    }


register_handler(HANDLER_ID, run_jetson_onnx_export)
