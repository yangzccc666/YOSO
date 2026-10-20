"""Export local YOLO26 PT weights to raw-head ONNX for Jetson/TensorRT."""

from __future__ import annotations

import json
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


def _weight_paths(context: RunContext) -> list[Path]:
    raw = str(context.paths.get("weights_file") or "").strip()
    if not raw:
        raise ValueError("请选择至少一个存在的本地 PT 权重文件。")
    try:
        candidates = json.loads(raw) if raw.startswith("[") else raw.splitlines()
    except json.JSONDecodeError as exc:
        raise ValueError("PT 权重路径格式不正确，请每行填写一个文件路径。") from exc
    if not isinstance(candidates, list):
        raise ValueError("PT 权重路径格式不正确，请每行填写一个文件路径。")
    weights: list[Path] = []
    seen: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        text = str(candidate).strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if path.suffix.lower() != ".pt" or not path.is_file():
            raise ValueError(f"第 {index} 个 PT 权重文件不存在或格式不正确：{path}")
        resolved = path.resolve()
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            weights.append(resolved)
    if not weights:
        raise ValueError("请选择至少一个存在的本地 PT 权重文件。")
    return weights


def _configuration(
    context: RunContext, weights: Path | None = None, total_weights: int | None = None
) -> dict[str, Any]:
    all_weights = _weight_paths(context) if weights is None else [weights]
    weights = all_weights[0]
    total_weights = total_weights or len(all_weights)
    output_folder = context.paths.get("output_folder") or weights.parent
    if not output_folder.is_dir():
        raise ValueError("输出文件夹不存在。")
    requested_output_name = str(context.parameters.get("output_filename", "")).strip()
    if total_weights > 1 and requested_output_name:
        raise ValueError("批量导出时请将“单模型输出文件名”留空，平台会分别按 PT 文件名生成 ONNX。")
    output_name = requested_output_name or f"{weights.stem}_cut.onnx"
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


def _export_one(config: dict[str, Any], context: RunContext) -> None:
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


def run_jetson_onnx_export(context: RunContext) -> dict[str, Any]:
    weights = _weight_paths(context)
    configs = [_configuration(context, weight, len(weights)) for weight in weights]
    output_paths = [str(config["output"]) for config in configs]
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("多个 PT 会生成同名 ONNX，请选择不同的输出文件夹，或调整 PT 文件名后重试。")
    first = configs[0]
    context.report(f"使用 YOLO 环境：{first['python']}")
    context.report(
        f"共 {len(configs)} 个 PT 等待导出；模型版本：{first['model_family']}；"
        f"任务：{first['task_name']}；平台：{first['platform_name']}；"
        f"imgsz={first['imgsz']}；opset={first['opset']}"
    )
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, config in enumerate(configs, start=1):
        context.check_cancelled()
        context.report(
            f"[{index}/{len(configs)}] 开始导出：{config['weights'].name}\n"
            f"输入权重：{config['weights']}\n输出 ONNX：{config['output']}"
        )
        try:
            _export_one(config, context)
        except TaskCancelled:
            raise
        except Exception as exc:
            message = str(exc)
            failures.append({"weights": str(config["weights"]), "error": message})
            context.report(f"[{index}/{len(configs)}] 导出失败：{config['weights'].name}：{message}")
            continue
        completed.append(config)
        size_mb = config["output"].stat().st_size / 1024 / 1024
        context.report(f"[{index}/{len(configs)}] 导出完成：{config['output']}（{size_mb:.1f} MB）")
        context.send_notification(
            "Jetson ONNX 导出完成",
            f"{index}/{len(configs)}：{config['weights'].name} → {config['output'].name}",
        )
    if not completed:
        summary = "；".join(f"{Path(item['weights']).name}：{item['error']}" for item in failures)
        raise RuntimeError(f"全部 {len(configs)} 个 PT 均导出失败：{summary}")
    outputs = [str(config["output"]) for config in completed]
    folders = list(dict.fromkeys(str(config["output"].parent) for config in completed))
    message = f"Jetson 适配 ONNX 批量导出完成：成功 {len(completed)} 个"
    if failures:
        message += f"，失败 {len(failures)} 个；失败项已在日志中列出"
    message += "。"
    return {
        "message": message,
        "outputFiles": outputs,
        "outputFolders": folders,
        "inputWeights": str(completed[0]["weights"]),
        "inputWeightsList": [str(config["weights"]) for config in completed],
        "outputOnnx": outputs[0],
        "outputOnnxList": outputs,
        "failedFiles": failures,
        "taskType": first["task_name"],
        "targetPlatform": first["platform_name"],
        "imgsz": first["imgsz"],
        "opset": first["opset"],
        "modelFamily": first["model_family"],
        "onnxQuantDefaults": {"inputOnnx": outputs[0]},
        "onnxQuantDefaultsList": [{"inputOnnx": output} for output in outputs],
    }


register_handler(HANDLER_ID, run_jetson_onnx_export)
