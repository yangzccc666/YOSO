"""Isolated Ultralytics process for local PT / standard ONNX detection.

The desktop's Python need not contain PyTorch. stdin/stdout carry length-prefixed
JSON headers and raw BGR frames; library chatter is redirected to stderr.
"""

from __future__ import annotations

import json
import ast
import os
import re
import struct
import sys
from pathlib import Path
from typing import Any, BinaryIO


MAX_HEADER = 65536
MAX_FRAME = 256 * 1024 * 1024


def normalize_names(value: Any) -> list[str]:
    """Keep class IDs, not dictionary insertion order; never evaluate model text."""
    if isinstance(value, str):
        if len(value) > MAX_HEADER:
            raise ValueError("模型类别元数据过长。")
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError) as exc:
            raise ValueError("模型 names 元数据不是有效的类别列表或字典。") from exc
    if isinstance(value, dict):
        if any(isinstance(key, bool) or not re.fullmatch(r"\d+", str(key)) for key in value):
            raise ValueError("模型类别 ID 必须是从 0 开始的连续整数。")
        try:
            indexed = {int(key): name for key, name in value.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError("模型类别 ID 必须是从 0 开始的连续整数。") from exc
        if len(indexed) != len(value) or sorted(indexed) != list(range(len(indexed))):
            raise ValueError("模型类别 ID 必须是从 0 开始的连续整数。")
        value = [indexed[index] for index in range(len(indexed))]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("模型没有有效的类别名称。")
    if any(not isinstance(name, str) or not name.strip() or "\x00" in name for name in value):
        raise ValueError("模型类别名称不能为空或包含空字符。")
    return [name.strip() for name in value]


def parse_label_text(text: str) -> list[str]:
    if not text.strip():
        return []
    labels = []
    for line in re.split(r"[,，\r\n]+", text):
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\d+)\s*[:：]\s*(.*)$", line)
        if match:
            if int(match[1]) != len(labels):
                raise ValueError("类别序号必须从 0 开始按顺序填写，避免框标签错位。")
            line = match[2].strip()
        labels.append(line)
    return normalize_names(labels)


def inspect_model_file(model_path: str) -> tuple[dict[str, Any], Any]:
    path = Path(model_path)
    model = None
    image_size = None
    if path.suffix.lower() == ".onnx":
        try:
            import onnx
        except ImportError as exc:
            raise RuntimeError("AI Python 缺少 onnx，请在平台设置的 AI 环境中安装 onnx 和 onnxruntime。") from exc
        graph = onnx.load(str(path), load_external_data=False)
        metadata = {entry.key: entry.value for entry in graph.metadata_props}
        task = metadata.get("task", "detect")
        if len(graph.graph.output) != 1 or len(graph.graph.output[0].type.tensor_type.shape.dim) != 3:
            raise ValueError("此 ONNX 不是标准 YOLO 检测输出。Jetson 裁剪头／多输出 ONNX 不能直接用于本地检测，请从 PT 导出标准 format=onnx 模型。")
        initializers = {entry.name for entry in graph.graph.initializer}
        inputs = [entry for entry in graph.graph.input if entry.name not in initializers]
        if len(inputs) != 1:
            raise ValueError("本地 ONNX 检测仅支持一个 NCHW 图像输入。")
        dims = inputs[0].type.tensor_type.shape.dim
        if len(dims) != 4 or dims[0].dim_value not in (0, 1) or dims[1].dim_value not in (0, 3):
            raise ValueError("本地 ONNX 检测需要 batch=1 的 NCHW 三通道输入。")
        if dims[2].dim_value > 0 and dims[3].dim_value > 0:
            image_size = [dims[2].dim_value, dims[3].dim_value]
        names = normalize_names(metadata["names"]) if metadata.get("names") else []
        source = "ONNX 内嵌 names 元数据"
    elif path.suffix.lower() == ".pt":
        from ultralytics import YOLO
        model = YOLO(str(path), task="detect")
        task = model.task
        names = normalize_names(model.names)
        source = "PT 权重 names"
    else:
        raise ValueError("请选择 .pt 或 .onnx 模型文件。")
    if task != "detect":
        raise ValueError(f"当前功能用于目标检测，该模型任务是 {task}，不能作为检测模型使用。")
    return {"labels": names, "count": len(names), "found": bool(names), "modelType": path.suffix[1:].upper(),
            "source": source, "imageSize": image_size,
            "message": f"已从{source}读取 {len(names)} 个类别。" if names else "ONNX 未保存类别名称，请按类别 ID 顺序手动填写；不能从模型数值推断语义名称。"}, model


def read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        part = stream.read(size - len(chunks))
        if not part:
            raise EOFError("视频帧通信已中断。")
        chunks.extend(part)
    return bytes(chunks)


def receive(stream: BinaryIO) -> tuple[dict[str, Any], bytes]:
    header_size, frame_size = struct.unpack("!II", read_exact(stream, 8))
    if not 0 < header_size <= MAX_HEADER or frame_size > MAX_FRAME:
        raise ValueError("视频帧通信数据大小异常。")
    header = json.loads(read_exact(stream, header_size))
    if not isinstance(header, dict):
        raise ValueError("视频帧通信格式异常。")
    return header, read_exact(stream, frame_size)


def send(stream: BinaryIO, header: dict[str, Any], frame: bytes = b"") -> None:
    encoded = json.dumps(header, ensure_ascii=False).encode("utf-8")
    stream.write(struct.pack("!II", len(encoded), len(frame)))
    stream.write(encoded)
    stream.write(frame)
    stream.flush()


def main() -> int:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--inspect"):
        print("需要指定 PT / ONNX 模型路径。", file=sys.stderr)
        return 2
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        # Do not silently pip-install into a user's environment during playback.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        metadata, model = inspect_model_file(sys.argv[1])
        if len(sys.argv) == 3:
            send(protocol_output, {"type": "metadata", **metadata})
            return 0
        import numpy as np
        from ultralytics import YOLO

        if model is None:
            try:
                import onnxruntime
            except ImportError as exc:
                raise RuntimeError("AI Python 缺少 onnxruntime，无法运行 ONNX 模型。") from exc
            model = YOLO(sys.argv[1], task="detect")
        send(protocol_output, {"type": "ready", "names": dict(enumerate(metadata["labels"])), "metadata": metadata})
        while True:
            header, raw_frame = receive(sys.stdin.buffer)
            if header.get("type") == "close":
                return 0
            if header.get("type") != "frame":
                raise ValueError("未知的视频帧通信命令。")
            shape = header.get("shape")
            if (not isinstance(shape, list) or len(shape) != 3 or
                    not all(isinstance(value, int) and value > 0 for value in shape) or
                    shape[2] != 3 or shape[0] * shape[1] * shape[2] != len(raw_frame)):
                raise ValueError("输入视频帧尺寸不正确。")
            frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(shape).copy()
            options = dict(header.get("options", {}))
            labels = options.pop("_class_names", None) or metadata["labels"]
            if not labels:
                raise ValueError("ONNX 中没有类别名称，请先在类别名称框中手动填写后再运行。")
            if metadata["labels"] and len(labels) != len(metadata["labels"]):
                raise ValueError(f"填写了 {len(labels)} 个类别，但模型有 {len(metadata['labels'])} 个类别，请检查顺序与数量。")
            if metadata["modelType"] == "ONNX":
                device = str(options.get("device", "cpu"))
                if device not in {"auto", "cpu"}:
                    if "," in device:
                        raise ValueError("本地 ONNX 推理请指定一张 GPU（例如 0），不支持多 GPU 编号。")
                    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
                        raise ValueError("当前 ONNX Runtime 不支持 CUDA，请选择 cpu，或安装匹配 CUDA 的 onnxruntime-gpu。")
                else:
                    options["device"] = "cpu"
                if metadata["imageSize"]:
                    options["imgsz"] = metadata["imageSize"]
            result = model.predict(frame, **options)[0]
            if result.boxes is not None and len(result.boxes) and int(result.boxes.cls.max()) >= len(labels):
                raise ValueError("模型输出的类别 ID 超过已填写的类别数量，请补全全部类别。")
            result.names = dict(enumerate(labels))
            annotated = result.plot()
            if annotated.ndim != 3 or annotated.shape[2] != 3 or annotated.dtype != np.uint8:
                raise ValueError("模型返回的带框视频帧格式不正确。")
            send(protocol_output, {"type": "frame", "shape": list(annotated.shape)}, annotated.tobytes())
    except EOFError:
        return 0
    except Exception as exc:
        send(protocol_output, {"type": "error", "message": str(exc) or exc.__class__.__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
