"""Isolated Ultralytics process for local PT inference.

The desktop's Python need not contain PyTorch. stdin/stdout carry length-prefixed
JSON headers and raw BGR frames; library chatter is redirected to stderr.
"""

from __future__ import annotations

import json
import struct
import sys
from typing import Any, BinaryIO


MAX_HEADER = 65536
MAX_FRAME = 256 * 1024 * 1024


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
    if len(sys.argv) != 2:
        print("需要指定 PT 模型路径。", file=sys.stderr)
        return 2
    protocol_output = sys.stdout.buffer
    sys.stdout = sys.stderr
    try:
        import numpy as np
        from ultralytics import YOLO

        model = YOLO(sys.argv[1])
        send(protocol_output, {"type": "ready", "names": model.names})
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
            annotated = model.predict(frame, **header.get("options", {}))[0].plot()
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
