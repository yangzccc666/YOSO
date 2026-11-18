"""Read model.labels from the known StarLink FlatBuffers container, without CUDA.

The package_tool format uses root slot 2 for model metadata, model slot 4 for
the ordered string vector, and root slot 3 for model bytes. Both size-prefixed
and ordinary FlatBuffers are supported. Never search engine strings for labels.
"""

from __future__ import annotations

import mmap
import struct
from pathlib import Path
from typing import Any


class _StarReader:
    def __init__(self, data: mmap.mmap):
        self.data = data

    def check(self, offset: int, size: int) -> None:
        if offset < 0 or size < 0 or offset + size > len(self.data):
            raise ValueError("STAR 元数据偏移越界或文件不完整。")

    def number(self, offset: int, fmt: str = "<I") -> int:
        self.check(offset, struct.calcsize(fmt))
        return struct.unpack_from(fmt, self.data, offset)[0]

    def pointer(self, offset: int) -> int:
        distance = self.number(offset)
        if distance < 4:
            raise ValueError("STAR 元数据指针无效。")
        result = offset + distance
        self.check(result, 4)
        return result

    def field(self, table: int, slot: int) -> int:
        vtable = table - self.number(table, "<i")
        length = self.number(vtable, "<H")
        object_size = self.number(vtable + 2, "<H")
        if length < 4 or length % 2 or object_size < 4:
            raise ValueError("不支持的 STAR 元数据表结构。")
        self.check(vtable, length)
        self.check(table, object_size)
        entry = 4 + 2 * slot
        if entry + 2 > length:
            raise ValueError("STAR 中没有所需的类别元数据字段。")
        offset = self.number(vtable + entry, "<H")
        if offset < 4 or offset + 4 > object_size:
            raise ValueError("STAR 中缺少有效的类别元数据字段。")
        return table + offset

    def text(self, offset: int) -> str:
        size = self.number(offset)
        if not 0 < size <= 4096:
            raise ValueError("STAR 类别或元数据字符串长度无效。")
        self.check(offset + 4, size + 1)
        if self.data[offset + 4 + size] != 0:
            raise ValueError("STAR 元数据字符串未正确结束。")
        return self.data[offset + 4:offset + 4 + size].decode("utf-8")

    def labels(self) -> list[str]:
        # package_tool emits a uint32 size prefix; raw FlatBuffers omit it.
        start = 4 if self.number(0) == len(self.data) - 4 else 0
        root = self.pointer(start)
        metadata = self.pointer(self.field(root, 0))
        hardware = self.pointer(self.field(root, 1))
        self.text(self.pointer(self.field(metadata, 0)))
        self.text(self.pointer(self.field(hardware, 0)))
        model = self.pointer(self.field(root, 2))
        payload = self.pointer(self.field(root, 3))
        payload_size = self.number(payload)
        if not payload_size:
            raise ValueError("STAR 未包含模型数据。")
        self.check(payload + 4, payload_size)
        vector = self.pointer(self.field(model, 4))
        count = self.number(vector)
        if not 0 < count <= 10000:
            raise ValueError("STAR 类别列表为空或数量超出安全限制。")
        self.check(vector + 4, count * 4)
        result = []
        total = 0
        for index in range(count):
            value = self.text(self.pointer(vector + 4 + index * 4))
            total += len(value.encode("utf-8"))
            if total > 1024 * 1024:
                raise ValueError("STAR 类别元数据过大。")
            if not value.strip() or any(char in value for char in (",", "，", "\r", "\n", "\0")):
                raise ValueError("STAR 包含空类别或不适合逗号分隔输入框的类别名称，请手动核对。")
            # Preserve order and duplicate entries: their positions are class IDs.
            result.append(value)
        return result


def detect_star_labels(model_file: str | Path) -> dict[str, Any]:
    path = Path(model_file).expanduser()
    if path.suffix.lower() != ".star":
        raise ValueError("读取类别仅支持含类别元数据的 .star 文件；.plan/.engine 通常不保存类别名，请手动填写或使用历史模型。")
    if not path.is_file():
        raise ValueError("STAR 模型文件不存在或不是文件，请检查本机路径。")
    try:
        with path.open("rb") as stream:
            if stream.seek(0, 2) < 16:
                raise ValueError("STAR 文件过短。")
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                labels = _StarReader(data).labels()
    except (ValueError, UnicodeError, struct.error) as exc:
        raise ValueError(f"无法从此 STAR 读取类别：{exc} 文件可能损坏、未保存类别，或使用了尚不支持的打包格式；原类别不会修改。") from exc
    except OSError as exc:
        raise ValueError(f"无法读取 STAR 文件：{exc}") from exc
    return {
        "ok": True, "labels": labels, "count": len(labels),
        "source": f"{path.resolve()}（STAR 内嵌 model.labels）",
        "message": f"已从 STAR 内嵌元数据读取 {len(labels)} 个类别，按模型类别 ID 顺序填充。",
    }
