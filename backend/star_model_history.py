"""A persistent model library, independent of the five-entry run history."""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class StarModelHistoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise ValueError("历史模型文件格式不正确，请备份后检查，原文件未被覆盖。")
        return records

    def _write(self, records: list[dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [{**row, "available": Path(row["path"]).is_file()} for row in self._read()]

    def remember(self, models: Any, *, used: bool = False, require_files: bool = False) -> None:
        if not isinstance(models, list) or not 1 <= len(models) <= 8:
            raise ValueError("请提供 1 到 8 个模型配置。")
        normalized = []
        for index, model in enumerate(models, 1):
            if not isinstance(model, dict):
                raise ValueError(f"第 {index} 个模型配置格式不正确。")
            path_value, labels = model.get("path"), model.get("labels")
            if not isinstance(path_value, str) or not path_value.strip() or not isinstance(labels, str) or not labels.strip():
                raise ValueError(f"请填写第 {index} 个模型的文件路径和类别名称。")
            path = Path(path_value.strip()).expanduser().resolve()
            if path.suffix.lower() not in {".star", ".plan", ".engine"}:
                raise ValueError(f"第 {index} 个模型只支持 .star / .plan / .engine 文件。")
            if require_files and not path.is_file():
                raise ValueError(f"第 {index} 个模型文件不存在：{path}")
            try:
                confidence = float(model.get("conf", 0.5))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"第 {index} 个模型的置信度必须在 0 到 1 之间。") from exc
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(f"第 {index} 个模型的置信度必须在 0 到 1 之间。")
            # Whitespace is cosmetic; class order (including empty positions) is not.
            normalized.append({"path": str(path), "labels": ",".join(part.strip() for part in labels.strip().split(",")), "conf": confidence})

        with self._lock:
            records = self._read()
            now = datetime.now().isoformat(timespec="microseconds")
            seen = set()
            refreshed = []
            for model in normalized:
                identity = (model["path"], model["labels"])
                if identity in seen:
                    continue
                seen.add(identity)
                existing = next((row for row in records if (row["path"], row["labels"]) == identity), {})
                refreshed.append({
                    "id": existing.get("id", uuid.uuid4().hex), **model,
                    "createdAt": existing.get("createdAt", now), "updatedAt": now,
                    "lastUsedAt": now if used else existing.get("lastUsedAt"),
                    "useCount": int(existing.get("useCount", 0)) + int(used),
                })
            self._write([*refreshed, *(row for row in records if (row["path"], row["labels"]) not in seen)])

    def remember_payload(self, payload: dict[str, Any]) -> None:
        """Called only after the inference manager has accepted the configuration."""
        parameters = payload.get("parameters", payload)
        models = parameters.get("star_models", [])
        if isinstance(models, str):
            models = json.loads(models or "[]")
        if not models:
            models = [{"path": payload.get("paths", {}).get("local_model_file", ""),
                       "labels": parameters.get("labels", ""), "conf": parameters.get("conf", 0.5)}]
        self.remember(models, used=True)

    def delete(self, record_id: str) -> None:
        with self._lock:
            records = self._read()
            remaining = [row for row in records if row["id"] != record_id]
            if len(remaining) == len(records):
                raise ValueError("历史模型不存在或已被删除。")
            self._write(remaining)
