"""Persist the five most recently finished runs for every platform function."""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class RunHistoryStore:
    LIMIT = 5

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("运行历史文件格式不正确。")
        return [record for record in data if isinstance(record, dict)]

    def _limit_by_function(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        limited: list[dict[str, Any]] = []
        for record in records:
            function_id = str(record.get("functionId", ""))
            if counts.get(function_id, 0) >= self.LIMIT:
                continue
            counts[function_id] = counts.get(function_id, 0) + 1
            limited.append(record)
        return limited

    def list(self, function_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = self._limit_by_function(self._read())
            if function_id is None:
                return records
            target = str(function_id)
            return [record for record in records if str(record.get("functionId", "")) == target]

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

    def delete(self, record_id: str) -> None:
        target = str(record_id).strip()
        if not target:
            raise ValueError("运行历史记录 ID 不能为空。")
        with self._lock:
            records = self._read()
            remaining = [record for record in records if str(record.get("id", "")) != target]
            if len(remaining) == len(records):
                raise ValueError("运行历史记录不存在或已被删除。")
            self._write(remaining)

    def add(
        self,
        task: dict[str, Any],
        *,
        status: str,
        message: str,
        logs: list[str] | None = None,
        details: dict[str, str] | None = None,
        outputs: list[str] | None = None,
    ) -> dict[str, Any]:
        if status not in {"completed", "failed", "stopped"}:
            raise ValueError("只能记录已结束的运行。")
        entry = {
            "id": str(task["id"]),
            "functionId": str(task["functionId"]),
            "name": str(task["name"]),
            "kind": str(task["kind"]),
            "status": status,
            "startedAt": str(task["startedAt"]),
            "finishedAt": datetime.now().isoformat(timespec="seconds"),
            "message": str(message)[:2000],
            "details": {str(key)[:80]: str(value)[:1000] for key, value in (details or {}).items() if str(value).strip()},
            "outputs": [str(value)[:1000] for value in (outputs or []) if str(value).strip()][:20],
            "logs": [str(line)[:2000] for line in (logs or [])][-500:],
        }
        with self._lock:
            records = self._limit_by_function([
                entry,
                *(old for old in self._read() if old.get("id") != entry["id"]),
            ])
            self._write(records)
        return entry
