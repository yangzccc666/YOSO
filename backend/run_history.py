"""Persist the five most recently finished platform runs."""

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

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self.path.exists():
                return []
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                raise ValueError("运行历史文件格式不正确。")
            return data[: self.LIMIT]

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
            records = [entry, *(old for old in self.list() if old.get("id") != entry["id"])][: self.LIMIT]
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
        return entry
