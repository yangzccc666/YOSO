"""Independent task coordination and cooperative cancellation."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable


class TaskCancelled(RuntimeError):
    """Raised by a handler after the user requests cancellation."""


@dataclass
class ActiveTask:
    id: str
    function_id: str
    name: str
    kind: str
    task_key: str
    status: str = "running"
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    stop_event: threading.Event = field(default_factory=threading.Event)
    stop_callback: Callable[[], Any] | None = None
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=500))

    def add_log(self, message: str) -> None:
        self.logs.append(str(message))

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "functionId": self.function_id,
            "name": self.name,
            "kind": self.kind,
            "status": self.status,
            "startedAt": self.started_at,
            "logs": list(self.logs),
        }


class PlatformTaskManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._active: dict[str, ActiveTask] = {}

    def start(self, function_id: str, name: str, kind: str = "local", task_key: str | None = None) -> ActiveTask:
        with self._lock:
            key = task_key or function_id
            existing = next((task for task in self._active.values() if task.task_key == key), None)
            if existing is not None:
                raise ValueError(
                    f"“{existing.name}”正在运行。请等待完成，或先点击“终止运行”。"
                )
            task = ActiveTask(
                id=uuid.uuid4().hex[:16],
                function_id=function_id,
                name=name,
                kind=kind,
                task_key=key,
            )
            self._active[task.id] = task
            return task

    def active(self) -> list[dict[str, Any]]:
        with self._lock:
            return [task.snapshot() for task in self._active.values()]

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._active.get(task_id)
            return task.snapshot() if task else None

    def set_stop_callback(self, task_id: str, callback: Callable[[], Any]) -> None:
        with self._lock:
            task = self._active.get(task_id)
            if task is not None:
                task.stop_callback = callback

    def stop(self, task_id: str) -> dict[str, Any]:
        with self._lock:
            task = self._active.get(task_id)
            if task is None:
                raise ValueError("指定任务已结束或不存在。")
            task.status = "stopping"
            task.stop_event.set()
            callback = task.stop_callback
            snapshot = task.snapshot()
        if callback is not None:
            try:
                callback()
            except Exception:
                # Cooperative cancellation remains active even if an optional
                # external-process/remote callback has already disconnected.
                pass
        return snapshot

    def finish(self, task_id: str) -> None:
        with self._lock:
            self._active.pop(task_id, None)

    def stop_all(self) -> None:
        with self._lock:
            task_ids = [
                task.id for task in self._active.values()
                if task.kind not in {"remote-training", "remote-build"}
            ]
        for task_id in task_ids:
            try:
                self.stop(task_id)
            except ValueError:
                pass


manager = PlatformTaskManager()
