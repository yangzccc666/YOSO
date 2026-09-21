"""Minimal extension interface for future data-processing handlers."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .task_manager import TaskCancelled


@dataclass
class RunContext:
    function_id: str
    paths: dict[str, Path]
    parameters: dict[str, Any]
    report: Callable[[str], None]
    stop_event: threading.Event | None = None
    notify: Callable[[str, str], None] | None = None

    def check_cancelled(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise TaskCancelled("运行已被用户终止。已生成的文件会保留。")

    def send_notification(self, title: str, message: str) -> None:
        """Send a best-effort desktop notification without affecting the task."""
        if self.notify is None:
            return
        try:
            self.notify(str(title), str(message))
        except Exception:
            pass


class Handler(Protocol):
    def __call__(self, context: RunContext) -> dict[str, Any]: ...


_handlers: dict[str, Handler] = {}


def register_handler(handler_id: str, handler: Handler) -> None:
    """Register one real processing function without changing the UI."""
    if not handler_id.strip():
        raise ValueError("处理器 ID 不能为空。")
    _handlers[handler_id] = handler


def has_handler(handler_id: str | None) -> bool:
    return bool(handler_id and handler_id in _handlers)


def execute(handler_id: str, context: RunContext) -> dict[str, Any]:
    try:
        handler = _handlers[handler_id]
    except KeyError as exc:
        raise ValueError("处理逻辑尚未接入。") from exc
    return handler(context)
