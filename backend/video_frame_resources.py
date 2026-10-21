"""Shared, bounded decoder slots for concurrent frame-extraction batches."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from typing import Any

from .handlers import RunContext

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]


MIB = 1024 * 1024
MEMORY_RESERVE = 512 * MIB
DEFAULT_WORKER_MEMORY = 512 * MIB
_sample_lock = threading.Lock()
_last_sample: tuple[float, float, float] | None = None
_cpu_percent: float | None = None


def machine_resources() -> dict[str, Any]:
    global _last_sample, _cpu_percent
    cpu_count = os.cpu_count() or 1
    if hasattr(os, "sched_getaffinity"):
        try:
            cpu_count = min(cpu_count, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    available = None
    warning = ""
    if psutil is not None:
        try:
            available = int(psutil.virtual_memory().available)
            times = psutil.cpu_times()
            total = sum(times) - getattr(times, "guest", 0) - getattr(times, "guest_nice", 0)
            idle = times.idle + getattr(times, "iowait", 0)
            with _sample_lock:
                now = time.monotonic()
                if _last_sample is None or now - _last_sample[0] >= 0.5:
                    if _last_sample is not None and total > _last_sample[1]:
                        _cpu_percent = round(max(0, min(100, 100 * (1 - (idle - _last_sample[2]) / (total - _last_sample[1])))), 1)
                    _last_sample = (now, total, idle)
        except Exception:
            warning = "暂时无法读取完整资源信息，建议使用较低并行数。"
    else:
        warning = "资源监测组件不可用，建议串行处理；安装平台依赖后可显示内存信息。"
    cpu_limit = min(16, max(1, cpu_count // 2))
    memory_limit = max(0, (available - MEMORY_RESERVE) // DEFAULT_WORKER_MEMORY) if available is not None else 1
    maximum = min(cpu_limit, memory_limit)
    if available is not None and memory_limit == 0:
        warning = "可用内存不足 1 GB，请关闭部分程序、释放内存后再启动抽帧。"
    elif _cpu_percent is not None and _cpu_percent >= 90:
        warning = f"当前 CPU 占用 {_cpu_percent:g}%，建议降低并行数，或等待其他任务完成。"
    return {
        "cpuCount": cpu_count, "cpuPercent": _cpu_percent,
        "availableMemoryMb": round(available / MIB) if available is not None else None,
        "maxWorkers": maximum, "recommendedWorkers": min(1 if (_cpu_percent or 0) >= 90 else 4, maximum),
        "warning": warning,
    }


class VideoFrameResourcePool:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active = 0
        self._reserved = 0

    def status(self) -> dict[str, Any]:
        snapshot = machine_resources()
        with self._condition:
            snapshot["activeWorkers"] = self._active
        return snapshot

    @contextmanager
    def acquire(self, context: RunContext, memory_bytes: int = DEFAULT_WORKER_MEMORY):
        memory_bytes = max(DEFAULT_WORKER_MEMORY, memory_bytes)
        reported = False
        with self._condition:
            while True:
                context.check_cancelled()
                snapshot = machine_resources()
                cpu_limit = min(16, max(1, snapshot["cpuCount"] // 2)) if snapshot["availableMemoryMb"] is not None else 1
                available = snapshot["availableMemoryMb"]
                enough_memory = available is None or available * MIB >= MEMORY_RESERVE + memory_bytes + self._reserved
                if self._active < cpu_limit and enough_memory:
                    self._active += 1
                    self._reserved += memory_bytes
                    break
                if self._active == 0 and not enough_memory:
                    raise RuntimeError("当前可用内存不足以安全解码该视频，请释放内存后重试。")
                if not reported:
                    context.report("其他抽帧任务正在使用资源，或可用内存不足，当前视频等待空闲资源；可随时终止。")
                    reported = True
                self._condition.wait(timeout=0.25)
        try:
            context.check_cancelled()
            yield
        finally:
            with self._condition:
                self._active -= 1
                self._reserved -= memory_bytes
                self._condition.notify_all()


resource_pool = VideoFrameResourcePool()
