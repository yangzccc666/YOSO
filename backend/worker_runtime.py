"""Dispatch packaged AI workers through the app, never app.exe script.py."""
from __future__ import annotations

import runpy
import os
import sys
from pathlib import Path

WORKERS = {"bundled_export_onnx_cut", "auto_annotation_worker", "local_pt_worker", "model_labels_worker"}


def restore_worker_stdio() -> None:
    """Windowed builds may set sys.std* to None despite redirected child pipes."""
    for name, descriptor, mode in (("stdin", 0, "r"), ("stdout", 1, "w"), ("stderr", 2, "w")):
        if getattr(sys, name) is not None:
            continue
        try:
            stream = open(descriptor, mode, encoding="utf-8", closefd=False)
        except OSError:
            if sys.platform != "win32":
                raise RuntimeError(f"AI worker 缺少 {name} 通信管道。") from None
            import ctypes
            import msvcrt
            get_handle = ctypes.windll.kernel32.GetStdHandle
            get_handle.argtypes = [ctypes.c_ulong]
            get_handle.restype = ctypes.c_void_p
            handle = get_handle((-10 - descriptor) & 0xFFFFFFFF)
            if handle in (None, 0, ctypes.c_void_p(-1).value):
                raise RuntimeError(f"AI worker 缺少 Windows {name} 通信句柄。")
            fd = msvcrt.open_osfhandle(handle, os.O_BINARY | (os.O_RDONLY if mode == "r" else os.O_WRONLY))
            stream = open(fd, mode, encoding="utf-8")
        setattr(sys, name, stream)


def worker_command(python: Path, script: Path, arguments: list[str]) -> list[str]:
    if getattr(sys, "frozen", False) and python.resolve() == Path(sys.executable).resolve():
        if script.stem not in WORKERS:
            raise ValueError("此 AI 工作脚本未集成到平台；请选择独立 Python 环境。")
        return [sys.executable, "--internal-worker", script.stem, *arguments]
    return [str(python), "-u", "-B", str(script), *arguments]


def run_worker(arguments: list[str]) -> None:
    if not arguments or arguments[0] not in WORKERS:
        raise ValueError("未知的内置 AI 工作脚本。")
    restore_worker_stdio()
    sys.argv = [arguments[0], *arguments[1:]]
    runpy.run_module(f"backend.{arguments[0]}", run_name="__main__")
