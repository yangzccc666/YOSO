"""Dependency inventory for the distributed application, without probing SSH/Docker."""
from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

AI_MODULES = ("torch", "torchvision", "ultralytics", "onnx", "onnxruntime", "onnxslim")


def module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def dependency_report() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    base = Path(getattr(sys, "_MEIPASS", root.parent))
    manifest_file = base / "distribution-manifest.json"
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        manifest = {"profile": "source" if not getattr(sys, "frozen", False) else "standard"}
    checks = []
    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})
    scripts = ("bundled_video_frames.py", "bundled_run_star_video.py", "remote_yolo_dataset_worker.py",
               "remote_calib_worker.py", "remote_star_worker.py", "bundled_export_onnx_cut.py", "auto_annotation_worker.py")
    missing_scripts = [name for name in scripts if not (root / name).is_file()]
    add("平台处理脚本", "missing" if missing_scripts else "ready",
        "缺少：" + "、".join(missing_scripts) if missing_scripts else "处理脚本随平台分发，不读取开发者的 tools 目录。")
    add("STAR 配置模板与类别读取", "ready", "默认 TOML 模板和 STAR 类别解析代码已内置；自定义模板可选。")
    try:
        from .video_clip import _find_ffmpeg
        ffmpeg = _find_ffmpeg()
        add("视频裁剪 FFmpeg", "ready", f"已找到：{ffmpeg}；视频处理结果仍需实际验证。")
    except (RuntimeError, OSError) as exc:
        add("视频裁剪 FFmpeg", "missing", str(exc))
    missing_ai = [name for name in AI_MODULES if not module_present(name)]
    add("本地 PT / ONNX 推理／反向标注／YOLO26 导出", "missing" if missing_ai else "detected",
        "当前进程缺少：" + "、".join(missing_ai) + "。可配置独立 AI Python，或使用经过验证的 AI 集成构建。" if missing_ai else
        "已发现 AI 库；这不代表模型兼容性、GPU 或导出结果已经验证。")
    add("STAR 第三方打包工具", "external", "仍需匹配操作系统的 package_tool；当前分发源码不含该第三方二进制，不能宣称此功能零配置。")
    add("经典 YOLOv5 导出", "external", "依赖与权重匹配的 YOLOv5 源码及环境；不是平台开发者的固定路径。")
    add("Docker ONNX 量化", "external", "Docker 命令已找到；仍需 EasyONNXQuant 容器／授权安装包。" if shutil.which("docker") else
        "尚未找到 Docker；还需要独立的 EasyONNXQuant 容器／授权安装包。")
    add("SSH 训练／推理／TensorRT 构建", "remote", "客户端与适配脚本已内置；目标设备仍需 GPU 驱动、对应推理/训练环境。此检查不连接设备。")
    add("X-AnyLabeling／Antigravity", "optional", "仅作为可选外部工具启动入口，不影响平台和数据处理功能；首次使用需配置合法安装程序。")
    versions = {}
    for name in (*AI_MODULES, "PySide6", "imageio-ffmpeg", "paramiko"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"profile": manifest.get("profile", "unknown"), "system": f"{platform.system()} {platform.machine()}",
            "checks": checks, "versions": versions,
            "note": "发现依赖不等于功能验收。本检查不读取数据集、模型或密码，不自动安装软件、不连接 SSH/Docker。"}
