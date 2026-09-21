"""Adapter for tools/data_processing_toolkit/02-video_processing/batch_vedio2img.py."""

from __future__ import annotations

import contextlib
import importlib.util
import io
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from .handlers import RunContext, register_handler
from .video_clip import parse_time


HANDLER_ID = "video.extract_frames"
SOURCE_SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "data_processing_toolkit" / "02-video_processing" / "batch_vedio2img.py"


@lru_cache(maxsize=1)
def _load_source_module() -> ModuleType:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(f"找不到视频切图脚本：{SOURCE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("processing_view_batch_video2img", SOURCE_SCRIPT)
    if not spec or not spec.loader:
        raise RuntimeError("无法加载视频切图脚本。")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _positive_number(value: Any, label: str, default: float) -> float:
    try:
        number = float(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc
    if number <= 0:
        raise ValueError(f"{label}必须大于 0。")
    return number


def _positive_integer(value: Any, label: str, default: int) -> int:
    number = _positive_number(value, label, default)
    if not number.is_integer():
        raise ValueError(f"{label}必须是整数。")
    return int(number)


def _optional_positive_integer(value: Any, label: str) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return _positive_integer(value, label, 1)


def _non_negative_number(value: Any, label: str, default: float = 0) -> float:
    try:
        number = float(value if value not in (None, "") else default)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}必须是数字。") from exc
    if number < 0:
        raise ValueError(f"{label}不能小于 0。")
    return number


def _unique_output_folder(parent: Path, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"{stem}_{timestamp}"
    index = 2
    while candidate.exists():
        candidate = parent / f"{stem}_{timestamp}_{index}"
        index += 1
    return candidate


def _resolve_videos(context: RunContext, module: ModuleType) -> list[Path]:
    inputs: list[str] = []
    folder = context.paths.get("video_folder")
    video_file = context.paths.get("video_file")
    if folder:
        inputs.append(str(folder))
    if video_file:
        inputs.append(str(video_file))
    if not inputs:
        raise ValueError("请选择视频文件夹，或者选择一个 MP4 视频。")
    videos = [Path(value).resolve() for value in module.parse_video_paths(inputs)]
    if not videos:
        raise ValueError("没有找到可处理的 MP4 视频，请检查输入路径。")
    return videos


def run_video_frames(context: RunContext) -> dict[str, Any]:
    module = _load_source_module()
    videos = _resolve_videos(context, module)
    output_root = context.paths.get("output_folder")
    if output_root:
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    mode = str(context.parameters.get("extraction_mode", "每秒抽取张数"))
    fallback_step = _positive_integer(context.parameters.get("fallback_step"), "备用帧间隔", 24)
    raw_max_images = (
        context.parameters.get("max_images_limit")
        if "max_images_limit" in context.parameters
        else context.parameters.get("max_images")
    )
    max_images = _optional_positive_integer(raw_max_images, "最大抽取图片数量")
    start_time_sec = parse_time(context.parameters.get("range_start_time", "00:00:00"), "开始时间")
    raw_end_time = context.parameters.get("range_end_time", "")
    end_time_sec = None if str(raw_end_time if raw_end_time is not None else "").strip() == "" else parse_time(raw_end_time, "结束时间")
    if end_time_sec is not None and end_time_sec <= start_time_sec:
        raise ValueError("结束时间必须晚于开始时间。")
    limit_strategy = str(context.parameters.get("limit_strategy", "全程均匀抽取"))
    if limit_strategy not in {"全程均匀抽取", "从前往后抽取"}:
        raise ValueError("超过数量上限时只能选择“全程均匀抽取”或“从前往后抽取”。")
    if mode == "按时间间隔":
        target_fps = None
        interval_sec = _positive_number(context.parameters.get("interval_sec"), "抽帧时间间隔", 1)
        mode_message = f"每 {interval_sec:g} 秒抽取 1 张"
    else:
        target_fps = _positive_number(context.parameters.get("target_fps"), "每秒抽取张数", 5)
        interval_sec = None
        mode_message = f"每秒抽取 {target_fps:g} 张"

    limit_description = f"每个视频最多 {max_images} 张" if max_images is not None else "不限制图片总数"
    context.report(f"找到 {len(videos)} 个视频，抽帧方式：{mode_message}，{limit_description}")
    context.report(
        f"有效区间：{start_time_sec:g} 秒 至 "
        f"{f'{end_time_sec:g} 秒' if end_time_sec is not None else '视频结尾'}；"
        f"{f'超过上限时采用“{limit_strategy}”' if max_images is not None else '按抽帧密度和有效时长自动计算数量'}"
    )
    total_images = 0
    output_folders: list[str] = []
    details: list[dict[str, Any]] = []

    for index, video in enumerate(videos, start=1):
        context.check_cancelled()
        parent = output_root or video.parent
        output_folder = _unique_output_folder(parent, video.stem)
        context.report(f"[{index}/{len(videos)}] 正在处理：{video.name}")

        captured = io.StringIO()
        original_tqdm = getattr(module, "tqdm", None)
        def cancellable_progress(iterable: Any) -> Any:
            for value in iterable:
                context.check_cancelled()
                yield value

        module.tqdm = cancellable_progress
        try:
            with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
                process_result = module.vedio2img(
                    vedio_path=str(video),
                    images_save_path=str(output_folder),
                    target_fps=target_fps,
                    interval_sec=interval_sec,
                    fallback_step=fallback_step,
                    max_images=max_images,
                    start_time_sec=start_time_sec,
                    end_time_sec=end_time_sec,
                    limit_strategy=limit_strategy,
                )
        finally:
            if original_tqdm is not None:
                module.tqdm = original_tqdm

        image_count = len(list(output_folder.glob("*.jpg"))) if output_folder.exists() else 0
        if image_count == 0:
            raise RuntimeError(f"{video.name} 没有成功生成图片，请确认视频可以正常播放。")
        total_images += image_count
        output_folders.append(str(output_folder))
        limit_reached = bool(process_result and process_result.get("limit_applied"))
        details.append({
            "video": str(video),
            "images": image_count,
            "maxImages": max_images,
            "limitReached": limit_reached,
            "limitStrategy": limit_strategy,
            "startTimeSec": process_result.get("start_time_sec", start_time_sec) if process_result else start_time_sec,
            "endTimeSec": process_result.get("end_time_sec", end_time_sec) if process_result else end_time_sec,
            "selectedFrameIndices": process_result.get("selected_frame_indices", []) if process_result else [],
            "output": str(output_folder),
        })
        if max_images is None:
            limit_message = "（按抽帧密度和有效时长完成）"
        elif limit_reached:
            limit_message = "（已达到上限）"
        else:
            limit_message = "（未达到设置上限）"
        context.report(f"[{index}/{len(videos)}] 已生成 {image_count} 张图片{limit_message}")

    return {
        "message": f"视频切图完成：处理 {len(videos)} 个视频，共生成 {total_images} 张图片。",
        "videoCount": len(videos),
        "imageCount": total_images,
        "maxImagesPerVideo": max_images,
        "outputFolders": output_folders,
        "details": details,
    }


register_handler(HANDLER_ID, run_video_frames)
