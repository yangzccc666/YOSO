"""Self-contained video frame extraction used by source and packaged builds."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import cv2
from tqdm import tqdm


def _filename_number(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _uniform_sample(items: list[int], count: int) -> list[int]:
    if count >= len(items):
        return list(items)
    if count <= 0:
        return []
    if count == 1:
        return [items[len(items) // 2]]
    return [items[round(index * (len(items) - 1) / (count - 1))] for index in range(count)]


def _frames_for_rate(start: int, end: int, source_fps: float, target_fps: float) -> list[int]:
    frames: list[int] = []
    position = float(start)
    increment = source_fps / target_fps
    while position < end:
        frame = min(end - 1, int(round(position)))
        if not frames or frame != frames[-1]:
            frames.append(frame)
        position += increment
    return frames


def parse_video_paths(inputs: str | list[str], check_cancelled: Callable[[], None] | None = None) -> list[str]:
    values = [inputs] if isinstance(inputs, str) else inputs
    if not isinstance(values, list):
        return []
    videos: list[str] = []
    for raw in values:
        if check_cancelled:
            check_cancelled()
        path = str(raw)
        if os.path.isdir(path):
            # Treat a selected folder as a dataset root and inspect every
            # descendant directory without following symlinks.
            found: list[str] = []
            for current, _directories, files in os.walk(path, followlinks=False):
                if check_cancelled:
                    check_cancelled()
                found.extend(
                    os.path.join(current, name)
                    for name in files
                    if name.lower().endswith(".mp4")
                )
            found.sort()
            videos.extend(found)
            print(f"[文件夹递归扫描] {path} -> 找到 {len(found)} 个 MP4 视频")
        elif os.path.isfile(path) and path.lower().endswith(".mp4"):
            videos.append(path)
            print(f"[文件] {path}")
        else:
            print(f"[跳过] 无效路径: {path}")
    result = sorted(set(videos))
    print(f"共 {len(result)} 个视频待处理")
    return result


def open_video(path: str):
    """Keep each FFmpeg decoder small instead of oversubscribing CPU threads."""
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        capture = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
        if capture.isOpened():
            return capture
        capture.release()
    return cv2.VideoCapture(path)


def vedio2img(
    vedio_path: str,
    images_save_path: str,
    target_fps: float | None = None,
    interval_sec: float | None = None,
    fallback_step: int = 24,
    max_images: int | None = None,
    start_skip_sec: float = 0,
    end_skip_sec: float = 0,
    limit_strategy: str = "全程均匀抽取",
    start_time_sec: float | None = None,
    end_time_sec: float | None = None,
    filename_prefix: str | None = None,
    priority_ranges: list[tuple[float, float]] | None = None,
    priority_fps: float = 25,
    background_fps: float = 5,
    check_cancelled: Callable[[], None] | None = None,
    report_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any] | None:
    Path(images_save_path).mkdir(parents=True, exist_ok=True)
    capture = open_video(vedio_path)
    if not capture.isOpened():
        capture.release()
        print(f"无法打开视频文件: {vedio_path}")
        return None
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    prefix = str(filename_prefix or Path(vedio_path).stem)
    mixed = bool(priority_ranges)
    try:
        if mixed:
            if source_fps <= 0:
                raise ValueError("无法读取源视频 FPS，不能使用工序时间段高密度抽帧")
            priority_fps = min(float(priority_fps), source_fps)
            background_fps = min(float(background_fps), source_fps)
            if priority_fps <= 0 or background_fps <= 0:
                raise ValueError("重点时间段和普通片段的抽帧帧率必须大于 0")
            step = max(1, round(source_fps / background_fps))
            filename_tag = f"priority{_filename_number(priority_fps)}_bg{_filename_number(background_fps)}"
        elif target_fps is not None:
            rate = float(target_fps)
            if rate <= 0:
                raise ValueError("target_fps 必须大于 0")
            rate = min(rate, source_fps) if source_fps > 0 else rate
            step = max(1, round(source_fps / rate)) if source_fps > 0 else max(1, int(fallback_step))
            filename_tag = f"fps{_filename_number(source_fps / step)}" if source_fps > 0 else f"step{step}"
        elif interval_sec is not None:
            interval = float(interval_sec)
            if interval <= 0:
                raise ValueError("interval_sec 必须大于 0")
            step = max(1, round(source_fps * interval)) if source_fps > 0 else max(1, int(fallback_step))
            filename_tag = f"fps{_filename_number(source_fps / step)}" if source_fps > 0 else f"step{step}"
        else:
            step = max(1, int(fallback_step))
            filename_tag = f"fps{_filename_number(source_fps / step)}" if source_fps > 0 else f"step{step}"

        requested_start = float(start_time_sec if start_time_sec is not None else start_skip_sec or 0)
        requested_end = float(end_time_sec) if end_time_sec is not None else None
        if requested_start < 0 or (requested_end is not None and requested_end <= requested_start):
            raise ValueError("结束时间必须晚于开始时间")
        needs_timing = requested_start > 0 or requested_end is not None or float(end_skip_sec or 0) > 0
        if needs_timing and source_fps <= 0:
            raise ValueError("无法读取源视频 FPS，不能按时间段抽帧")
        start_frame = max(0, min(frame_count, round(requested_start * source_fps))) if source_fps > 0 else 0
        if requested_end is not None:
            end_frame = round(requested_end * source_fps)
        else:
            end_frame = frame_count - round(float(end_skip_sec or 0) * source_fps)
        end_frame = max(0, min(frame_count, end_frame))
        if start_frame >= end_frame:
            raise ValueError("有效抽帧区间为空，请检查开始和结束时间")

        priority_candidates: list[int] = []
        background_candidates: list[int] = []
        if mixed:
            normalized: list[tuple[int, int]] = []
            for raw_start, raw_end in priority_ranges or []:
                left = max(start_frame, min(end_frame, round(float(raw_start) * source_fps)))
                right = max(start_frame, min(end_frame, round(float(raw_end) * source_fps)))
                if left < right:
                    normalized.append((left, right))
            normalized.sort()
            merged: list[tuple[int, int]] = []
            for left, right in normalized:
                if merged and left <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            if not merged:
                raise ValueError("填写的工序时间段不在当前视频的有效抽帧区间内")
            cursor = start_frame
            for left, right in merged:
                background_candidates.extend(_frames_for_rate(cursor, left, source_fps, background_fps))
                priority_candidates.extend(_frames_for_rate(left, right, source_fps, priority_fps))
                cursor = right
            background_candidates.extend(_frames_for_rate(cursor, end_frame, source_fps, background_fps))
            candidates = sorted(set(priority_candidates + background_candidates))
        else:
            candidates = list(range(start_frame, end_frame, step))

        limit_applied = max_images is not None and len(candidates) > int(max_images)
        if limit_applied and mixed and len(priority_candidates) < int(max_images):
            selected = sorted(priority_candidates + _uniform_sample(background_candidates, int(max_images) - len(priority_candidates)))
        elif limit_applied and mixed:
            selected = _uniform_sample(priority_candidates, int(max_images))
        elif limit_applied and limit_strategy == "全程均匀抽取":
            selected = _uniform_sample(candidates, int(max_images))
        elif limit_applied:
            selected = candidates[:int(max_images)]
        else:
            selected = candidates

        success = 0
        errors = 0
        progress = selected if report_progress is not None else tqdm(selected)
        for position, frame_index in enumerate(progress, start=1):
            if check_cancelled is not None:
                check_cancelled()
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                errors += 1
                continue
            time_tag = f"t{round(frame_index * 1000 / source_fps):09d}ms" if source_fps > 0 else "tunknown"
            image = Path(images_save_path) / f"{prefix}_{filename_tag}_{time_tag}_f{frame_index:08d}.jpg"
            if cv2.imwrite(str(image), frame):
                success += 1
            else:
                errors += 1
            if report_progress is not None:
                report_progress(position, len(selected))
        return {
            "success_count": success, "error_count": errors, "candidate_count": len(candidates),
            "selected_frame_indices": selected, "limit_applied": limit_applied,
            "limit_strategy": limit_strategy, "start_frame": start_frame, "end_frame": end_frame,
            "start_time_sec": start_frame / source_fps if source_fps > 0 else 0,
            "end_time_sec": end_frame / source_fps if source_fps > 0 else None,
            "filename_tag": filename_tag, "priority_candidate_count": len(priority_candidates),
            "background_candidate_count": len(background_candidates),
        }
    finally:
        capture.release()
