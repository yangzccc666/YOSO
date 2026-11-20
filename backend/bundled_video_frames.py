"""Self-contained video frame extraction used by source and packaged builds."""

from __future__ import annotations

import os
import json
import math
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
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


def _ffmpeg_capture(source, options: list[int]):
    try:
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, options)
    except cv2.error:
        return None  # Older builds may reject extra options or file objects.
    if capture.isOpened():
        return capture
    capture.release()
    return None


@contextmanager
def open_video(path: str):
    """Keep each FFmpeg decoder small instead of oversubscribing CPU threads."""
    options = []
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        options.extend([cv2.CAP_PROP_N_THREADS, 1])
    for name, milliseconds in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 10000), ("CAP_PROP_READ_TIMEOUT_MSEC", 15000)):
        if hasattr(cv2, name):
            options.extend([getattr(cv2, name), milliseconds])
    if sys.platform == "win32" and not path.isascii():
        # Open non-ASCII Windows paths through Python. The decoder receives
        # a seekable file stream and never needs to interpret the filename.
        with Path(path).open("rb") as source:
            capture = _ffmpeg_capture(source, options)
            if capture is None and options:
                source.seek(0)
                capture = _ffmpeg_capture(source, [])
            if capture is not None:
                try:
                    yield capture
                finally:
                    capture.release()
                return
    capture = _ffmpeg_capture(path, options) if options else None
    if capture is None:
        capture = cv2.VideoCapture(path)
    try:
        yield capture
    finally:
        capture.release()


def _save_jpeg(path: Path, frame) -> None:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("JPEG 编码失败")
    stream = path.open("xb")
    try:
        with stream:
            stream.write(encoded)
    except OSError:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass  # Preserve the original write error.
        raise


def _stream_frame_count(path: str, frame_count: int, fps: float,
                        check_cancelled: Callable[[], None] | None,
                        report: Callable[[str], None]) -> tuple[int, float]:
    """Optional stream-only metadata, never the container/audio duration.

    Packaged builds without ffprobe still use bounded sequential decoding.
    Metadata inspection is cancellable, time-bounded and never changes input.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        report("未发现 ffprobe，使用解码器估算总帧数；以实际可解码范围为准。")
        return frame_count, fps
    process = None
    try:
        def probe(fields: str, count_packets: bool = False) -> dict[str, Any]:
            nonlocal process
            options = ["-count_packets"] if count_packets else []
            process = subprocess.Popen([ffprobe, "-v", "error", "-select_streams", "v:0", *options,
                "-show_entries", fields, "-of", "json", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            deadline = time.monotonic() + 8
            while True:
                if check_cancelled:
                    check_cancelled()
                try:
                    stdout, _ = process.communicate(timeout=.2)
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        report("视频流信息检查超时，改用顺序解码，不会按异常元数据反复跳帧。")
                        return {}
            if process.returncode:
                return {}
            streams = json.loads(stdout).get("streams", [])
            return streams[0] if streams else {}
        stream = probe("stream=duration,nb_frames,avg_frame_rate")
        if not stream:
            return frame_count, fps
        if fps <= 0:
            numerator, denominator = str(stream.get("avg_frame_rate", "0/1")).split("/")
            fps = float(numerator) / float(denominator)
        exact = int(stream.get("nb_frames") or 0) if str(stream.get("nb_frames", "")).isdigit() else 0
        duration = float(stream.get("duration") or 0) if str(stream.get("duration", "")) != "N/A" else 0
        corrected = exact
        if corrected <= 0 and math.isfinite(duration) and duration > 0 and fps > 0:
            if frame_count <= 0 or abs(frame_count / fps - duration) > max(1.0, duration * .02):
                corrected = math.ceil(duration * fps - 1e-6)
                # NVR MP4 files may also contain timestamp gaps, so duration ×
                # FPS can still overshoot. Demux sample packets only for these
                # suspicious files; this does not decode or change the video.
                report("发现视频时长元数据不一致，正在核对视频流采样数量（不重新编码）。")
                counted = probe("stream=nb_read_packets", count_packets=True)
                raw_count = str(counted.get("nb_read_packets", ""))
                if raw_count.isdigit() and int(raw_count) > 0:
                    corrected = int(raw_count)
        if corrected > 0 and corrected != frame_count:
            report(f"视频流校正：总帧数 {frame_count} → {corrected}；仅采用视频轨道信息，不采用容器总时长。")
            frame_count = corrected
        return frame_count, fps
    except (OSError, ValueError, TypeError, KeyError, ZeroDivisionError, AttributeError, OverflowError):
        report("视频流元数据不可用，使用顺序解码，并在无法继续读取时及时结束。")
        return frame_count, fps
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.communicate()


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
    report_status: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    Path(images_save_path).mkdir(parents=True, exist_ok=True)
    prefix = str(filename_prefix or Path(vedio_path).stem)
    mixed = bool(priority_ranges)
    with open_video(vedio_path) as capture:
        if not capture.isOpened():
            raise RuntimeError("无法打开输入视频。请确认文件可读取，且视频编码受当前解码器支持。")
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_count = max(0, int(raw_count)) if math.isfinite(raw_count) else 0
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(source_fps) or source_fps < 0:
            source_fps = 0.0
        emit = report_status or (lambda _message: None)
        frame_count, source_fps = _stream_frame_count(vedio_path, frame_count, source_fps, check_cancelled, emit)
        if frame_count <= 0:
            raise RuntimeError("视频已打开，但未读到有效的总帧数。请检查视频轨道或重新导出该视频。")
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
        read_errors = 0
        write_errors = 0
        skipped = 0
        processed = 0
        saved_indices: list[int] = []
        warnings: list[str] = []
        cursor = -1
        last_status = 0.0
        def warn(message: str) -> None:
            warnings.append(message)
            emit(message)
        emit(f"顺序解码已开始，计划抽取 {len(selected)} 张；进度表示已处理计划项，不等于成功保存数量。")
        progress = selected if report_progress is not None else tqdm(selected)
        for position, frame_index in enumerate(progress, start=1):
            if check_cancelled is not None:
                check_cancelled()
            readable = True
            # Advance once in source order. grab avoids converting skipped
            # frames into full RGB arrays and eliminates repeated GOP seeks.
            while cursor < frame_index:
                if check_cancelled is not None:
                    check_cancelled()
                if not capture.grab():
                    readable = False
                    break
                cursor += 1
                now = time.monotonic()
                if now - last_status >= 5:
                    emit(f"正在顺序解码源帧 {cursor + 1}/{frame_count}；已保存 {success} 张，读取异常 {read_errors} 次，保存失败 {write_errors} 张。")
                    last_status = now
            processed = position
            if not readable:
                read_errors += 1
                skipped = len(selected) - position
                warn(f"在源帧 {cursor + 1} 无法继续解码，可能已提前到结尾或视频存在损坏；已停止本视频的无效读取，保留 {success} 张，剩余 {skipped} 个计划项未执行。")
            else:
                ok, frame = capture.retrieve()
                if not ok or frame is None or frame.size == 0:
                    read_errors += 1
                    if read_errors == 1:
                        warn(f"源帧 {frame_index} 图像读取失败，将继续后续帧并持续更新进度。")
                else:
                    time_tag = f"t{round(frame_index * 1000 / source_fps):09d}ms" if source_fps > 0 else "tunknown"
                    image = Path(images_save_path) / f"{prefix}_{filename_tag}_{time_tag}_f{frame_index:08d}.jpg"
                    try:
                        _save_jpeg(image, frame)
                    except (cv2.error, OSError, ValueError) as exc:
                        write_errors += 1
                        if write_errors == 1:
                            warn(f"图片保存失败（源帧 {frame_index}）：{image}。原因：{exc}")
                    else:
                        success += 1
                        saved_indices.append(frame_index)
            if report_progress is not None:
                report_progress(position, len(selected))
            if not readable:
                break
        emit(f"本视频解码结束：计划 {len(selected)} 张，实际保存 {success} 张，读取异常 {read_errors} 次，保存失败 {write_errors} 张，未执行 {skipped} 项。")
        return {
            "success_count": success, "error_count": read_errors + write_errors, "candidate_count": len(candidates),
            "read_error_count": read_errors, "write_error_count": write_errors, "skipped_count": skipped,
            "planned_count": len(selected), "processed_count": processed, "warnings": warnings,
            "selected_frame_indices": saved_indices, "planned_frame_indices": selected, "limit_applied": limit_applied,
            "limit_strategy": limit_strategy, "start_frame": start_frame, "end_frame": end_frame,
            "start_time_sec": start_frame / source_fps if source_fps > 0 else 0,
            "end_time_sec": end_frame / source_fps if source_fps > 0 else None,
            "filename_tag": filename_tag, "priority_candidate_count": len(priority_candidates),
            "background_candidate_count": len(background_candidates),
        }
