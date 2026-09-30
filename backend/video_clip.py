"""Video clipping handler backed by FFmpeg."""

from __future__ import annotations

import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "video.clip"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
CLIP_MODES = {"指定开始和结束时间", "指定开始时间和持续时长", "跳过开头和结尾"}
COPY_MODE = "原画质裁剪（推荐，不重新编码）"
PRECISE_MODE = "精确裁剪（重新编码）"
LEGACY_ENCODING_MODES = {
    "快速裁剪（不重新编码）": COPY_MODE,
    "精确裁剪（推荐）": PRECISE_MODE,
}
ENCODING_MODES = {COPY_MODE, PRECISE_MODE}


def parse_time(value: Any, label: str) -> float:
    """Parse seconds, MM:SS, or HH:MM:SS into non-negative seconds."""
    if isinstance(value, (int, float)):
        seconds = float(value)
    else:
        text = str(value if value is not None else "").strip()
        if not text:
            raise ValueError(f"请输入{label}。")
        parts = text.split(":")
        try:
            if len(parts) == 1:
                seconds = float(parts[0])
            elif len(parts) == 2:
                minutes, second_part = (float(part) for part in parts)
                if not 0 <= second_part < 60:
                    raise ValueError
                seconds = minutes * 60 + second_part
            elif len(parts) == 3:
                hours, minutes, second_part = (float(part) for part in parts)
                if not 0 <= minutes < 60 or not 0 <= second_part < 60:
                    raise ValueError
                seconds = hours * 3600 + minutes * 60 + second_part
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"{label}格式不正确，请输入秒数或 HH:MM:SS。") from exc
    if seconds < 0:
        raise ValueError(f"{label}不能小于 0。")
    return seconds


def _find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(
            "未找到 FFmpeg。请安装系统 FFmpeg，或运行 pip install imageio-ffmpeg。"
        ) from exc


def _run_process(command: list[str], context: RunContext) -> subprocess.CompletedProcess[str]:
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        while True:
            context.check_cancelled()
            try:
                stdout, stderr = process.communicate(timeout=0.2)
                return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
            except subprocess.TimeoutExpired:
                continue
    except TaskCancelled:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise


def _probe_duration(video: Path, context: RunContext | None = None) -> float:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
                ffprobe,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video),
            ]
        completed = _run_process(command, context) if context else subprocess.run(
            command, capture_output=True, text=True, check=False
        )
        try:
            duration = float(completed.stdout.strip())
            if duration > 0:
                return duration
        except ValueError:
            pass

    try:
        import cv2

        capture = cv2.VideoCapture(str(video))
        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        finally:
            capture.release()
        if fps > 0 and frame_count > 0:
            return frame_count / fps
    except ImportError:
        pass
    raise RuntimeError(f"无法读取视频时长：{video.name}")


def _resolve_videos(context: RunContext) -> list[Path]:
    candidates: list[Path] = []
    folder = context.paths.get("video_folder")
    video_file = context.paths.get("video_file")
    if folder:
        folder = folder.expanduser().resolve()
        if not folder.is_dir():
            raise ValueError(f"视频文件夹不存在：{folder}")
        candidates.extend(
            path for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
        )
    if video_file:
        video_file = video_file.expanduser().resolve()
        if not video_file.is_file() or video_file.suffix.lower() not in VIDEO_SUFFIXES:
            raise ValueError(f"视频文件不存在或格式不支持：{video_file}")
        candidates.append(video_file)
    if not folder and not video_file:
        raise ValueError("请选择视频文件夹，或者选择一个视频文件。")
    videos = sorted(set(candidates), key=lambda path: str(path).casefold())
    if not videos:
        raise ValueError("没有找到可裁剪的视频，支持 MP4、MOV、MKV、AVI、M4V 和 WEBM。")
    return videos


def _time_token(seconds: float) -> str:
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    token = f"{hours:02d}{minutes:02d}{whole_seconds:02d}"
    return f"{token}_{millis:03d}" if millis else token


def _unique_output_file(parent: Path, video: Path, start: float, end: float, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{video.stem}_clip_{_time_token(start)}-{_time_token(end)}_{timestamp}"
    candidate = parent / f"{base}{suffix}"
    index = 2
    while candidate.exists():
        candidate = parent / f"{base}_{index}{suffix}"
        index += 1
    return candidate


def _configured_range(parameters: dict[str, Any], video_duration: float) -> tuple[float, float, bool]:
    mode = str(parameters.get("clip_mode", "指定开始和结束时间"))
    if mode not in CLIP_MODES:
        raise ValueError("请选择有效的裁剪时间方案。")

    if mode == "指定开始和结束时间":
        start = parse_time(parameters.get("range_start_time", "00:00:00"), "开始时间")
        raw_end = parameters.get("range_end_time", "")
        requested_end = (
            video_duration
            if str(raw_end if raw_end is not None else "").strip() == ""
            else parse_time(raw_end, "结束时间")
        )
        if requested_end <= start:
            raise ValueError("结束时间必须晚于开始时间。")
    elif mode == "指定开始时间和持续时长":
        start = parse_time(parameters.get("duration_start_time", "00:00:00"), "开始时间")
        duration = parse_time(parameters.get("clip_duration", "00:01:00"), "持续时长")
        if duration <= 0:
            raise ValueError("持续时长必须大于 0。")
        requested_end = start + duration
    else:
        head_skip = parse_time(parameters.get("head_skip_time", "00:00:00"), "跳过开头时长")
        tail_skip = parse_time(parameters.get("tail_skip_time", "00:00:00"), "忽略结尾时长")
        start = head_skip
        requested_end = video_duration - tail_skip
        if requested_end <= start:
            raise ValueError("跳过开头和结尾后没有可保留的视频内容。")

    if start >= video_duration:
        raise ValueError(f"开始时间已超出视频总时长（{video_duration:.2f} 秒）。")
    end = min(requested_end, video_duration)
    if end - start <= 0.01:
        raise ValueError("有效裁剪时长过短，请调整时间参数。")
    return start, end, requested_end > video_duration


def _clip_one(
    ffmpeg: str,
    video: Path,
    output_parent: Path,
    start: float,
    end: float,
    encoding_mode: str,
    context: RunContext,
) -> Path:
    precise = encoding_mode == PRECISE_MODE
    suffix = ".mp4" if precise else video.suffix.lower()
    output = _unique_output_file(output_parent, video, start, end, suffix)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    duration = end - start

    command_prefix = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-ss", f"{start:.6f}",
        "-i", str(video),
        "-t", f"{duration:.6f}",
    ]
    if precise:
        output_options = [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
        ]
    else:
        output_options = [
            "-map", "0",
            "-c", "copy",
            "-map_metadata", "0",
            "-avoid_negative_ts", "make_zero",
        ]

    def execute(options: list[str]) -> subprocess.CompletedProcess[str]:
        temporary.unlink(missing_ok=True)
        return _run_process([*command_prefix, *options, str(temporary)], context)

    def succeeded(completed: subprocess.CompletedProcess[str]) -> bool:
        return completed.returncode == 0 and temporary.is_file() and temporary.stat().st_size > 0

    try:
        completed = execute(output_options)
        incompatible_mp4_audio = (
            not precise
            and output.suffix.lower() in {".mp4", ".m4v", ".mov"}
            and not succeeded(completed)
            and any(marker in completed.stderr for marker in (
                "Could not find tag for codec",
                "codec not currently supported in container",
                "Could not write header",
            ))
        )
        if incompatible_mp4_audio:
            context.report(
                f"{video.name}：原音频编码与 MP4 不兼容，视频保持原编码，仅将音频转换为 AAC 后重试。"
            )
            completed = execute([
                "-map", "0:v:0",
                "-map", "0:a?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "128k",
                "-map_metadata", "0",
                "-avoid_negative_ts", "make_zero",
                "-movflags", "+faststart",
            ])
    except TaskCancelled:
        temporary.unlink(missing_ok=True)
        raise
    if not succeeded(completed):
        temporary.unlink(missing_ok=True)
        detail = completed.stderr.strip().splitlines()
        reason = next(
            (line for line in detail if "Could not find tag for codec" in line or "not currently supported in container" in line),
            detail[-1] if detail else "FFmpeg 未生成结果文件",
        )
        raise RuntimeError(f"裁剪失败：{reason}")
    temporary.replace(output)
    return output


def run_video_clip(context: RunContext) -> dict[str, Any]:
    videos = _resolve_videos(context)
    ffmpeg = _find_ffmpeg()
    encoding_mode = str(context.parameters.get("encoding_mode", COPY_MODE))
    encoding_mode = LEGACY_ENCODING_MODES.get(encoding_mode, encoding_mode)
    if encoding_mode not in ENCODING_MODES:
        raise ValueError("请选择有效的裁剪处理方式。")

    output_root = context.paths.get("output_folder")
    if output_root:
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    context.report(f"找到 {len(videos)} 个视频，处理方式：{encoding_mode}")
    if encoding_mode == COPY_MODE:
        context.report("将直接复制原视频编码流，不改变分辨率、码率和画质；裁剪起点会对齐到附近的关键帧。")
    outputs: list[Path] = []
    details: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for index, video in enumerate(videos, start=1):
        try:
            context.check_cancelled()
            video_duration = _probe_duration(video, context)
            start, end, clamped = _configured_range(context.parameters, video_duration)
            if clamped:
                context.report(f"[{index}/{len(videos)}] {video.name}：结束位置超出视频，已自动截到结尾")
            context.report(
                f"[{index}/{len(videos)}] 正在裁剪 {video.name}："
                f"{start:.3f} 秒 → {end:.3f} 秒（{end - start:.3f} 秒）"
            )
            output_parent = output_root or video.parent
            output_parent.mkdir(parents=True, exist_ok=True)
            output = _clip_one(ffmpeg, video, output_parent, start, end, encoding_mode, context)
            source_size = video.stat().st_size
            output_size = output.stat().st_size
            outputs.append(output)
            details.append({
                "video": str(video),
                "output": str(output),
                "startSec": start,
                "endSec": end,
                "durationSec": end - start,
                "sourceDurationSec": video_duration,
                "encodingMode": encoding_mode,
                "sourceSizeBytes": source_size,
                "outputSizeBytes": output_size,
            })
            context.report(
                f"[{index}/{len(videos)}] 完成：{output.name}，"
                f"文件大小 {source_size / 1024 / 1024:.1f} MB → {output_size / 1024 / 1024:.1f} MB"
            )
        except TaskCancelled:
            raise
        except (ValueError, RuntimeError) as exc:
            failures.append({"video": str(video), "error": str(exc)})
            context.report(f"[{index}/{len(videos)}] 跳过 {video.name}：{exc}")

    if not outputs:
        reason = failures[0]["error"] if failures else "没有生成结果"
        raise RuntimeError(f"所有视频均未成功裁剪：{reason}")

    output_folders = list(dict.fromkeys(str(path.parent) for path in outputs))
    message = f"视频裁剪完成：成功 {len(outputs)} 个"
    if failures:
        message += f"，跳过 {len(failures)} 个"
    return {
        "message": message + "。",
        "videoCount": len(videos),
        "successCount": len(outputs),
        "failedCount": len(failures),
        "outputFiles": [str(path) for path in outputs],
        "outputFolders": output_folders,
        "details": details,
        "failures": failures,
    }


register_handler(HANDLER_ID, run_video_clip)
