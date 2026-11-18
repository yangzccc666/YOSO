"""Video clipping handler backed by FFmpeg."""

from __future__ import annotations

import math
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from .handlers import RunContext, register_handler
from .task_manager import TaskCancelled


HANDLER_ID = "video.clip"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
REMOVE_MODE = "删除多个时间段并拼接剩余"
CLIP_MODES = {"指定开始和结束时间", "指定开始时间和持续时长", "跳过开头和结尾", REMOVE_MODE}
COPY_MODE = "原画质裁剪（推荐，不重新编码）"
PRECISE_MODE = "精确裁剪（重新编码）"
LEGACY_ENCODING_MODES = {
    "快速裁剪（不重新编码）": COPY_MODE,
    "精确裁剪（推荐）": PRECISE_MODE,
}
ENCODING_MODES = {COPY_MODE, PRECISE_MODE}
SAVE_AS = '另存为新视频'
REPLACE_SOURCE = '替换原视频（自动备份）'
_replacing_sources: set[Path] = set()
_replacement_lock = threading.Lock()


@contextmanager
def _claim_source(video: Path):
    with _replacement_lock:
        if video in _replacing_sources:
            raise ValueError('该视频正在被另一个任务原位裁剪，请等待完成。')
        _replacing_sources.add(video)
    try:
        yield
    finally:
        with _replacement_lock:
            _replacing_sources.discard(video)


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
    if not math.isfinite(seconds):
        raise ValueError(f"{label}必须是有限的时间数值。")
    if seconds < 0:
        raise ValueError(f"{label}不能小于 0。")
    return seconds


def parse_removal_ranges(value: Any) -> list[tuple[float, float | None]]:
    """Read removal intervals on the original timeline, never a shifted one."""
    rows = [row.strip() for row in re.split(r"[\n;；]+", str(value or "")) if row.strip()]
    if not rows:
        raise ValueError("请填写需要删除的时间段，每行一个，例如 00:10-00:20。")
    ranges = []
    for index, row in enumerate(rows, 1):
        parts = re.split(r"\s*(?:-|–|—|~|～|至)\s*", row)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError(f"第 {index} 个删除时间段格式不正确：{row}。请填写 开始时间-结束时间。")
        start = parse_time(parts[0].strip(), f"第 {index} 段开始时间")
        raw_end = parts[1].strip()
        end = None if raw_end in {"", "结尾", "末尾", "end"} else parse_time(raw_end, f"第 {index} 段结束时间")
        if end is not None and end <= start:
            raise ValueError(f"第 {index} 个删除时间段的结束时间必须晚于开始时间。")
        ranges.append((start, end))
    return ranges


def remaining_ranges(
    ranges: list[tuple[float, float | None]], duration: float,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], bool]:
    """Clamp, sort and merge deleted intervals, then compute their complement."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("视频时长无效。")
    clipped = []
    adjusted = False
    for start, end in ranges:
        stop = duration if end is None else min(end, duration)
        adjusted |= start >= duration or (end is not None and end > duration)
        if start < stop:
            clipped.append((start, stop))
    merged: list[tuple[float, float]] = []
    for start, end in sorted(clipped):
        if merged and start <= merged[-1][1] + 1e-6:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    if not merged:
        raise ValueError("删除时间段均在视频时长之外，没有可删除的内容。")
    kept = []
    cursor = 0.0
    for start, end in merged:
        if start - cursor > 1e-6:
            kept.append((cursor, start))
        cursor = end
    if duration - cursor > 1e-6:
        kept.append((cursor, duration))
    if not kept:
        raise ValueError("删除时间段覆盖了整个视频，没有可保留的内容，请调整时间段。")
    return kept, merged, adjusted


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
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()


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


def _configured_output_name(value: Any) -> str:
    name = str(value or "").strip()
    if not name:
        return ""
    if re.search(r'[<>:"/\\|?*\x00-\x1f\x7f]', name):
        raise ValueError('保存文件名不能包含路径或以下字符：< > : " / \\ | ? *。请在输出文件夹中设置保存位置。')
    if name.lower() in VIDEO_SUFFIXES:
        raise ValueError("保存文件名不能只有扩展名，请填写可区分的名称。")
    # Accept an optional video extension; the actual container follows the encoding mode.
    if Path(name).suffix.lower() in VIDEO_SUFFIXES:
        name = name[:-len(Path(name).suffix)]
    if not name or name in {".", ".."} or name.endswith((".", " ")):
        raise ValueError("保存文件名不能只有扩展名、句点，或以句点和空格结尾。")
    if name.split(".", 1)[0].upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("保存文件名使用了系统保留名称，请更换名称。")
    if len(name.encode("utf-8")) > 180:
        raise ValueError("保存文件名过长，请限制在 180 字节以内（约 60 个汉字或 180 个英文字母）。")
    return name


def _named_output_stem(name: str, video: Path, batch: bool, label: str = "") -> str:
    if not name:
        return ""
    if batch:
        source = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f]', "_", video.stem)
        # Leave space for the role, duplicate sequence and container extension.
        budget = 220 - len(name.encode("utf-8")) - 1
        source = source.encode("utf-8")[:budget].decode("utf-8", errors="ignore").rstrip(". ") or "video"
        name = f"{name}_{source}"
    return f"{name}_{label}" if label else name


def _unique_output_file(parent: Path, video: Path, start: float, end: float, suffix: str, label: str = "clip", output_name: str = "") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_name or f"{video.stem}_{label}_{_time_token(start)}-{_time_token(end)}_{timestamp}"
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
    label: str = "clip",
    output_name: str = "",
) -> Path:
    precise = encoding_mode == PRECISE_MODE
    suffix = ".mp4" if precise else video.suffix.lower()
    output = _unique_output_file(output_parent, video, start, end, suffix, label, output_name)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    duration = end - start

    command_prefix = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(video),
    ]
    if precise:
        output_options = [
            "-map", "0:v:0",
            "-map", "0:a?",
            "-vf", f"setpts=PTS-STARTPTS,trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS",
            "-af", f"asetpts=PTS-STARTPTS,atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS",
            "-vsync", "vfr",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
        ]
    else:
        command_prefix = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                          '-ss', f'{start:.6f}', '-i', str(video), '-t', f'{duration:.6f}']
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
    except BaseException:
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


def _concat_clips(
    ffmpeg: str,
    clips: list[Path],
    output: Path,
    context: RunContext,
) -> Path:
    """Join compatible clips without another video re-encode."""
    if not clips:
        raise ValueError("没有可拼接的视频片段。")
    context.check_cancelled()
    if len(clips) == 1:
        clips[0].replace(output)
        return output
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", encoding="utf-8", delete=False) as listing:
        list_path = Path(listing.name)
        for clip in clips:
            escaped = str(clip.resolve()).replace("'", "'\\''")
            listing.write(f"file '{escaped}'\n")
    try:
        completed = _run_process([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(list_path),
            "-map", "0", "-c", "copy", "-avoid_negative_ts", "make_zero",
            str(temporary),
        ], context)
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
            detail = completed.stderr.strip().splitlines()
            raise RuntimeError(f"视频片段合并失败：{detail[-1] if detail else 'FFmpeg 未生成结果文件'}")
        context.check_cancelled()
        temporary.replace(output)
        return output
    except TaskCancelled:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)
        list_path.unlink(missing_ok=True)


def _has_audio(ffmpeg: str, video: Path, context: RunContext) -> bool:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        completed = _run_process([
            ffprobe, "-v", "error", "-select_streams", "a:0",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(video),
        ], context)
        if completed.returncode != 0:
            raise RuntimeError(f"无法检查视频音轨：{video.name}")
        return bool(completed.stdout.strip())
    # The bundled FFmpeg distribution need not contain an ffprobe executable.
    completed = _run_process([ffmpeg, "-hide_banner", "-i", str(video)], context)
    return bool(re.search(r"Stream #.*Audio:", completed.stderr))


def _join_remaining(
    ffmpeg: str, video: Path, parent: Path, ranges: list[tuple[float, float]],
    duration: float, encoding_mode: str, context: RunContext, output_name: str = "",
) -> Path:
    precise = encoding_mode == PRECISE_MODE
    suffix = ".mp4" if precise else video.suffix.lower()
    output = _unique_output_file(parent, video, 0, duration, suffix, "removed_joined", output_name)
    temporary = output.with_name(f".{output.stem}.partial{suffix}")
    try:
        if not precise:
            with tempfile.TemporaryDirectory(prefix=".video_remove_parts_", dir=parent) as staging:
                parts = []
                for index, (start, end) in enumerate(ranges, 1):
                    context.report(f"保留片段 {index}/{len(ranges)}：{start:.3f} 秒 → {end:.3f} 秒")
                    parts.append(_clip_one(ffmpeg, video, Path(staging), start, end, encoding_mode, context, f"part{index}"))
                context.report(f"正在按原始顺序拼接 {len(parts)} 个保留片段")
                return _concat_clips(ffmpeg, parts, output, context)

        # Trim and concatenate before encoding: no per-clip AAC padding or
        # repeated encoding, which can otherwise accumulate A/V drift.
        audio = _has_audio(ffmpeg, video, context)
        filters = []
        count = len(ranges)
        filters.append(f"[0:v:0]setpts=PTS-STARTPTS,split={count}" + "".join(f"[sv{i}]" for i in range(count)))
        if audio:
            filters.append(f"[0:a:0]asetpts=PTS-STARTPTS,asplit={count}" + "".join(f"[sa{i}]" for i in range(count)))
        inputs = []
        for index, (start, end) in enumerate(ranges):
            context.report(f"保留片段 {index + 1}/{count}：{start:.3f} 秒 → {end:.3f} 秒")
            filters.append(f"[sv{index}]trim=start={start:.6f}:end={end:.6f},setpts=PTS-STARTPTS[v{index}]")
            inputs.append(f"[v{index}]")
            if audio:
                filters.append(
                    f"[sa{index}]atrim=start={start:.6f}:end={end:.6f},asetpts=PTS-STARTPTS,"
                    f"apad=whole_dur={end-start:.6f},atrim=duration={end-start:.6f}[a{index}]"
                )
                inputs.append(f"[a{index}]")
        filters.append("".join(inputs) + f"concat=n={count}:v=1:a={int(audio)}[vout]" + ("[aout]" if audio else ""))
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
            "-filter_complex_threads", "1", "-filter_complex", ";".join(filters), "-map", "[vout]",
        ]
        if audio:
            command += ["-map", "[aout]", "-c:a", "aac", "-b:a", "128k"]
        command += ["-vsync", "vfr", "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-map_metadata", "0", "-movflags", "+faststart", str(temporary)]
        audio_note = "保留首条音轨" if audio else "原视频无音轨"
        context.report(f"正在精确删除并拼接 {count} 个保留片段（一次编码，{audio_note}）")
        completed = _run_process(command, context)
        if completed.returncode != 0 or not temporary.is_file() or temporary.stat().st_size <= 0:
            lines = completed.stderr.strip().splitlines()
            raise RuntimeError(f"删除并拼接失败：{lines[-1] if lines else 'FFmpeg 未生成结果文件'}")
        context.check_cancelled()
        temporary.replace(output)
        return output
    finally:
        temporary.unlink(missing_ok=True)


def _video_timing(video: Path, context: RunContext) -> dict[str, float]:
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        result = _run_process([ffprobe, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=duration,avg_frame_rate,width,height', '-of', 'json', str(video)], context)
        try:
            stream = json.loads(result.stdout)['streams'][0]
            numerator, denominator = str(stream['avg_frame_rate']).split('/')
            fps = float(numerator) / float(denominator)
            duration = float(stream.get('duration', 0))
            if result.returncode == 0 and fps > 0 and duration > 0:
                return {'fps': fps, 'duration': duration, 'width': int(stream['width']), 'height': int(stream['height'])}
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
            pass
    import cv2
    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError(f'无法验证裁剪结果的视频轨道：{video.name}')
    return {'fps': fps, 'duration': frames / fps, 'width': width, 'height': height}


def _validate_precise_output(output: Path, expected: float, source: dict[str, float], context: RunContext, segments=1):
    timing = _video_timing(output, context)
    # Fractional boundaries can only select existing frames. Each kept segment
    # contributes at most one source-frame rounding interval for CFR video.
    tolerance = max(0.02, segments / source['fps'] + 0.005)
    if (timing['width'], timing['height']) != (source['width'], source['height']):
        raise RuntimeError('裁剪结果分辨率校验失败，原视频未修改。')
    if abs(timing['duration'] - expected) > tolerance:
        raise RuntimeError(f"裁剪结果时间校验失败：期望{expected:.6f}秒，实际视频轨道{timing['duration']:.6f}秒，容差{tolerance:.6f}秒；原视频未修改。")
    context.report(f"时间校验：期望 {expected:.6f} 秒，实际视频轨道 {timing['duration']:.6f} 秒（帧级容差 {tolerance:.6f} 秒）。")
    return timing['duration']


def _source_signature(video: Path):
    state = video.stat()
    return state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns


def _replace_video(ffmpeg: str, video: Path, duration: float, context: RunContext):
    """Encode away from the source, validate, copy a recoverable backup, then atomically commit."""
    with _claim_source(video):
        signature = _source_signature(video)
        source = _video_timing(video, context)
        remove = context.parameters.get('clip_mode') == REMOVE_MODE
        if remove:
            kept, removed, adjusted = remaining_ranges(parse_removal_ranges(context.parameters.get('remove_time_ranges')), duration)
            if adjusted:
                raise ValueError('删除区间超出视频时长，原位精确裁剪不会自动截断；请修改时间。')
        else:
            start, end, clamped = _configured_range(context.parameters, duration)
            if clamped:
                raise ValueError('结束时间超出视频时长，原位精确裁剪不会自动截断；请修改时间或留空到结尾。')
            kept, removed = [(start, end)], []
        expected = sum(end-start for start, end in kept)
        context.report(f'原位裁剪 {video.name}：保留 {kept}，预计 {expected:.6f} 秒。')
        with tempfile.TemporaryDirectory(prefix='.video_clip_staging_', dir=video.parent) as temporary:
            staging = Path(temporary)
            output = (_join_remaining(ffmpeg, video, staging, kept, duration, PRECISE_MODE, context, 'replacement')
                      if remove else _clip_one(ffmpeg, video, staging, kept[0][0], kept[0][1], PRECISE_MODE, context, output_name='replacement'))
            if output.suffix.lower() != video.suffix.lower():
                remuxed = staging / f'replacement{video.suffix.lower()}'
                container = {'.mkv': 'matroska', '.mov': 'mov', '.m4v': 'ipod'}[video.suffix.lower()]
                completed = _run_process([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(output),
                    '-map', '0', '-c', 'copy', '-f', container, str(remuxed)], context)
                if completed.returncode != 0:
                    raise RuntimeError(f'无法保持原视频容器格式，原视频未修改：{completed.stderr.strip()}')
                output = remuxed
            actual = _validate_precise_output(output, expected, source, context, len(kept))
            context.report('正在完整解码验证裁剪结果，验证通过后才会备份并替换原视频。')
            completed = _run_process([ffmpeg, '-hide_banner', '-loglevel', 'error', '-xerror', '-i', str(output),
                '-map', '0:v:0', '-an', '-f', 'null', '-'], context)
            if completed.returncode != 0:
                raise RuntimeError('裁剪结果完整解码校验失败，原视频未修改。')
            context.check_cancelled()
            if _source_signature(video) != signature:
                raise RuntimeError('处理期间原视频被其他程序修改，已取消替换。')
            backup_root = video.parent / '.video_clip_backups'
            if backup_root.is_symlink():
                raise ValueError('备份目录不能是符号链接，原视频未修改。')
            backup_root.mkdir(exist_ok=True)
            backup_dir = Path(tempfile.mkdtemp(prefix=datetime.now().strftime('%Y%m%d_%H%M%S_'), dir=backup_root))
            backup = backup_dir / video.name
            try:
                with video.open('rb') as source_stream, backup.open('xb') as target:
                    while chunk := source_stream.read(8 * 1024 * 1024):
                        context.check_cancelled()
                        target.write(chunk)
                    target.flush()
                    os.fsync(target.fileno())
                shutil.copystat(video, backup)
                context.check_cancelled()
                if _source_signature(video) != signature:
                    raise RuntimeError('备份期间原视频被其他程序修改，已取消替换。')
                with output.open('rb+') as stream:
                    os.fsync(stream.fileno())
                output.chmod(video.stat().st_mode & 0o777)
            except BaseException:
                backup.unlink(missing_ok=True)
                backup_dir.rmdir()
                raise
            output_size = output.stat().st_size
            # Do not check cancellation between commit and recording its result:
            # a completed atomic replacement must remain reported as completed.
            context.report(f'原视频完整备份已保存：{backup}')
            context.check_cancelled()
            try:
                output.replace(video)
            except OSError as exc:
                raise RuntimeError(f'原视频替换失败，原文件保留，完整备份位于 {backup}：{exc}') from exc
            context.report(f'原视频已安全替换：{video}；原始备份：{backup}')
            return {'video': str(video), 'output': str(video), 'backupFile': str(backup),
                'saveMode': REPLACE_SOURCE, 'encodingMode': PRECISE_MODE, 'clipMode': context.parameters.get('clip_mode'),
                'keptRanges': [{'startSec': start, 'endSec': end} for start, end in kept],
                'removedRanges': [{'startSec': start, 'endSec': end} for start, end in removed],
                'durationSec': expected, 'actualDurationSec': actual, 'sourceDurationSec': duration,
                'sourceSizeBytes': signature[2], 'outputSizeBytes': output_size}


def run_video_clip(context: RunContext) -> dict[str, Any]:
    save_mode = context.parameters.get('clip_save_mode', SAVE_AS)
    if save_mode not in {SAVE_AS, REPLACE_SOURCE}:
        raise ValueError('请选择有效的视频保存方式。')
    replace_source = save_mode == REPLACE_SOURCE
    if replace_source and context.parameters.get('clip_replace_confirmed') is not True:
        raise ValueError('替换原视频前必须确认；请在界面点击运行并确认替换操作。')
    custom_name = '' if replace_source else _configured_output_name(context.parameters.get("clip_output_name", ""))
    videos = _resolve_videos(context)
    if replace_source:
        if context.parameters.get('split_train_test') and context.parameters.get('clip_mode') != REMOVE_MODE:
            raise ValueError('替换原视频不能同时拆分训练/测试集，请选择另存为新视频。')
        if (context.paths.get('video_file') and context.paths['video_file'].is_symlink()) or any(video.is_symlink() for video in videos):
            raise ValueError('原位裁剪不支持符号链接，请选择实际视频文件或另存为新视频。')
        if any(video.suffix.lower() not in {'.mp4', '.m4v', '.mov', '.mkv'} for video in videos):
            raise ValueError('原位精确裁剪支持 MP4、M4V、MOV、MKV；其他容器请另存为新 MP4。')
    ffmpeg = _find_ffmpeg()
    encoding_mode = str(context.parameters.get("encoding_mode", COPY_MODE))
    encoding_mode = LEGACY_ENCODING_MODES.get(encoding_mode, encoding_mode)
    if encoding_mode not in ENCODING_MODES:
        raise ValueError("请选择有效的裁剪处理方式。")
    exact_time = replace_source or context.parameters.get('clip_exact_time', False) is True
    if exact_time:
        encoding_mode = PRECISE_MODE
    remove_mode = context.parameters.get("clip_mode") == REMOVE_MODE
    removal = parse_removal_ranges(context.parameters.get("remove_time_ranges")) if remove_mode else []
    split_train_test = not remove_mode and bool(context.parameters.get("split_train_test", False))

    output_root = None if replace_source else context.paths.get("output_folder")
    if output_root:
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    context.report(f"找到 {len(videos)} 个视频，保存方式：{save_mode}；处理方式：{encoding_mode}")
    if encoding_mode == PRECISE_MODE:
        context.report('按原视频帧时间戳选择 [开始, 结束) 范围；精确到现有视频帧，不按关键帧取整。重新编码保持原分辨率，不保证文件体积与原视频相同。')
    if custom_name:
        context.report(f"自定义保存名称：{custom_name}。批量追加原视频名，训练/测试追加 train/test，同名结果自动加序号。")
    if remove_mode and not replace_source:
        context.report("删除时间均以原视频时间轴为准；重叠/相邻区间自动合并，剩余片段按原顺序拼成一个新视频，原视频不修改。")
    if split_train_test:
        context.report("已开启训练集/测试集拆分：指定时间段输出到 train，其余片段合并后输出到 test。")
    if encoding_mode == COPY_MODE:
        context.report("将直接复制原视频编码流，不改变分辨率、码率和画质；裁剪起点会对齐到附近的关键帧。")
    outputs: list[Path] = []
    backups: list[str] = []
    details: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    successful_videos = 0

    for index, video in enumerate(videos, start=1):
        created_for_video: list[Path] = []
        staging_dir: Path | None = None
        try:
            context.check_cancelled()
            source_timing = _video_timing(video, context) if encoding_mode == PRECISE_MODE else None
            video_duration = source_timing['duration'] if source_timing else _probe_duration(video, context)
            if replace_source:
                detail = _replace_video(ffmpeg, video, video_duration, context)
                outputs.append(video)
                backups.append(detail['backupFile'])
                details.append(detail)
                successful_videos += 1
                continue
            if remove_mode:
                kept, removed, adjusted = remaining_ranges(removal, video_duration)
                if adjusted and exact_time:
                    raise ValueError('删除时间段超出视频时长，精确时间模式不会自动截断，请修改时间。')
                if adjusted:
                    context.report(f"[{index}/{len(videos)}] {video.name}：超出视频结尾的区间已截断或忽略。")
                kept_duration = sum(end - start for start, end in kept)
                context.report(f"[{index}/{len(videos)}] {video.name}：删除 {len(removed)} 段，保留 {len(kept)} 段，预计输出 {kept_duration:.3f} 秒。")
                parent = output_root or video.parent
                parent.mkdir(parents=True, exist_ok=True)
                output = _join_remaining(ffmpeg, video, parent, kept, video_duration, encoding_mode, context,
                                         _named_output_stem(custom_name, video, len(videos) > 1))
                created_for_video.append(output)
                actual_duration = _validate_precise_output(output, kept_duration, source_timing, context, len(kept)) if source_timing else None
                source_size, output_size = video.stat().st_size, output.stat().st_size
                outputs.append(output)
                successful_videos += 1
                details.append({
                    "video": str(video), "output": str(output), "clipMode": REMOVE_MODE,
                    "removedRanges": [{"startSec": start, "endSec": end} for start, end in removed],
                    "keptRanges": [{"startSec": start, "endSec": end} for start, end in kept],
                    "durationSec": kept_duration, "sourceDurationSec": video_duration,
                    "encodingMode": encoding_mode, "sourceSizeBytes": source_size, "outputSizeBytes": output_size,
                    'actualDurationSec': actual_duration, 'saveMode': SAVE_AS,
                })
                context.report(f"[{index}/{len(videos)}] 完成：{output.name}，输出 {output_size / 1024 / 1024:.1f} MB")
                continue
            start, end, clamped = _configured_range(context.parameters, video_duration)
            if clamped and exact_time:
                raise ValueError('结束时间超出视频时长，精确时间模式不会自动截断；请修改时间或留空到结尾。')
            if clamped:
                context.report(f"[{index}/{len(videos)}] {video.name}：结束位置超出视频，已自动截到结尾")
            context.report(
                f"[{index}/{len(videos)}] 正在裁剪 {video.name}："
                f"{start:.3f} 秒 → {end:.3f} 秒（{end - start:.3f} 秒）"
            )
            base_output_parent = output_root or video.parent
            if split_train_test:
                train_parent = base_output_parent / "train"
                test_parent = base_output_parent / "test"
                train_parent.mkdir(parents=True, exist_ok=True)
                test_parent.mkdir(parents=True, exist_ok=True)
                output = _clip_one(ffmpeg, video, train_parent, start, end, encoding_mode, context, "train",
                                   _named_output_stem(custom_name, video, len(videos) > 1, "train"))
                created_for_video.append(output)
                suffix = ".mp4" if encoding_mode == PRECISE_MODE else video.suffix.lower()
                test_output = _unique_output_file(test_parent, video, 0, video_duration, suffix, "test",
                                                  _named_output_stem(custom_name, video, len(videos) > 1, "test"))
                test_ranges = [
                    (part_start, part_end)
                    for part_start, part_end in ((0.0, start), (end, video_duration))
                    if part_end - part_start > 0.01
                ]
                if not test_ranges:
                    raise ValueError("指定训练片段覆盖了整个视频，没有剩余内容可生成测试集视频。")
                if encoding_mode == PRECISE_MODE:
                    test_output = _join_remaining(ffmpeg, video, test_parent, test_ranges, video_duration, encoding_mode, context,
                        _named_output_stem(custom_name, video, len(videos) > 1, 'test'))
                else:
                    staging_dir = Path(tempfile.mkdtemp(prefix=".video_split_parts_", dir=test_parent))
                    parts = [
                        _clip_one(ffmpeg, video, staging_dir, part_start, part_end, encoding_mode, context, f"part{part_index}")
                        for part_index, (part_start, part_end) in enumerate(test_ranges, 1)
                    ]
                    context.report(f"[{index}/{len(videos)}] 正在合并 {len(parts)} 个剩余片段为测试集视频")
                    _concat_clips(ffmpeg, parts, test_output, context)
                created_for_video.append(test_output)
                output_group = [output, test_output]
            else:
                base_output_parent.mkdir(parents=True, exist_ok=True)
                output = _clip_one(ffmpeg, video, base_output_parent, start, end, encoding_mode, context,
                                   output_name=_named_output_stem(custom_name, video, len(videos) > 1))
                created_for_video.append(output)
                output_group = [output]
            actual_duration = _validate_precise_output(output, end-start, source_timing, context) if source_timing else None
            if split_train_test and source_timing:
                _validate_precise_output(test_output, sum(stop-begin for begin, stop in test_ranges), source_timing, context, len(test_ranges))
            source_size = video.stat().st_size
            output_size = sum(path.stat().st_size for path in output_group)
            outputs.extend(output_group)
            successful_videos += 1
            details.append({
                "video": str(video),
                "output": str(output),
                "trainOutput": str(output) if split_train_test else None,
                "testOutput": str(output_group[1]) if split_train_test else None,
                "startSec": start,
                "endSec": end,
                "durationSec": end - start,
                "sourceDurationSec": video_duration,
                "encodingMode": encoding_mode,
                "sourceSizeBytes": source_size,
                "outputSizeBytes": output_size,
                'actualDurationSec': actual_duration, 'saveMode': SAVE_AS,
            })
            context.report(
                f"[{index}/{len(videos)}] 完成："
                f"{'训练集与测试集各 1 个视频' if split_train_test else output.name}，"
                f"源文件 {source_size / 1024 / 1024:.1f} MB，输出合计 {output_size / 1024 / 1024:.1f} MB"
            )
        except TaskCancelled:
            for path in created_for_video:
                path.unlink(missing_ok=True)
            raise
        except (ValueError, RuntimeError, OSError) as exc:
            for path in created_for_video:
                path.unlink(missing_ok=True)
            failures.append({"video": str(video), "error": str(exc)})
            context.report(f"[{index}/{len(videos)}] 跳过 {video.name}：{exc}")
        finally:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)

    if not outputs:
        reason = failures[0]["error"] if failures else "没有生成结果"
        raise RuntimeError(f"所有视频均未成功裁剪：{reason}")

    output_folders = list(dict.fromkeys([*(str(path.parent) for path in outputs), *(str(Path(path).parent) for path in backups)]))
    message = f"视频裁剪完成：成功处理 {successful_videos} 个视频"
    if replace_source:
        message += f'，已替换原视频并保留 {len(backups)} 份原始备份'
    if split_train_test:
        message += f"，生成训练集和测试集视频共 {len(outputs)} 个"
    if failures:
        message += f"，跳过 {len(failures)} 个"
    return {
        "message": message + "。",
        "videoCount": len(videos),
        "successCount": successful_videos,
        "failedCount": len(failures),
        "outputFiles": [str(path) for path in outputs],
        "outputFolders": output_folders,
        "details": details,
        "failures": failures,
        'backupFiles': backups, 'saveMode': save_mode,
    }


register_handler(HANDLER_ID, run_video_clip)
