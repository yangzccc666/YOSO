"""Adapter for tools/data_processing_toolkit/02-video_processing/batch_vedio2img.py."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from types import ModuleType
from typing import Any

from .handlers import RunContext, register_handler
from .video_clip import parse_time
from .task_manager import TaskCancelled
from .video_frame_resources import DEFAULT_WORKER_MEMORY, MIB, resource_pool


HANDLER_ID = "video.extract_frames"
def _load_source_module() -> ModuleType:
    from . import bundled_video_frames
    return bundled_video_frames


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


def _priority_time_ranges(value: Any) -> list[tuple[float, float]]:
    text = str(value or "").strip()
    if not text:
        raise ValueError("请填写至少一个工序时间段，例如 00:10-00:20。")
    ranges: list[tuple[float, float]] = []
    for index, raw_range in enumerate(re.split(r"[;；,，\n]+", text), start=1):
        item = raw_range.strip()
        if not item:
            continue
        parts = re.split(r"\s*(?:-|~|～|至)\s*", item, maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"第 {index} 个工序时间段格式不正确：{item}。请使用 00:10-00:20。")
        start = parse_time(parts[0], f"第 {index} 个工序开始时间")
        end = parse_time(parts[1], f"第 {index} 个工序结束时间")
        if end <= start:
            raise ValueError(f"第 {index} 个工序时间段的结束时间必须晚于开始时间。")
        ranges.append((start, end))
    if not ranges:
        raise ValueError("请填写至少一个有效的工序时间段。")
    return ranges


def _unique_output_folder(parent: Path, stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / f"{stem}_{timestamp}"
    index = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = parent / f"{stem}_{timestamp}_{index}"
            index += 1


def video_frame_task_key(function_id: str, paths: dict[str, Path], parameters: dict[str, Any]) -> str:
    snapshot = {"paths": {key: str(path.resolve()) for key, path in paths.items()}, "parameters": parameters}
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return f"{function_id}:{fingerprint}"


def _filename_number(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _english_identifier(value: Any, label: str) -> str:
    identifier = re.sub(r"[_-]+", "_", str(value or "").strip().lower().replace(" ", "_"))
    if not identifier:
        raise ValueError(f"请填写{label}，例如 tcl 或 line2_station3。")
    if len(identifier) > 48 or not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)*", identifier):
        raise ValueError(f"{label}只能包含英文字母、数字、空格、下划线或短横线，且不能超过 48 个字符。")
    return identifier


def _sampling_filename_info(
    module: ModuleType,
    video: Path,
    target_fps: float | None,
    interval_sec: float | None,
    fallback_step: int,
) -> tuple[str, float]:
    capture = module.open_video(str(video))
    try:
        source_fps = float(capture.get(module.cv2.CAP_PROP_FPS) or 0.0)
    finally:
        capture.release()
    if source_fps > 0:
        if target_fps is not None:
            step_frames = max(1, int(round(source_fps / min(target_fps, source_fps))))
        elif interval_sec is not None:
            step_frames = max(1, int(round(source_fps * interval_sec)))
        else:
            step_frames = fallback_step
        actual_rate = f"fps{_filename_number(source_fps / step_frames)}"
        if interval_sec is not None:
            return f"every{_filename_number(interval_sec)}s_{actual_rate}", source_fps
        return actual_rate, source_fps
    if interval_sec is not None:
        return f"every{_filename_number(interval_sec)}s_step{fallback_step}", 0.0
    return f"step{fallback_step}", 0.0


def _ensure_informative_filenames(
    output_folder: Path,
    video: Path,
    english_prefix: str,
    frame_indices: list[int],
    sampling_tag: str,
    source_fps: float,
) -> None:
    """Rename legacy script output while remaining compatible with newer scripts."""
    legacy_stems = {video.stem, video.name.split(".")[0]}
    for frame_index in frame_indices:
        time_tag = (
            f"t{int(round(frame_index * 1000.0 / source_fps)):09d}ms"
            if source_fps > 0 else "tunknown"
        )
        destination = output_folder / f"{english_prefix}_{sampling_tag}_{time_tag}_f{frame_index:08d}.jpg"
        if destination.exists():
            continue
        source_sampling_tag = sampling_tag.rsplit("_", 1)[-1]
        candidates = [
            output_folder / f"{english_prefix}_{source_sampling_tag}_{time_tag}_f{frame_index:08d}.jpg",
            output_folder / f"{video.stem}_{sampling_tag}_{time_tag}_f{frame_index:08d}.jpg",
            *(output_folder / f"{stem}_{frame_index}.jpg" for stem in legacy_stems),
        ]
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is not None:
            source.replace(destination)


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
    context.check_cancelled()
    module = _load_source_module()
    videos = _resolve_videos(context, module)
    output_root = context.paths.get("output_folder")
    if output_root:
        output_root = output_root.expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

    project_code = _english_identifier(context.parameters.get("project_code"), "项目英文标识")
    scene_code = _english_identifier(context.parameters.get("scene_code"), "场景英文标识")

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
    priority_ranges: list[tuple[float, float]] = []
    priority_fps: float | None = None
    background_fps: float | None = None
    if mode == "工序时间段高密度":
        priority_ranges = _priority_time_ranges(context.parameters.get("priority_time_ranges"))
        priority_fps = _positive_number(context.parameters.get("priority_fps"), "工序时间段每秒抽取张数", 25)
        background_fps = _positive_number(context.parameters.get("background_fps"), "普通片段每秒抽取张数", 5)
        max_images = _positive_integer(context.parameters.get("priority_max_images"), "总图片数量上限", 500)
        target_fps = None
        interval_sec = None
        mode_message = f"工序时间段每秒 {priority_fps:g} 张，其余片段每秒 {background_fps:g} 张"
    elif mode == "按时间间隔":
        target_fps = None
        interval_sec = _positive_number(context.parameters.get("interval_sec"), "抽帧时间间隔", 1)
        mode_message = f"每 {interval_sec:g} 秒抽取 1 张"
    else:
        target_fps = _positive_number(context.parameters.get("target_fps"), "每秒抽取张数", 5)
        interval_sec = None
        mode_message = f"每秒抽取 {target_fps:g} 张"

    limit_description = f"每个视频最多 {max_images} 张" if max_images is not None else "不限制图片总数"
    context.report(f"已递归扫描输入文件夹，共找到 {len(videos)} 个 MP4 视频；抽帧方式：{mode_message}，{limit_description}")
    context.report(f"英文命名：项目 {project_code}；场景 {scene_code}")
    context.report(
        f"有效区间：{start_time_sec:g} 秒 至 "
        f"{f'{end_time_sec:g} 秒' if end_time_sec is not None else '视频结尾'}；"
        f"{f'超过上限时采用“{limit_strategy}”' if max_images is not None else '按抽帧密度和有效时长自动计算数量'}"
    )
    requested_workers = _non_negative_number(context.parameters.get("parallel_workers"), "并行视频数", 0)
    if not requested_workers.is_integer() or requested_workers > 16:
        raise ValueError("并行视频数应为 0 到 16 之间的整数；0 表示自动。")
    resources = resource_pool.status()
    if resources["maxWorkers"] < 1:
        raise ValueError(resources["warning"] or "当前内存不足，请释放资源后重试。")
    workers = min(len(videos), int(requested_workers) or resources["recommendedWorkers"], resources["maxWorkers"])
    if resources["warning"]:
        context.report(f"资源提示：{resources['warning']}")
    if requested_workers > workers and len(videos) > workers:
        context.report(f"资源提示：设置并行数 {int(requested_workers)}，本次自动限制为 {workers}，避免占用过多 CPU 或内存。")
    context.report(f"本批最多同时处理 {workers} 个视频；所有批次共享资源限额，资源忙时自动等待。")

    stop_event = context.stop_event or threading.Event()
    report_lock = threading.RLock()
    def report(message: str) -> None:
        with report_lock:
            context.report(message)

    worker_context = RunContext(context.function_id, context.paths, context.parameters, report, stop_event, context.notify)

    def extract(index: int, video: Path) -> dict[str, Any]:
        worker_context.check_cancelled()
        parent = output_root or video.parent
        if shutil.disk_usage(parent).free < 256 * MIB:
            raise RuntimeError(f"输出磁盘可用空间不足 256 MB：{parent}。请释放磁盘空间。")
        if mode == "工序时间段高密度":
            sampling_tag = f"priority{_filename_number(priority_fps or 25)}_bg{_filename_number(background_fps or 5)}"
            capture = module.open_video(str(video))
            try:
                source_fps = float(capture.get(module.cv2.CAP_PROP_FPS) or 0.0)
            finally:
                capture.release()
        else:
            sampling_tag, source_fps = _sampling_filename_info(
                module, video, target_fps, interval_sec, fallback_step,
            )
        english_prefix = f"{project_code}_{scene_code}_video{index:02d}"
        output_folder = _unique_output_folder(parent, f"{english_prefix}_{sampling_tag}")
        report(f"[{index}/{len(videos)}] 正在处理：{video}\n输出目录：{output_folder}")
        probe = module.open_video(str(video))
        try:
            width = max(0, int(probe.get(module.cv2.CAP_PROP_FRAME_WIDTH)))
            height = max(0, int(probe.get(module.cv2.CAP_PROP_FRAME_HEIGHT)))
        finally:
            probe.release()
        memory_estimate = max(DEFAULT_WORKER_MEMORY, width * height * 3 * 16 + 128 * MIB)
        last_progress = 0.0
        def progress(done: int, total: int) -> None:
            nonlocal last_progress
            now = time.monotonic()
            if done == total or now - last_progress >= 1.0:
                last_progress = now
                report(f"[{index}/{len(videos)}] {video.name}：{done}/{total} 帧（{done / max(1, total):.0%}）")
                if shutil.disk_usage(parent).free < 64 * MIB:
                    raise RuntimeError("输出磁盘即将写满，已停止该视频；已生成的图片会保留。")
        with resource_pool.acquire(worker_context, memory_estimate):
            process_result = module.vedio2img(
                vedio_path=str(video), images_save_path=str(output_folder),
                target_fps=target_fps, interval_sec=interval_sec, fallback_step=fallback_step,
                max_images=max_images, start_time_sec=start_time_sec, end_time_sec=end_time_sec,
                limit_strategy=limit_strategy, priority_ranges=priority_ranges,
                priority_fps=priority_fps or 25, background_fps=background_fps or 5,
                filename_prefix=english_prefix, check_cancelled=worker_context.check_cancelled,
                report_progress=progress,
            )

        selected_frame_indices = process_result.get("selected_frame_indices", []) if process_result else []
        _ensure_informative_filenames(
            output_folder, video, english_prefix, selected_frame_indices, sampling_tag, source_fps,
        )
        image_count = len(list(output_folder.glob("*.jpg"))) if output_folder.exists() else 0
        if image_count == 0:
            raise RuntimeError(f"{video.name} 没有成功生成图片，请确认视频可以正常播放。")
        limit_reached = bool(process_result and process_result.get("limit_applied"))
        detail = {
            "index": index,
            "video": str(video),
            "images": image_count,
            "maxImages": max_images,
            "limitReached": limit_reached,
            "limitStrategy": limit_strategy,
            "startTimeSec": process_result.get("start_time_sec", start_time_sec) if process_result else start_time_sec,
            "endTimeSec": process_result.get("end_time_sec", end_time_sec) if process_result else end_time_sec,
            "selectedFrameIndices": selected_frame_indices,
            "samplingTag": sampling_tag,
            "priorityRanges": priority_ranges,
            "priorityCandidateCount": process_result.get("priority_candidate_count", 0) if process_result else 0,
            "backgroundCandidateCount": process_result.get("background_candidate_count", 0) if process_result else 0,
            "projectCode": project_code,
            "sceneCode": scene_code,
            "englishPrefix": english_prefix,
            "output": str(output_folder),
        }
        if max_images is None:
            limit_message = "（按抽帧密度和有效时长完成）"
        elif limit_reached:
            limit_message = "（已达到上限）"
        else:
            limit_message = "（未达到设置上限）"
        report(
            f"[{index}/{len(videos)}] 已生成 {image_count} 张图片{limit_message}；"
            f"文件名抽帧标记：{sampling_tag}"
        )
        if process_result and process_result.get("error_count"):
            report(f"[{index}/{len(videos)}] 提示：{process_result['error_count']} 帧读取或保存失败，已保留成功图片。")
        return detail

    completed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="video-frames")
    iterator = iter(enumerate(videos, start=1))
    pending = {}
    def submit_next() -> None:
        item = next(iterator, None)
        if item is not None:
            pending[executor.submit(extract, *item)] = item
    try:
        for _ in range(workers):
            submit_next()
        while pending:
            context.check_cancelled()
            finished, _ = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
            for future in finished:
                index, video = pending.pop(future)
                try:
                    completed.append(future.result())
                except TaskCancelled:
                    raise
                except Exception as exc:
                    failures.append({"video": str(video), "error": str(exc)})
                    report(f"[{index}/{len(videos)}] 失败：{video}：{exc}")
                submit_next()
    except BaseException:
        stop_event.set()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    context.check_cancelled()
    if not completed:
        raise RuntimeError("所有视频均未成功抽帧：" + "；".join(f"{item['video']}：{item['error']}" for item in failures))
    details = sorted(completed, key=lambda item: item["index"])
    total_images = sum(item["images"] for item in details)
    output_folders = [item["output"] for item in details]
    message = f"视频切图完成：成功 {len(details)}/{len(videos)} 个视频，共生成 {total_images} 张图片。"
    if failures:
        message += f"失败 {len(failures)} 个，原因已在日志列出；成功结果已保留。"

    return {
        "message": message,
        "videoCount": len(videos),
        "imageCount": total_images,
        "maxImagesPerVideo": max_images,
        "projectCode": project_code,
        "sceneCode": scene_code,
        "outputFolders": output_folders,
        "details": details,
        "failedVideos": failures,
        "parallelWorkers": workers,
    }


register_handler(HANDLER_ID, run_video_frames)
