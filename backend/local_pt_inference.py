"""Local PT / ONNX detection with the same live preview protocol as SSH inference."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import importlib.util
import io
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from .remote_inference import RemoteInferenceSession, TERMINAL_STATES
from .handlers import RunContext, register_handler
from .local_pt_worker import receive as receive_worker_packet, send as send_worker_packet
from .local_pt_worker import parse_label_text
from .worker_runtime import worker_command
from .local_image_dataset import collect_images, read_image, save_image, validate_image_paths
from .remote_star_worker import (
    capture_frame_index,
    seek_video_frame,
    skip_late_video_frames,
    video_position_seconds,
    write_timeline_frame,
)


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".ts", ".mts"}
WORKER_SCRIPT = Path(__file__).with_name("local_pt_worker.py")


def validate_model_path(value: Any) -> Path:
    model = Path(str(value or "").strip()).expanduser()
    if model.suffix.lower() not in {".pt", ".onnx"} or not model.is_file():
        raise ValueError("请选择存在的本地 .pt 或 .onnx 模型文件。")
    return model.resolve()


def detect_local_model_labels(model_file: str) -> dict[str, Any]:
    model = validate_model_path(model_file)
    try:
        result = subprocess.run(worker_command(_find_yolo_python(), WORKER_SCRIPT, [str(model), "--inspect"]),
                                capture_output=True, timeout=90, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("读取模型类别超时（90 秒），请检查模型和 AI Python 环境。") from exc
    try:
        header, _ = receive_worker_packet(io.BytesIO(result.stdout))
    except (EOFError, ValueError, OSError) as exc:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[-1500:]
        raise ValueError(f"无法读取模型类别：{detail or 'AI 工作进程没有返回有效信息'}") from exc
    if result.returncode != 0 or header.get("type") != "metadata":
        raise ValueError(f"读取模型类别失败：{header.get('message', 'AI 工作进程异常')}")
    return {key: value for key, value in header.items() if key != "type"}


def _find_yolo_python() -> Path:
    from .app_settings import settings_store
    override = os.environ.get("PROCESSING_VIEW_YOLO_PYTHON", "").strip() or settings_store.payload().get("aiPython", "")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return candidate.resolve()
        raise RuntimeError(f"指定的 YOLO Python 不存在：{candidate}")

    # Full AI builds and ordinary venv/pip installations need no named conda env.
    if all(importlib.util.find_spec(name) is not None for name in ("ultralytics", "torch")):
        return Path(sys.executable).resolve()
    roots = [Path(sys.executable).resolve().parent.parent]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        roots.append(Path(conda_prefix))
    conda_command = shutil.which("conda")
    if conda_command:
        roots.append(Path(conda_command).resolve().parent.parent)
    roots.extend((Path.home() / "anaconda3", Path.home() / "miniconda3"))
    for root in roots:
        for relative in ("envs/yolo26/bin/python", "envs/yolo26/python.exe"):
            candidate = root / relative
            if candidate.is_file():
                return candidate.resolve()
    raise RuntimeError(
        "当前版本未内置可用的 Ultralytics/PyTorch。请使用 AI 集成版，"
        "或在本机设置中选择任意已安装这些库的 Python；不要求使用 conda 或固定环境名。"
    )


class _AnnotatedResult:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def plot(self) -> np.ndarray:
        return self.frame


class _SubprocessYoloModel:
    def __init__(self, python_path: Path, model_path: str, session: "LocalPTSession",
                 worker_script: Path = WORKER_SCRIPT) -> None:
        self._recent_errors: deque[str] = deque(maxlen=12)
        self._process = subprocess.Popen(
            worker_command(python_path, worker_script, [model_path]),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            bufsize=0,
        )
        session.worker = self._process
        threading.Thread(target=self._collect_errors, name=f"local-pt-stderr-{session.id}", daemon=True).start()
        try:
            header, _payload = self._receive()
            if header.get("type") != "ready":
                raise RuntimeError("YOLO 子进程启动响应异常。")
            self.names = header.get("names", {})
            metadata = header.get("metadata", {})
            if metadata.get("message"):
                session.add_log(metadata["message"])
        except Exception:
            self.close()
            raise

    def _collect_errors(self) -> None:
        if self._process.stderr is None:
            return
        for raw in self._process.stderr:
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._recent_errors.append(line)

    def _receive(self) -> tuple[dict[str, Any], bytes]:
        if self._process.stdout is None:
            raise RuntimeError("YOLO 子进程没有输出通道。")
        try:
            header, payload = receive_worker_packet(self._process.stdout)
        except (EOFError, OSError, ValueError) as exc:
            detail = self._recent_errors[-1] if self._recent_errors else "子进程已退出或通信中断"
            raise RuntimeError(f"YOLO 子进程异常：{detail}") from exc
        if header.get("type") == "error":
            raise RuntimeError(f"YOLO 推理失败：{header.get('message', '未知错误')}")
        return header, payload

    def predict(self, frame: np.ndarray, **options: Any) -> list[_AnnotatedResult]:
        if self._process.stdin is None:
            raise RuntimeError("YOLO 子进程没有输入通道。")
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        try:
            send_worker_packet(self._process.stdin, {
                "type": "frame", "shape": list(frame.shape), "options": options,
            }, frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            detail = self._recent_errors[-1] if self._recent_errors else "子进程已退出"
            raise RuntimeError(f"无法向 YOLO 子进程发送视频帧：{detail}") from exc
        header, payload = self._receive()
        shape = header.get("shape")
        if (header.get("type") != "frame" or not isinstance(shape, list) or len(shape) != 3 or
                not all(isinstance(value, int) and value > 0 for value in shape) or
                shape[2] != 3 or shape[0] * shape[1] * shape[2] != len(payload)):
            raise RuntimeError("YOLO 子进程返回的视频帧格式异常。")
        return [_AnnotatedResult(np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy())]

    def close(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.terminate()
            except ProcessLookupError:
                pass
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
                self._process.wait(timeout=3)
        for stream in (self._process.stdin, self._process.stdout, self._process.stderr):
            if stream is not None:
                stream.close()


def _number(raw: Any, name: str, minimum: float, maximum: float) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是数字。") from exc
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name}应在 {minimum:g} 到 {maximum:g} 之间。")
    return value


def validate_local_payload(raw: dict[str, Any]) -> dict[str, Any]:
    paths = raw.get("paths", {})
    params = raw.get("parameters", {})
    if not isinstance(paths, dict) or not isinstance(params, dict):
        raise ValueError("本地推理参数格式不正确。")
    model = validate_model_path(paths.get("model_file", ""))
    input_mode = str(params.get("input_mode", "视频检测")).strip() or "视频检测"
    if input_mode not in {"视频检测", "图片文件夹检测"}:
        raise ValueError("输入类型请选择视频检测或图片文件夹检测。")
    output_text = str(paths.get("output_folder", "")).strip()
    video = None
    image_folder = None
    if input_mode == "图片文件夹检测":
        image_folder, output_folder = validate_image_paths(str(paths.get("image_folder", "")), output_text)
    else:
        video = Path(str(paths.get("video_file", "")).strip()).expanduser()
        if video.suffix.lower() not in VIDEO_SUFFIXES or not video.is_file():
            raise ValueError("请选择存在的本地视频文件。")
        output_folder = Path(output_text).expanduser() if output_text else None
        if output_folder is not None and not output_folder.is_dir():
            raise ValueError("带框视频输出文件夹不存在。")
    output = output_folder / f"{video.stem}_detected.mp4" if output_folder is not None and video is not None else None
    if output is not None:
        if output.exists():
            raise ValueError(f"输出视频已存在：{output}。请更换输出文件夹或移走旧文件，以免覆盖。")
        if output.resolve() == video.resolve():
            raise ValueError("输出视频不能覆盖原视频。")
    device = str(params.get("device", "auto")).strip() or "auto"
    if device != "auto" and device != "cpu" and not device.replace(",", "").isdigit():
        raise ValueError("推理设备请填写 auto、cpu 或 GPU 编号（如 0）。")
    label_owner = str(params.get("labels_model_file", "")).strip()
    # Labels copied from model A must never be reused for a newly selected model B.
    labels = parse_label_text(str(params.get("labels", ""))) if not label_owner or Path(label_owner).expanduser().resolve() == model else []
    return {
        "model": model.resolve(), "video": video.resolve() if video else None, "output": output.resolve() if output else None,
        "input_mode": "images" if image_folder else "video", "image_folder": image_folder,
        "output_folder": output_folder.resolve() if output_folder else None,
        "recursive_images": bool(params.get("recursive_images", True)),
        "labels": labels,
        "conf": _number(params.get("conf", 0.5), "置信度阈值", 0, 1),
        "iou": _number(params.get("iou", 0.45), "NMS IoU 阈值", 0, 1),
        "max_det": int(_number(params.get("max_det", 300), "每帧最大检测框数", 1, 10000)),
        "device": device, "realtime": bool(params.get("realtime", True)),
        "preview_fps": _number(params.get("preview_fps", 20), "预览帧率", 0, 60),
        "stream_width": int(_number(params.get("stream_width", 960), "预览宽度", 320, 3840)),
        "jpeg_quality": int(_number(params.get("jpeg_quality", 70), "预览画质", 20, 100)),
        "replay_grace_seconds": int(_number(params.get("replay_grace_seconds", 120), "结束后回看时间", 0, 3600)),
    }


class LocalPTSession(RemoteInferenceSession):
    def __init__(self, session_id: str) -> None:
        super().__init__(id=session_id, host="本机", username="", status="starting", message="正在加载本地 PT / ONNX 模型……", stream_prefix="/api/local-pt-inference")
        self.paused = False
        self.seek_target: float | None = None
        self.seek_frame: int | None = None
        self.worker: subprocess.Popen[bytes] | None = None
        self.input_mode = "video"
        self.total_images = 0
        self.processed_images = 0
        self.saved_images = 0
        self.failed_images = 0
        self.current_image = ""
        self.output_folder = ""

    def snapshot(self) -> dict[str, Any]:
        with self.condition:
            return {**super().snapshot(), "inputMode": self.input_mode, "totalImages": self.total_images,
                    "processedImages": self.processed_images, "savedImages": self.saved_images,
                    "failedImages": self.failed_images, "currentImage": self.current_image,
                    "outputFolder": self.output_folder,
                    "result": {"outputFolders": [self.output_folder]} if self.input_mode == "images" and self.saved_images else {},
                    "progressPercent": round(self.processed_images * 100 / self.total_images) if self.total_images else 0}


class LocalPTInferenceManager:
    def __init__(self, model_factory: Callable[[str], Any] | None = None) -> None:
        self._sessions: dict[str, LocalPTSession] = {}
        self._lock = threading.RLock()
        self._model_factory = model_factory

    def get(self, session_id: str) -> LocalPTSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise ValueError("本地 PT 推理任务不存在或已经失效。")
        return session

    def start(self, raw: dict[str, Any], on_finished: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
        config = validate_local_payload(raw)
        session = LocalPTSession(uuid.uuid4().hex[:16])
        session.input_mode = config["input_mode"]
        if session.input_mode == "images":
            session.message = "正在扫描图片文件夹……"
            session.output_folder = str(config["output_folder"])
        with self._lock:
            self._sessions[session.id] = session
        threading.Thread(target=self._run, args=(session, config, on_finished), name=f"local-pt-{session.id}", daemon=True).start()
        return session.snapshot()

    def stop(self, session_id: str) -> dict[str, Any]:
        session = self.get(session_id)
        with session.condition:
            session.stop_event.set()
            if session.status not in TERMINAL_STATES:
                session.status = "stopping"
            session.message = "正在停止本地推理……"
            session.condition.notify_all()
            worker = session.worker
        if worker is not None and worker.poll() is None:
            try:
                worker.terminate()
            except ProcessLookupError:
                pass
        return session.snapshot()

    def stop_all(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.stop(session_id)

    def set_paused(self, session_id: str, paused: bool) -> dict[str, Any]:
        session = self.get(session_id)
        if session.input_mode == "images":
            raise ValueError("图片批量检测不支持视频播放控制，可使用停止检测按钮终止。")
        if not isinstance(paused, bool):
            raise ValueError("暂停状态必须是布尔值。")
        with session.condition:
            if session.status not in {"running", "paused"} or session.at_end:
                raise ValueError("当前视频不能暂停或继续，请确认视频正在播放。")
            session.paused = paused
            session.status = "paused" if paused else "running"
            session.message = "视频已暂停。" if paused else "已继续从当前位置推理。"
            session.condition.notify_all()
        return session.snapshot()

    def seek(self, session_id: str, seconds: Any) -> dict[str, Any]:
        session = self.get(session_id)
        target = _number(seconds, "跳转时间", 0, 10**9)
        with session.condition:
            if session.status not in {"running", "paused"} or session.duration_seconds <= 0:
                raise ValueError("当前视频不能跳转。")
            target = min(target, max(0, session.duration_seconds - 0.001))
            session.seek_target = target
            session.seek_frame = None
            session.seek_generation += 1
            session.pause_preview_pending = session.paused
            session.at_end = False
            session.replay_deadline = None
            session.condition.notify_all()
        return {"ok": True, "requestedSeconds": target}

    def step_frame(self, session_id: str, direction: Any) -> dict[str, Any]:
        session = self.get(session_id)
        with session.condition:
            target = session.frame_step_target(direction)
            session.paused = True
            session.status = "paused"
            session.seek_target = None
            session.seek_frame = target
            session.seek_generation += 1
            generation = session.seek_generation
            session.pause_preview_pending = True
            session.frame_step_pending = True
            session.at_end = False
            session.replay_deadline = None
            session.message = f"逐帧查看：第 {target + 1} 帧（已暂停）。"
            session.condition.notify_all()
            session.condition.wait_for(lambda: session.last_frame_generation >= generation or session.status in TERMINAL_STATES, timeout=2.0)
        return session.snapshot()

    def _run(self, session: LocalPTSession, config: dict[str, Any], on_finished: Callable[[dict[str, Any]], None] | None) -> None:
        capture = None
        writer = None
        model: Any = None
        output_created = False
        try:
            images = []
            if config["input_mode"] == "images":
                images = collect_images(config["image_folder"], config["output_folder"], config["recursive_images"], session.stop_event)
                if session.stop_event.is_set():
                    self._finish_images(session)
                    return
                if not images:
                    raise ValueError("图片文件夹中未找到支持的图片（JPG、JPEG、PNG、BMP、WEBP、TIF、TIFF）。请检查目录或开启包含子文件夹。")
                with session.condition:
                    session.total_images = len(images)
                    session.message = f"找到 {len(images)} 张图片，正在加载模型……"
                session.add_log(f"图片目录：{config['image_folder']}；共 {len(images)} 张，输出目录：{config['output_folder']}")
            if self._model_factory is None:
                python_path = _find_yolo_python()
                session.add_log(f"AI 运行环境：{python_path}")
                model = _SubprocessYoloModel(python_path, str(config["model"]), session)
            else:
                model = self._model_factory(str(config["model"]))
            if config["input_mode"] == "images":
                self._run_images(session, config, model, images)
                return
            capture = cv2.VideoCapture(str(config["video"]))
            if not capture.isOpened():
                raise RuntimeError("无法打开本地视频，请检查视频格式和文件权限。")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if math.isfinite(fps) and fps > 0 and math.isfinite(total_frames) and total_frames > 0 else 0.0
            with session.condition:
                session.source_fps = fps if math.isfinite(fps) and fps > 0 else 0
                session.duration_seconds = duration
                session.source_frame_count = int(total_frames) if math.isfinite(total_frames) and total_frames > 0 else 0
                session.condition.notify_all()
            session.add_log(f"本地模型：{config['model'].name}；视频：{config['video'].name}")
            session.add_log(f"类别：{getattr(model, 'names', '由权重文件提供')}")
            if config["labels"]:
                session.add_log(f"本次标签采用填写的类别（按 ID 顺序）：{dict(enumerate(config['labels']))}")
            if config["model"].suffix.lower() == ".onnx":
                session.add_log("ONNX 使用 ONNX Runtime；auto 默认 CPU，指定 GPU 时要求已安装可用的 CUDA Execution Provider。")
            if config["realtime"] and fps > 0:
                session.add_log("实时播放已开启：推理速度不足时会跳过落后的源帧，避免视频变成慢动作。")
            last_preview = 0.0
            saved = 0
            skipped_frames = 0
            processed_frames = 0
            inference_fps_ema = 0.0
            playback_anchor_source = capture_frame_index(capture) / fps if fps > 0 else 0.0
            playback_anchor_wall = time.monotonic()
            playback_clock_needs_reset = False
            previous_saved_frame = None
            previous_saved_index = None
            while not session.stop_event.is_set():
                with session.condition:
                    while session.paused and session.seek_target is None and session.seek_frame is None and not session.stop_event.is_set():
                        playback_clock_needs_reset = True
                        session.condition.wait(0.1)
                    if session.stop_event.is_set():
                        break
                    seek_target = session.seek_target
                    frame_target = session.seek_frame
                    generation = session.seek_generation
                    session.seek_target = None
                    session.seek_frame = None
                jumping = seek_target is not None or frame_target is not None
                if jumping:
                    if frame_target is not None:
                        if not seek_video_frame(capture, frame_target):
                            raise RuntimeError(f"视频解码器无法定位到第 {frame_target + 1} 帧。")
                    else:
                        capture.set(cv2.CAP_PROP_POS_MSEC, seek_target * 1000)
                    last_preview = 0
                    playback_anchor_source = capture_frame_index(capture) / fps if fps > 0 else seek_target
                    playback_anchor_wall = time.monotonic()
                    playback_clock_needs_reset = False
                    if writer is not None:
                        writer.release()
                        writer = None
                        previous_saved_frame = None
                        previous_saved_index = None
                        session.add_log("已跳转：停止向原保存文件写入，避免拼接不连续片段。")
                if config["realtime"] and fps > 0 and not jumping and not session.paused:
                    if playback_clock_needs_reset:
                        playback_anchor_source = capture_frame_index(capture) / fps
                        playback_anchor_wall = time.monotonic()
                        playback_clock_needs_reset = False
                    skipped_frames += skip_late_video_frames(
                        capture, fps, total_frames,
                        playback_anchor_source, playback_anchor_wall,
                    )
                ok, frame = capture.read()
                if not ok:
                    if duration <= 0 or config["replay_grace_seconds"] == 0:
                        break
                    with session.condition:
                        session.at_end = True
                        session.position_seconds = duration
                        session.replay_deadline = time.monotonic() + config["replay_grace_seconds"]
                        session.message = "视频已到结尾，可拖动进度条回看。"
                        while session.seek_target is None and session.seek_frame is None and not session.stop_event.is_set() and session.replay_deadline is not None and time.monotonic() < session.replay_deadline:
                            session.condition.wait(0.1)
                        if session.seek_target is None and session.seek_frame is None:
                            break
                    continue
                source_index = capture_frame_index(capture, after_read=True)
                position = source_index / fps if frame_target is not None and fps > 0 else video_position_seconds(capture, fps, source_index)
                inference_started = time.perf_counter()
                annotated = self._predict_frame(model, config, frame)
                inference_elapsed = max(time.perf_counter() - inference_started, 1e-9)
                inference_fps = 1.0 / inference_elapsed
                inference_fps_ema = inference_fps if processed_frames == 0 else 0.9 * inference_fps_ema + 0.1 * inference_fps
                processed_frames += 1
                cv2.putText(
                    annotated,
                    f"Infer {inference_fps_ema:.1f} FPS  skipped {skipped_frames}",
                    (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2, cv2.LINE_AA,
                )
                if frame_target is None and config["output"] is not None and (writer is not None or not output_created):
                    if writer is None:
                        height, width = annotated.shape[:2]
                        writer = cv2.VideoWriter(str(config["output"]), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 25, (width, height))
                        if not writer.isOpened():
                            raise RuntimeError("无法创建带框视频，请检查输出路径和编码器。")
                        output_created = True
                    previous_saved_frame, previous_saved_index, written = write_timeline_frame(
                        writer, annotated, source_index,
                        previous_saved_frame, previous_saved_index,
                    )
                    saved += written
                if config["realtime"] and fps > 0 and not session.paused:
                    deadline = playback_anchor_wall + max(0.0, position - playback_anchor_source)
                    delay = deadline - time.monotonic()
                    if delay > 0 and session.stop_event.wait(delay):
                        break
                now = time.monotonic()
                preview_due = config["preview_fps"] == 0 or now - last_preview >= 1 / config["preview_fps"]
                if preview_due or jumping:
                    height, width = annotated.shape[:2]
                    if width > config["stream_width"]:
                        annotated = cv2.resize(annotated, (config["stream_width"], round(height * config["stream_width"] / width)))
                    success, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]])
                    if not success and jumping:
                        raise RuntimeError("目标帧画面编码失败，请检查视频是否损坏。")
                    if success:
                        session.publish_frame(encoded.tobytes(), position, generation, source_index)
                        last_preview = now
            with session.condition:
                session.status = "stopped" if session.stop_event.is_set() else "completed"
                session.message = "本地推理已停止。" if session.stop_event.is_set() else f"视频检测完成，共预览 {session.frame_count} 帧。"
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.at_end = False
                session.condition.notify_all()
            if saved and config["output"] is not None:
                session.add_log(f"已保存 {saved} 帧带框视频：{config['output']}")
            if skipped_frames:
                session.add_log(f"为保持正常播放速度，本次跳过了 {skipped_frames} 个落后的源视频帧。")
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            with session.condition:
                session.status = "stopped" if session.stop_event.is_set() else "failed"
                session.message = "本地推理已停止。" if session.stop_event.is_set() else "本地 PT / ONNX 推理失败。"
                session.error = None if session.stop_event.is_set() else message
                session.finished_at = datetime.now().isoformat(timespec="seconds")
                session.logs.append(f"错误：{message}")
                session.condition.notify_all()
        finally:
            if isinstance(model, _SubprocessYoloModel):
                model.close()
                session.worker = None
            if writer is not None:
                writer.release()
            if capture is not None:
                capture.release()
            if on_finished is not None:
                try:
                    on_finished(session.snapshot())
                except Exception:
                    pass

    @staticmethod
    def _predict_frame(model: Any, config: dict[str, Any], frame: np.ndarray) -> np.ndarray:
        options = {"conf": config["conf"], "iou": config["iou"], "max_det": config["max_det"], "verbose": False}
        if isinstance(model, _SubprocessYoloModel):
            options["_class_names"] = config["labels"]
        if config["device"] != "auto":
            options["device"] = config["device"]
        return model.predict(frame, **options)[0].plot()

    @staticmethod
    def _finish_images(session: LocalPTSession) -> None:
        with session.condition:
            session.status = "stopped" if session.stop_event.is_set() else "completed" if session.saved_images else "failed"
            summary = f"已处理 {session.processed_images}/{session.total_images} 张，保存 {session.saved_images} 张，读取失败 {session.failed_images} 张。"
            session.message = ("图片检测已停止。" if session.stop_event.is_set() else "图片检测完成。") + summary
            if session.status == "failed":
                session.error = "没有成功保存任何检测图片，请查看读取失败日志。"
            session.finished_at = datetime.now().isoformat(timespec="seconds")
            session.logs.append(session.message)
            session.condition.notify_all()

    def _run_images(self, session: LocalPTSession, config: dict[str, Any], model: Any, images: list[Path]) -> None:
        output: Path = config["output_folder"]
        output.mkdir(parents=True, exist_ok=True)
        session.add_log(f"本地模型：{config['model'].name}；类别：{config['labels'] or getattr(model, 'names', {})}")
        session.add_log("所有图片逐张检测并保存带框图片（无目标也保存），不修改原图；结果保留子目录结构，重名自动加后缀。")
        last_preview = 0.0
        with session.condition:
            session.status = "stopping" if session.stop_event.is_set() else "running"
        for index, path in enumerate(images):
            if session.stop_event.is_set():
                break
            relative = path.relative_to(config["image_folder"])
            with session.condition:
                session.current_image = str(relative)
                session.message = f"正在检测图片 {index + 1}/{len(images)}：{relative}"
                session.condition.notify_all()
            try:
                frame = read_image(path)
            except (OSError, ValueError, cv2.error) as exc:
                with session.condition:
                    session.failed_images += 1
                    session.processed_images += 1
                session.add_log(f"[{index + 1}/{len(images)}] 跳过无法读取的图片 {relative}：{exc}")
                continue
            annotated = self._predict_frame(model, config, frame)
            if session.stop_event.is_set():
                break
            try:
                saved = save_image(output / relative, annotated, output)
            except (OSError, RuntimeError, cv2.error) as exc:
                raise RuntimeError(f"保存带框图片失败：{relative}；{exc}。已保存结果予以保留。") from exc
            with session.condition:
                session.saved_images += 1
                session.processed_images += 1
            session.add_log(f"[{index + 1}/{len(images)}] 已保存：{saved.relative_to(output)}")
            now = time.monotonic()
            if config["preview_fps"] == 0 or now - last_preview >= 1 / config["preview_fps"] or index == len(images) - 1:
                height, width = annotated.shape[:2]
                preview = cv2.resize(annotated, (config["stream_width"], round(height * config["stream_width"] / width))) if width > config["stream_width"] else annotated
                ok, encoded = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, config["jpeg_quality"]])
                if ok:
                    session.publish_frame(encoded.tobytes())
                    last_preview = now
        self._finish_images(session)


manager = LocalPTInferenceManager()


def _run_from_panel(_context: RunContext) -> dict[str, Any]:
    raise ValueError("该功能请使用“开始本地检测”按钮。")


register_handler("local.pt_inference", _run_from_panel)
