#!/usr/bin/env python3
"""Temporary remote worker that streams annotated STAR inference frames.

This file is uploaded to the inference device and imports the user's existing
``run_star_video.py``.  It deliberately contains no model-specific decoding
logic so the original script remains the single source of truth.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import importlib.util
import json
import os
import queue
import struct
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


@contextlib.contextmanager
def redirect_process_stdout_to_stderr():
    """Keep native TensorRT/CUDA diagnostics out of the binary frame stream.

    ``redirect_stdout`` only changes Python's ``sys.stdout``. TensorRT and CUDA
    extensions can still write straight to file descriptor 1, corrupting the
    length-prefixed JPEG protocol used by the desktop preview.
    """
    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()
    sys.stdout.flush()
    saved_stdout = os.dup(stdout_fd)
    try:
        os.dup2(stderr_fd, stdout_fd)
        yield
        try:
            ctypes.CDLL(None).fflush(None)
        except Exception:
            pass
        sys.stdout.flush()
    finally:
        os.dup2(saved_stdout, stdout_fd)
        os.close(saved_stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream annotated STAR inference frames")
    parser.add_argument("--script", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--models-json", default="")
    parser.add_argument("--source", required=True)
    parser.add_argument("--labels", default="class0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save", default="")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--preview-fps", type=float, default=20.0)
    parser.add_argument("--stream-width", type=int, default=960)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--replay-grace-seconds", type=int, default=120)
    parser.add_argument("--cuda-backend", default="auto")
    parser.add_argument("--session-token", required=True)
    return parser.parse_args()


def load_user_script(path_value: str):
    path = Path(path_value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Remote inference script does not exist: {path}")
    spec = importlib.util.spec_from_file_location("yolo_user_star_video", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import remote inference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_frame(stream, frame: np.ndarray, width: int, quality: int, position_seconds: float, generation: int = 0) -> None:
    preview = frame
    if width > 0 and frame.shape[1] > width:
        scale = width / frame.shape[1]
        preview = cv2.resize(
            frame,
            (width, max(1, round(frame.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(
        ".jpg",
        preview,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not ok:
        raise RuntimeError("Could not encode the annotated preview frame")
    payload = encoded.tobytes()
    stream.write(struct.pack(">IdI", generation, position_seconds, len(payload)))
    stream.write(payload)
    stream.flush()


class PreviewSender:
    """Keep only the newest preview so slow SSH clients cannot stall inference."""

    def __init__(self, stream, width: int, quality: int) -> None:
        self.stream = stream
        self.width = width
        self.quality = quality
        self.pending: queue.Queue[tuple[np.ndarray, float, int] | None] = queue.Queue(maxsize=1)
        self.error: Exception | None = None
        self.thread = threading.Thread(target=self._send, daemon=True)
        self.thread.start()

    def _send(self) -> None:
        try:
            while True:
                item = self.pending.get()
                if item is None:
                    return
                frame, position, generation = item
                write_frame(self.stream, frame, self.width, self.quality, position, generation)
        except Exception as exc:
            self.error = exc

    def submit(self, frame: np.ndarray, position: float, generation: int = 0) -> None:
        if self.error is not None:
            raise self.error
        try:
            self.pending.put_nowait((frame, position, generation))
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait((frame, position, generation))

    def discard_pending(self) -> None:
        try:
            self.pending.get_nowait()
        except queue.Empty:
            pass

    def close(self) -> None:
        try:
            self.pending.put(None, timeout=2)
        except queue.Full:
            try:
                self.pending.get_nowait()
            except queue.Empty:
                pass
            self.pending.put_nowait(None)
        self.thread.join(timeout=2)


def capture_frame_index(capture, *, after_read: bool = False) -> int:
    """Return a stable zero-based frame index from OpenCV's next-frame cursor."""
    value = capture.get(cv2.CAP_PROP_POS_FRAMES)
    if not np.isfinite(value) or value < 0:
        return 0
    index = int(round(value))
    return max(0, index - 1) if after_read else max(0, index)


def skip_late_video_frames(
    capture,
    source_fps: float,
    source_frames: float,
    anchor_source_seconds: float,
    anchor_wall_time: float,
) -> int:
    """Drop source frames that are already behind the real-time playhead.

    Inference can be slower than the video's FPS, especially with several
    models. Processing every source frame would turn the preview into slow
    motion. ``grab`` advances decoding without inference so the next frame
    stays aligned with elapsed wall-clock time.
    """
    if source_fps <= 0 or not np.isfinite(source_frames) or source_frames <= 0:
        return 0
    next_index = capture_frame_index(capture)
    elapsed = max(0.0, time.monotonic() - anchor_wall_time)
    target_index = min(
        max(0, int((anchor_source_seconds + elapsed) * source_fps)),
        max(0, int(source_frames) - 1),
    )
    skipped = 0
    while next_index < target_index:
        if not capture.grab():
            break
        next_index += 1
        skipped += 1
    return skipped


def video_position_seconds(capture, source_fps: float, frame_index: int) -> float:
    index_position = (frame_index + 1) / source_fps if source_fps > 0 else 0.0
    position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    tolerance = max(0.5, 2.0 / source_fps) if source_fps > 0 else 0.5
    if not np.isfinite(position) or position <= 0 or abs(position - index_position) > tolerance:
        position = index_position
    return max(0.0, float(position))


def write_timeline_frame(writer, frame: np.ndarray, source_index: int,
                         previous_frame: np.ndarray | None,
                         previous_index: int | None) -> tuple[np.ndarray, int, int]:
    """Keep a saved video's duration correct when real-time preview drops frames."""
    written = 0
    if previous_frame is not None and previous_index is not None:
        for _ in range(max(0, source_index - previous_index - 1)):
            writer.write(previous_frame)
            written += 1
    writer.write(frame)
    return frame.copy(), source_index, written + 1


def read_control_commands(seeks: queue.SimpleQueue[tuple[float, int]], paused: threading.Event) -> None:
    """Receive seek and playback controls without blocking the inference loop."""
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            action = payload.get("action")
            if action == "pause":
                paused.set()
            elif action == "resume":
                paused.clear()
            elif action == "seek":
                generation = int(payload["generation"])
                if 0 <= generation <= 0xFFFFFFFF:
                    seeks.put((float(payload["seconds"]), generation))
        except (ValueError, TypeError, KeyError, AttributeError):
            continue


def infer_and_draw_models(user_script, loaded_models, frame: np.ndarray, iou: float, max_det: int) -> int:
    """Run every model on the unannotated frame, then overlay all detections."""
    detections = []
    total = 0
    for runner, labels, model_conf in loaded_models:
        tensor, gain, pad = user_script.preprocess(frame, runner.input_shape)
        outputs = runner.infer(tensor)
        boxes, scores, class_ids = user_script.decode_yolo26(
            outputs=outputs,
            input_hw=(runner.input_shape[2], runner.input_shape[3]),
            original_hw=frame.shape[:2],
            gain=gain,
            pad=pad,
            conf_threshold=model_conf,
            iou_threshold=iou,
            max_det=max_det,
            num_classes=len(labels),
        )
        total += len(boxes)
        detections.append((boxes, scores, class_ids, labels))
    for boxes, scores, class_ids, display_labels in detections:
        user_script.draw_detections(frame, boxes, scores, class_ids, display_labels)
    return total


def main() -> int:
    args = parse_args()
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")
    if args.max_det <= 0 or args.preview_fps < 0:
        raise ValueError("--max-det must be positive and --preview-fps cannot be negative")
    if not 0 <= args.replay_grace_seconds <= 3600:
        raise ValueError("--replay-grace-seconds must be between 0 and 3600")

    frame_stream = sys.stdout.buffer
    if args.models_json:
        model_specs = json.loads(args.models_json)
        if not isinstance(model_specs, list) or not 1 <= len(model_specs) <= 8:
            raise ValueError("--models-json must contain 1 to 8 models")
    else:
        model_specs = [{"path": args.model, "labels": args.labels, "conf": args.conf}]
    loaded_models = []
    # Imported TensorRT helpers print useful diagnostics. Keep stdout binary-only
    # for the length-prefixed JPEG protocol and route all text to SSH stderr.
    try:
        with redirect_process_stdout_to_stderr(), contextlib.redirect_stdout(sys.stderr):
            user_script = load_user_script(args.script)
            for index, spec in enumerate(model_specs, 1):
                if not isinstance(spec, dict):
                    raise ValueError(f"Model {index} configuration is invalid")
                path = str(spec["path"])
                labels = [label.strip() for label in str(spec["labels"]).split(",") if label.strip()]
                conf = float(spec.get("conf", args.conf))
                if not labels or not 0 <= conf <= 1:
                    raise ValueError(f"Model {index} labels or confidence is invalid")
                engine_bytes, offset = user_script.read_engine_bytes(path)
                print(f"Loaded model {index}/{len(model_specs)} ({', '.join(labels)}): TensorRT offset={offset}, "
                      f"size={len(engine_bytes) / 1024 / 1024:.2f} MiB", file=sys.stderr, flush=True)
                runner = user_script.TensorRTRunner(engine_bytes, cuda_backend=args.cuda_backend)
                loaded_models.append((runner, labels, conf))
            capture = user_script.open_source(args.source)
    except Exception:
        for runner, _labels, _conf in loaded_models:
            runner.close()
        raise

    writer = None
    saved_video_paths = []
    save_segment = 1
    frame_index = 0
    fps_ema = 0.0
    skipped_frames = 0
    last_preview_at = 0.0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_period = 1.0 / source_fps if np.isfinite(source_fps) and source_fps > 0 else 0.0
    preview_period = 1.0 / args.preview_fps if args.preview_fps > 0 else 0.0
    source_frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = source_frames / source_fps if np.isfinite(source_frames) and source_frames > 0 and frame_period > 0 else 0.0
    print("VIDEO_META_JSON:" + json.dumps({"durationSeconds": duration, "sourceFps": source_fps if frame_period > 0 else 0.0}), file=sys.stderr, flush=True)
    seeks: queue.SimpleQueue[tuple[float, int]] = queue.SimpleQueue()
    paused = threading.Event()
    threading.Thread(target=read_control_commands, args=(seeks, paused), daemon=True).start()
    preview_sender = PreviewSender(frame_stream, args.stream_width, args.jpeg_quality)
    generation = 0
    playback_anchor_source = capture_frame_index(capture) / source_fps if frame_period > 0 else 0.0
    playback_anchor_wall = time.monotonic()
    playback_clock_needs_reset = False
    previous_saved_frame = None
    previous_saved_index = None
    saved_frame_count = 0

    if args.realtime and frame_period > 0:
        print("Real-time playback enabled: slow inference will skip late source frames to preserve video speed.", file=sys.stderr, flush=True)

    try:
        while True:
            seek_target = None
            while not seeks.empty():
                seek_target = seeks.get_nowait()
            if seek_target is not None and duration > 0:
                requested_seconds, generation = seek_target
                target_frame = min(max(0, round(requested_seconds * source_fps)), max(0, int(source_frames) - 1))
                preview_sender.discard_pending()
                if not capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame):
                    print("WARNING: video decoder could not seek to the requested position", file=sys.stderr, flush=True)
                if writer is not None:
                    writer.release()
                    writer = None
                    save_segment += 1
                    print(f"Seek started a new saved-video segment #{save_segment}; previous segment was kept.", file=sys.stderr, flush=True)
                previous_saved_frame = None
                previous_saved_index = None
                last_preview_at = 0.0
                playback_anchor_source = target_frame / source_fps
                playback_anchor_wall = time.monotonic()
                playback_clock_needs_reset = False
            if paused.is_set() and seek_target is None:
                playback_clock_needs_reset = True
                time.sleep(0.02)
                continue
            if args.realtime and frame_period > 0:
                if playback_clock_needs_reset:
                    playback_anchor_source = capture_frame_index(capture) / source_fps
                    playback_anchor_wall = time.monotonic()
                    playback_clock_needs_reset = False
                skipped_frames += skip_late_video_frames(
                    capture, source_fps, source_frames,
                    playback_anchor_source, playback_anchor_wall,
                )
            ok, frame = capture.read()
            if not ok:
                if writer is not None:
                    writer.release()
                    writer = None
                    save_segment += 1
                previous_saved_frame = None
                previous_saved_index = None
                if duration <= 0 or args.replay_grace_seconds == 0:
                    break
                print("VIDEO_EOF_JSON:" + json.dumps({"graceSeconds": args.replay_grace_seconds}), file=sys.stderr, flush=True)
                deadline = time.monotonic() + args.replay_grace_seconds
                while time.monotonic() < deadline and seeks.empty():
                    time.sleep(0.05)
                if seeks.empty():
                    break
                continue
            source_index = capture_frame_index(capture, after_read=True)
            source_position = video_position_seconds(capture, source_fps, source_index)
            started = time.perf_counter()
            total_detections = infer_and_draw_models(user_script, loaded_models, frame, args.iou, args.max_det)
            elapsed = max(time.perf_counter() - started, 1e-9)
            current_fps = 1.0 / elapsed
            fps_ema = current_fps if frame_index == 0 else 0.9 * fps_ema + 0.1 * current_fps
            frame_index += 1

            cv2.putText(
                frame,
                f"Infer {fps_ema:.1f} FPS  models {len(loaded_models)}  skipped {skipped_frames}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save:
                if writer is None:
                    save_path = Path(args.save)
                    if save_segment > 1:
                        if save_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
                            save_path = save_path.with_name(f"{save_path.stem}_part{save_segment}{save_path.suffix}")
                        else:
                            save_path = save_path / f"result_part{save_segment}.mp4"
                    writer, saved_video_path = user_script.create_writer(str(save_path), capture, frame)
                    saved_video_paths.append(saved_video_path)
                previous_saved_frame, previous_saved_index, written = write_timeline_frame(
                    writer, frame, source_index, previous_saved_frame, previous_saved_index,
                )
                saved_frame_count += written

            if args.realtime and frame_period > 0:
                deadline = playback_anchor_wall + max(0.0, source_position - playback_anchor_source)
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)

            now = time.monotonic()
            if preview_period == 0 or now - last_preview_at >= preview_period:
                preview_sender.submit(frame, min(max(0.0, source_position), duration) if duration > 0 else max(0.0, source_position), generation)
                last_preview_at = now
    finally:
        preview_sender.close()
        capture.release()
        if writer is not None:
            writer.release()
        for runner, _labels, _conf in loaded_models:
            runner.close()

    print(f"Processed {frame_index} frames; skipped {skipped_frames} late source frames", file=sys.stderr, flush=True)
    if saved_frame_count:
        print(f"Saved timeline frames: {saved_frame_count}", file=sys.stderr, flush=True)
    for saved_video_path in saved_video_paths:
        print(f"Saved annotated video: {saved_video_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except BrokenPipeError:
        raise SystemExit(0)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
