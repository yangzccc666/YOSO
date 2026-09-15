#!/usr/bin/env python3
"""Temporary remote worker that streams annotated STAR inference frames.

This file is uploaded to the inference device and imports the user's existing
``run_star_video.py``.  It deliberately contains no model-specific decoding
logic so the original script remains the single source of truth.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import struct
import sys
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream annotated STAR inference frames")
    parser.add_argument("--script", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--labels", default="class0")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--save", default="")
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--preview-fps", type=float, default=12.0)
    parser.add_argument("--stream-width", type=int, default=1280)
    parser.add_argument("--jpeg-quality", type=int, default=80)
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


def write_frame(stream, frame: np.ndarray, width: int, quality: int) -> None:
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
    stream.write(struct.pack(">I", len(payload)))
    stream.write(payload)
    stream.flush()


def main() -> int:
    args = parse_args()
    if not 0 <= args.jpeg_quality <= 100:
        raise ValueError("--jpeg-quality must be between 0 and 100")
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")
    if args.max_det <= 0 or args.preview_fps < 0:
        raise ValueError("--max-det must be positive and --preview-fps cannot be negative")

    frame_stream = sys.stdout.buffer
    # Imported TensorRT helpers print useful diagnostics. Keep stdout binary-only
    # for the length-prefixed JPEG protocol and route all text to SSH stderr.
    with contextlib.redirect_stdout(sys.stderr):
        user_script = load_user_script(args.script)
        engine_bytes, offset = user_script.read_engine_bytes(args.model)
        print(
            f"Loaded TensorRT engine: offset={offset}, "
            f"size={len(engine_bytes) / 1024 / 1024:.2f} MiB",
            file=sys.stderr,
            flush=True,
        )
        labels = [label.strip() for label in args.labels.split(",") if label.strip()]
        runner = user_script.TensorRTRunner(engine_bytes, cuda_backend=args.cuda_backend)
        capture = user_script.open_source(args.source)

    writer = None
    saved_video_path = None
    frame_index = 0
    fps_ema = 0.0
    last_preview_at = 0.0
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    frame_period = 1.0 / source_fps if np.isfinite(source_fps) and source_fps > 0 else 0.0
    preview_period = 1.0 / args.preview_fps if args.preview_fps > 0 else 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            started = time.perf_counter()
            tensor, gain, pad = user_script.preprocess(frame, runner.input_shape)
            outputs = runner.infer(tensor)
            boxes, scores, class_ids = user_script.decode_yolo26(
                outputs=outputs,
                input_hw=(runner.input_shape[2], runner.input_shape[3]),
                original_hw=frame.shape[:2],
                gain=gain,
                pad=pad,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                max_det=args.max_det,
                num_classes=len(labels),
            )
            elapsed = max(time.perf_counter() - started, 1e-9)
            current_fps = 1.0 / elapsed
            fps_ema = current_fps if frame_index == 0 else 0.9 * fps_ema + 0.1 * current_fps
            frame_index += 1

            user_script.draw_detections(frame, boxes, scores, class_ids, labels)
            cv2.putText(
                frame,
                f"FPS {fps_ema:.1f}  detections {len(boxes)}  conf {args.conf:.2f}",
                (18, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save:
                if writer is None:
                    writer, saved_video_path = user_script.create_writer(args.save, capture, frame)
                writer.write(frame)

            now = time.monotonic()
            if preview_period == 0 or now - last_preview_at >= preview_period:
                write_frame(frame_stream, frame, args.stream_width, args.jpeg_quality)
                last_preview_at = now

            if args.realtime and frame_period > 0:
                remaining = frame_period - (time.perf_counter() - started)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        runner.close()

    print(f"Processed {frame_index} frames", file=sys.stderr, flush=True)
    if saved_video_path is not None:
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
