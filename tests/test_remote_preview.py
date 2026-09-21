from __future__ import annotations

import io
import json
import os
import queue
import select
import struct
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.remote_inference import PreviewPacketDecoder, RemoteInferenceManager, RemoteInferenceSession
from backend.remote_star_worker import PreviewSender, infer_and_draw_models, read_control_commands, write_frame, write_timeline_frame


class RemotePreviewTests(unittest.TestCase):
    def test_saved_video_fills_dropped_timeline_frames(self) -> None:
        class Writer:
            def __init__(self) -> None:
                self.frames: list[np.ndarray] = []

            def write(self, frame: np.ndarray) -> None:
                self.frames.append(frame.copy())

        writer = Writer()
        first = np.full((2, 2, 3), 10, dtype=np.uint8)
        fourth = np.full((2, 2, 3), 40, dtype=np.uint8)
        previous, index, written = write_timeline_frame(writer, first, 0, None, None)
        previous, index, more_written = write_timeline_frame(writer, fourth, 3, previous, index)

        self.assertEqual(written + more_written, 4)
        self.assertEqual([int(frame[0, 0, 0]) for frame in writer.frames], [10, 10, 10, 40])

    def test_multiple_models_use_same_clean_frame_before_drawing(self) -> None:
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        seen_thresholds: list[float] = []
        drawn_labels: list[str] = []

        class Runner:
            input_shape = (1, 3, 24, 32)

            def infer(self, _tensor):
                return object()

        class Script:
            @staticmethod
            def preprocess(image, _shape):
                self.assertEqual(int(image.sum()), 0)
                return object(), 1.0, (0, 0)

            @staticmethod
            def decode_yolo26(**kwargs):
                seen_thresholds.append(kwargs["conf_threshold"])
                return np.array([[1, 1, 5, 5]]), np.array([0.9]), np.array([0])

            @staticmethod
            def draw_detections(image, _boxes, _scores, _class_ids, labels):
                drawn_labels.extend(labels)
                image[0, 0] = 255

        count = infer_and_draw_models(Script(), [
            (Runner(), ["person"], 0.5),
            (Runner(), ["car"], 0.7),
        ], frame, 0.45, 300)
        self.assertEqual(count, 2)
        self.assertEqual(seen_thresholds, [0.5, 0.7])
        self.assertEqual(drawn_labels, ["person", "car"])

    def test_frame_protocol_includes_playback_position(self) -> None:
        stream = io.BytesIO()
        frame = np.zeros((20, 32, 3), dtype=np.uint8)
        write_frame(stream, frame, 960, 70, 12.5)
        payload = stream.getvalue()
        generation, position, length = struct.unpack(">IdI", payload[:16])
        self.assertEqual(generation, 0)
        self.assertEqual(position, 12.5)
        self.assertEqual(length, len(payload) - 16)
        self.assertIsNotNone(cv2.imdecode(np.frombuffer(payload[16:], dtype=np.uint8), cv2.IMREAD_COLOR))
        decoder = PreviewPacketDecoder()
        self.assertEqual(decoder.feed(payload[:5]), [])
        self.assertEqual(decoder.feed(payload[5:19]), [])
        self.assertEqual(decoder.feed(payload[19:]), [(0, 12.5, payload[16:])])

    def test_preview_sender_flushes_latest_frame_on_close(self) -> None:
        stream = io.BytesIO()
        sender = PreviewSender(stream, 960, 70)
        sender.submit(np.zeros((8, 8, 3), dtype=np.uint8), 1.0)
        sender.close()
        self.assertIsNone(sender.error)
        self.assertGreaterEqual(len(stream.getvalue()), 16)

    def test_playback_commands_and_seek_accept_only_valid_actions(self) -> None:
        commands: queue.SimpleQueue[tuple[float, int]] = queue.SimpleQueue()
        paused = threading.Event()
        with patch("sys.stdin", io.StringIO('bad json\n{"action":"pause"}\n{"action":"seek","seconds":4.25,"generation":2}\n')):
            read_control_commands(commands, paused)
        self.assertTrue(paused.is_set())
        self.assertEqual(commands.get_nowait(), (4.25, 2))
        self.assertTrue(commands.empty())
        with patch("sys.stdin", io.StringIO('{"action":"resume"}\n')):
            read_control_commands(commands, paused)
        self.assertFalse(paused.is_set())

    def test_manager_parses_metadata_and_sends_seek(self) -> None:
        manager = RemoteInferenceManager()
        session = RemoteInferenceSession("sample", "device", "user")
        manager._sessions[session.id] = session
        metadata = b'VIDEO_META_JSON:{"durationSeconds": 45.5, "sourceFps": 25}\n'

        class Channel:
            def __init__(self, payload: bytes) -> None:
                self.ready = True
                self.payload = payload

            def recv_stderr_ready(self) -> bool:
                return self.ready

            def recv_stderr(self, _size: int) -> bytes:
                self.ready = False
                return self.payload

        manager._drain_stderr(Channel(metadata), session, bytearray())
        self.assertEqual(session.snapshot()["durationSeconds"], 45.5)
        self.assertEqual(session.snapshot()["sourceFps"], 25)
        manager._drain_stderr(Channel(b'VIDEO_EOF_JSON:{"graceSeconds": 30}\n'), session, bytearray())
        self.assertTrue(session.snapshot()["atEnd"])
        self.assertGreater(session.snapshot()["replayRemainingSeconds"], 0)
        session.status = "running"
        session.control_stdin = io.StringIO()
        result = manager.seek(session.id, 60)
        self.assertLess(result["requestedSeconds"], 45.5)
        sent = json.loads(session.control_stdin.getvalue())
        self.assertEqual(sent["action"], "seek")
        self.assertEqual(sent["seconds"], result["requestedSeconds"])
        self.assertEqual(sent["generation"], 1)
        self.assertFalse(session.snapshot()["atEnd"])
        session.publish_frame(b"old", 1.0, 0)
        self.assertEqual(session.frame_count, 0)
        session.publish_frame(b"new", 2.0, 1)
        self.assertEqual(session.frame_count, 1)
        paused = manager.set_paused(session.id, True)
        self.assertEqual(paused["status"], "paused")
        self.assertIn('"action": "pause"', session.control_stdin.getvalue())
        session.publish_frame(b"late", 2.1, 1)
        self.assertEqual(session.frame_count, 1)
        manager.seek(session.id, 5.0)
        session.publish_frame(b"paused seek preview", 5.0, 2)
        session.publish_frame(b"another paused frame", 5.1, 2)
        self.assertEqual(session.frame_count, 2)
        self.assertEqual(session.position_seconds, 5.0)
        resumed = manager.set_paused(session.id, False)
        self.assertEqual(resumed["status"], "running")
        self.assertIn('"action": "resume"', session.control_stdin.getvalue())
        with self.assertRaisesRegex(ValueError, "有效秒数"):
            manager.seek(session.id, float("nan"))
        manager._drain_stderr(Channel(b'VIDEO_EOF_JSON:{"graceSeconds": 30}\n'), session, bytearray())
        self.assertTrue(session.snapshot()["atEnd"])
        session.publish_frame(b"late old frame", 45.4, 2)
        self.assertTrue(session.snapshot()["atEnd"])
        stopped = manager.stop(session.id)
        self.assertFalse(stopped["atEnd"])
        self.assertEqual(stopped["replayRemainingSeconds"], 0)

    def test_worker_seeks_in_video_without_tensorrt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(90):
                writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
            writer.release()
            module = root / "fake_inference.py"
            module.write_text(textwrap.dedent('''
                import cv2
                import numpy as np
                def read_engine_bytes(_path): return b"engine", 0
                def open_source(path): return cv2.VideoCapture(path)
                def preprocess(_frame, _shape): return np.zeros((1, 3, 32, 32)), 1.0, (0, 0)
                def decode_yolo26(**_kwargs): return [], [], []
                def draw_detections(*_args): pass
                class TensorRTRunner:
                    input_shape = (1, 3, 32, 32)
                    def __init__(self, *_args, **_kwargs): pass
                    def infer(self, _tensor): return []
                    def close(self): pass
            '''), encoding="utf-8")
            model = root / "dummy.star"
            model.write_bytes(b"test")
            worker = Path(__file__).resolve().parents[1] / "backend" / "remote_star_worker.py"
            process = subprocess.Popen([
                sys.executable, "-u", str(worker), "--script", str(module), "--model", str(model),
                "--source", str(video), "--labels", "object", "--session-token", "test-seek",
                "--realtime", "--preview-fps", "20", "--stream-width", "64", "--replay-grace-seconds", "3",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            decoder = PreviewPacketDecoder()
            first_position = None
            seek_position = None
            replay_position = None
            try:
                deadline = time.monotonic() + 7
                while time.monotonic() < deadline and seek_position is None:
                    ready, _, _ = select.select([process.stdout], [], [], 0.25)
                    if not ready:
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    for generation, position, _frame in decoder.feed(chunk):
                        if first_position is None:
                            first_position = position
                            self.assertIsNotNone(process.stdin)
                            process.stdin.write(b'{"action":"seek","seconds":2.0,"generation":1}\n')
                            process.stdin.flush()
                        elif generation == 1 and position >= 1.8:
                            seek_position = position
                            break
                self.assertIsNotNone(first_position)
                self.assertIsNotNone(seek_position, process.stderr.read().decode() if process.poll() is not None else "No seeked frame received")
                self.assertLess(first_position, 1.0)
                stderr_buffer = bytearray()
                while time.monotonic() < deadline and b"VIDEO_EOF_JSON:" not in stderr_buffer:
                    ready, _, _ = select.select([process.stdout, process.stderr], [], [], 0.25)
                    for pipe in ready:
                        chunk = os.read(pipe.fileno(), 65536)
                        if pipe is process.stderr:
                            stderr_buffer.extend(chunk)
                        elif chunk:
                            decoder.feed(chunk)
                self.assertIn(b"VIDEO_EOF_JSON:", stderr_buffer)
                process.stdin.write(b'{"action":"seek","seconds":0.5,"generation":2}\n')
                process.stdin.flush()
                while time.monotonic() < deadline and replay_position is None:
                    ready, _, _ = select.select([process.stdout], [], [], 0.25)
                    if not ready:
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    for generation, position, _frame in decoder.feed(chunk):
                        if generation == 2 and position >= 0.4:
                            replay_position = position
                            break
                self.assertIsNotNone(replay_position, "Could not drag backward after reaching video end")
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def test_worker_pauses_resumes_and_seeks_while_paused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(80):
                writer.write(np.full((48, 64, 3), index, dtype=np.uint8))
            writer.release()
            module = root / "fake_inference.py"
            module.write_text(textwrap.dedent('''
                import cv2
                import numpy as np
                def read_engine_bytes(_path): return b"engine", 0
                def open_source(path): return cv2.VideoCapture(path)
                def preprocess(_frame, _shape): return np.zeros((1, 3, 32, 32)), 1.0, (0, 0)
                def decode_yolo26(**_kwargs): return [], [], []
                def draw_detections(*_args): pass
                class TensorRTRunner:
                    input_shape = (1, 3, 32, 32)
                    def __init__(self, *_args, **_kwargs): pass
                    def infer(self, _tensor): return []
                    def close(self): pass
            '''), encoding="utf-8")
            model = root / "dummy.star"
            model.write_bytes(b"test")
            process = subprocess.Popen([
                sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "backend" / "remote_star_worker.py"),
                "--script", str(module), "--model", str(model), "--source", str(video),
                "--labels", "object", "--session-token", "test-pause", "--realtime", "--preview-fps", "20",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            decoder = PreviewPacketDecoder()

            def read_frames(seconds: float) -> list[tuple[int, float, bytes]]:
                frames = []
                deadline = time.monotonic() + seconds
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([process.stdout], [], [], min(0.05, max(0, deadline - time.monotonic())))
                    if ready:
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            break
                        frames.extend(decoder.feed(chunk))
                return frames

            try:
                self.assertIsNotNone(process.stdin)
                self.assertTrue(read_frames(0.7), "The worker did not start streaming")
                process.stdin.write(b'{"action":"pause"}\n')
                process.stdin.flush()
                read_frames(0.35)  # Drain the frame already in flight when pause arrived.
                self.assertEqual(read_frames(0.35), [], "Frames continued while paused")
                process.stdin.write(b'{"action":"resume"}\n')
                process.stdin.flush()
                self.assertTrue(read_frames(0.6), "No frames arrived after resuming")
                process.stdin.write(b'{"action":"pause"}\n')
                process.stdin.flush()
                read_frames(0.35)
                process.stdin.write(b'{"action":"seek","seconds":2.0,"generation":1}\n')
                process.stdin.flush()
                seek_frames = read_frames(0.6)
                self.assertTrue(any(generation == 1 and position >= 1.8 for generation, position, _ in seek_frames))
                self.assertEqual(read_frames(0.3), [], "Seek while paused should preview only one frame")
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def test_slow_inference_keeps_source_timeline_near_normal_speed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "source.mp4"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(120):
                writer.write(np.full((48, 64, 3), index % 255, dtype=np.uint8))
            writer.release()
            module = root / "slow_inference.py"
            module.write_text(textwrap.dedent('''
                import cv2
                import numpy as np
                import time
                def read_engine_bytes(_path): return b"engine", 0
                def open_source(path): return cv2.VideoCapture(path)
                def preprocess(_frame, _shape): return np.zeros((1, 3, 32, 32)), 1.0, (0, 0)
                def decode_yolo26(**_kwargs): return [], [], []
                def draw_detections(*_args): pass
                class TensorRTRunner:
                    input_shape = (1, 3, 32, 32)
                    def __init__(self, *_args, **_kwargs): pass
                    def infer(self, _tensor):
                        time.sleep(0.12)
                        return []
                    def close(self): pass
            '''), encoding="utf-8")
            model = root / "dummy.star"
            model.write_bytes(b"test")
            process = subprocess.Popen([
                sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "backend" / "remote_star_worker.py"),
                "--script", str(module), "--model", str(model), "--source", str(video),
                "--labels", "object", "--session-token", "slow-timeline", "--realtime",
                "--preview-fps", "0", "--stream-width", "64", "--replay-grace-seconds", "0",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
            decoder = PreviewPacketDecoder()
            first_position = None
            latest_position = None
            first_received_at = None
            try:
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    ready, _, _ = select.select([process.stdout], [], [], 0.2)
                    if not ready:
                        continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    for _generation, position, _frame in decoder.feed(chunk):
                        if first_position is None:
                            first_position = position
                            first_received_at = time.monotonic()
                        latest_position = position
                    if first_received_at is not None and time.monotonic() - first_received_at >= 0.85:
                        break
                self.assertIsNotNone(first_position)
                self.assertIsNotNone(latest_position)
                self.assertIsNotNone(first_received_at)
                elapsed = time.monotonic() - first_received_at
                source_progress = latest_position - first_position
                self.assertGreater(source_progress, elapsed * 0.55)
            finally:
                process.terminate()
                try:
                    process.communicate(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
