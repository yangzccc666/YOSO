from __future__ import annotations

import io
import json
import os
import queue
import select
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from backend.local_pt_inference import LocalPTInferenceManager
from backend.remote_inference import PreviewPacketDecoder, RemoteInferenceManager, RemoteInferenceSession
from backend.remote_star_worker import read_control_commands, seek_video_frame


class FrameStepUnitTests(unittest.TestCase):
    def test_api_routes_dispatch_both_step_controls(self):
        from backend import server
        for prefix, manager_name in (("local-pt-inference", "local_pt_inference_manager"), ("remote-inference", "remote_inference_manager")):
            payload = b'{"direction":1}'
            raw = f"POST /api/{prefix}/test-id/step HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload
            response = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=response.extend)
            with patch.object(server.license_store, "status", return_value={"activated": True, "role": "user"}), patch.object(getattr(server, manager_name), "step_frame", return_value={"sourceFrameIndex": 12, "status": "paused"}) as call:
                server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
            call.assert_called_once_with("test-id", 1)
            head, body = bytes(response).split(b"\r\n\r\n", 1)
            self.assertIn(b"200", head)
            self.assertEqual(json.loads(body)["sourceFrameIndex"], 12)

    def remote(self, index=50):
        manager = RemoteInferenceManager()
        session = RemoteInferenceSession("step-test", "device", "user", status="running")
        session.source_fps = 25
        session.source_frame_count = 250
        session.duration_seconds = 10
        session.publish_frame(b"initial", index / 25)
        manager._sessions[session.id] = session
        commands = []
        class Sink:
            def write(self, value):
                commands.append(json.loads(value))
            def flush(self):
                command = commands[-1]
                def publish():
                    session.publish_frame(b"target", command["frameIndex"] / 25, command["generation"])
                threading.Thread(target=publish, daemon=True).start()
        session.control_stdin = Sink()
        return manager, session, commands

    def test_remote_step_uses_accepted_source_index_and_rejects_late_frames(self):
        manager, session, commands = self.remote()
        result = manager.step_frame(session.id, 1)
        self.assertEqual(commands[-1], {"action": "step", "frameIndex": 51, "generation": 1})
        self.assertEqual(result["sourceFrameIndex"], 51)
        self.assertEqual(result["status"], "paused")
        self.assertFalse(result["frameStepPending"])
        count = session.frame_count
        session.publish_frame(b"late", 6.0, 0)
        session.publish_frame(b"extra", 6.0, 1)
        self.assertEqual(session.frame_count, count)
        self.assertEqual(manager.step_frame(session.id, -1)["sourceFrameIndex"], 50)

    def test_boundary_clamp_and_step_back_from_eof(self):
        manager, session, _ = self.remote(0)
        self.assertEqual(manager.step_frame(session.id, -1)["sourceFrameIndex"], 0)
        with session.condition:
            session.source_frame_index = 249
            session.at_end = True
            session.position_seconds = 10
            session.replay_deadline = time.monotonic() + 10
        result = manager.step_frame(session.id, -1)
        self.assertEqual(result["sourceFrameIndex"], 248)
        self.assertFalse(result["atEnd"])
        self.assertEqual(result["positionSeconds"], 248 / 25)
        self.assertEqual(manager.step_frame(session.id, 1)["sourceFrameIndex"], 249)
        self.assertEqual(manager.step_frame(session.id, 1)["sourceFrameIndex"], 249)

    def test_invalid_direction_missing_metadata_and_pending_requests(self):
        manager, session, commands = self.remote()
        for value in (-2, 0, 2, True, 1.0, "1", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                manager.step_frame(session.id, value)
        self.assertEqual(commands, [])
        for field, value in (("source_fps", 0), ("source_frame_count", 0), ("source_frame_index", -1), ("status", "stopped"), ("seek_generation", 1)):
            original = getattr(session, field)
            setattr(session, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                manager.step_frame(session.id, 1)
            setattr(session, field, original)

    def test_failed_control_write_does_not_change_playback_state(self):
        manager, session, _ = self.remote()
        sink = io.StringIO()
        sink.close()
        session.control_stdin = sink
        with self.assertRaises(RuntimeError):
            manager.step_frame(session.id, 1)
        self.assertEqual(session.status, "running")
        self.assertFalse(session.frame_step_pending)
        self.assertEqual(session.seek_generation, 0)

    def test_step_control_pauses_and_ignores_invalid_commands(self):
        commands, paused = queue.SimpleQueue(), threading.Event()
        source = '\n'.join(json.dumps(value) for value in (
            {"action": "step", "frameIndex": 17, "generation": 1},
            {"action": "step", "frameIndex": -1, "generation": 2},
            {"action": "step", "frameIndex": True, "generation": 3},
            {"action": "step", "frameIndex": 8, "generation": -1},
            {"action": "seek", "seconds": float("nan"), "generation": 2},
        ))
        with patch("sys.stdin", io.StringIO(source)):
            read_control_commands(commands, paused)
        self.assertTrue(paused.is_set())
        self.assertEqual(commands.get_nowait(), (17, 1, "frame"))
        self.assertTrue(commands.empty())

    def test_late_eof_metadata_cannot_cancel_new_frame_step(self):
        manager, session, _ = self.remote()
        manager.step_frame(session.id, 1)
        class Channel:
            ready = True
            def recv_stderr_ready(self): return self.ready
            def recv_stderr(self, _):
                self.ready = False
                return b'VIDEO_EOF_JSON:{"graceSeconds":10,"generation":0}\n'
        manager._drain_stderr(Channel(), session, bytearray())
        self.assertFalse(session.at_end)


class IdentityModel:
    names = {0: "object"}
    def __init__(self, _): pass
    def predict(self, frame, **_):
        class Result:
            def plot(self): return frame.copy()
        return [Result()]


class FrameStepIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="frame_step_")
        self.root = Path(self.temporary.name)
        self.video = self.root / "source.mp4"
        self.model = self.root / "fake.pt"
        self.model.write_bytes(b"fixture")
        self.fps = 30000 / 1001
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (160, 120))
        self.assertTrue(writer.isOpened())
        for index in range(100):
            writer.write(np.full((120, 160, 3), index * 2, dtype=np.uint8))
        writer.release()
        self.manager = LocalPTInferenceManager(IdentityModel)

    def tearDown(self):
        self.manager.stop_all()
        for session in self.manager._sessions.values():
            self.until(lambda: session.status in {"stopped", "completed", "failed"})
        self.temporary.cleanup()

    def until(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate(): return
            time.sleep(0.01)
        self.fail("Timed out waiting for frame")

    def start(self, **parameters):
        result = self.manager.start({"paths": {"model_file": str(self.model), "video_file": str(self.video)},
            "parameters": {"realtime": True, "preview_fps": 1, "replay_grace_seconds": 3, **parameters}})
        session = self.manager.get(result["id"])
        self.until(lambda: session.frame_count > 0)
        return session

    def test_local_steps_are_exact_at_fractional_fps_even_when_preview_is_throttled(self):
        session = self.start()
        self.manager.set_paused(session.id, True)
        base = session.source_frame_index
        for direction in (1, 1, -1, 1):
            count = session.frame_count
            result = self.manager.step_frame(session.id, direction)
            base += direction
            self.assertEqual(result["sourceFrameIndex"], base)
            self.assertEqual(result["sourceFrameCount"], 100)
            self.assertEqual(result["status"], "paused")
            self.assertFalse(result["frameStepPending"])
            self.assertEqual(session.frame_count, count + 1)
            image = cv2.imdecode(np.frombuffer(session.latest_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertAlmostEqual(float(image[80:110].mean()), base * 2, delta=7)
        count = session.frame_count
        time.sleep(0.1)
        self.assertEqual(session.frame_count, count)
        self.manager.set_paused(session.id, False)
        self.until(lambda: session.frame_count > count)

    def test_local_running_step_auto_pauses_and_step_back_from_eof(self):
        session = self.start(realtime=False, preview_fps=0)
        self.until(lambda: session.at_end)
        self.assertEqual(session.source_frame_index, 99)
        result = self.manager.step_frame(session.id, -1)
        self.assertEqual(result["sourceFrameIndex"], 98)
        self.assertEqual(result["status"], "paused")
        self.assertFalse(result["atEnd"])
        self.manager.set_paused(session.id, False)
        with session.condition:
            base = session.source_frame_index
            result = self.manager.step_frame(session.id, -1)
        self.assertEqual(result["sourceFrameIndex"], max(0, base - 1))
        self.assertEqual(result["status"], "paused")

    def test_decoder_positions_to_exact_original_frames_in_both_directions(self):
        capture = cv2.VideoCapture(str(self.video))
        try:
            for target in (40, 39, 41, 0, 99):
                self.assertTrue(seek_video_frame(capture, target))
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertAlmostEqual(float(frame.mean()), target * 2, delta=7)
        finally:
            capture.release()

    @unittest.skipIf(sys.platform == "win32", "Pipe select is POSIX-only")
    def test_remote_worker_steps_one_source_frame_without_realtime_skips(self):
        script = self.root / "fake_script.py"
        script.write_text(textwrap.dedent('''
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
        process = subprocess.Popen([
            sys.executable, "-u", str(Path(__file__).resolve().parents[1] / "backend/remote_star_worker.py"),
            "--script", str(script), "--model", str(self.model), "--source", str(self.video), "--labels", "object",
            "--session-token", "frame-step-test", "--realtime", "--preview-fps", "1", "--replay-grace-seconds", "5",
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        decoder = PreviewPacketDecoder()
        def read_frames(generation, timeout=2):
            found = []
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                ready, _, _ = select.select([process.stdout], [], [], 0.05)
                if ready:
                    data = os.read(process.stdout.fileno(), 65536)
                    if not data: break
                    found.extend(frame for frame in decoder.feed(data) if frame[0] == generation)
                    if found: break
            return found
        try:
            self.assertTrue(read_frames(0))
            for generation, target in enumerate((20, 19, 20), 1):
                process.stdin.write((json.dumps({"action": "step", "frameIndex": target, "generation": generation}) + "\n").encode())
                process.stdin.flush()
                frames = read_frames(generation)
                self.assertEqual(len(frames), 1)
                _, position, jpeg = frames[0]
                self.assertAlmostEqual(position * self.fps, target, delta=0.01)
                image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertAlmostEqual(float(image[80:110].mean()), target * 2, delta=7)
                self.assertEqual(read_frames(generation, timeout=0.1), [])
        finally:
            process.terminate()
            try: process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


if __name__ == "__main__":
    unittest.main()
