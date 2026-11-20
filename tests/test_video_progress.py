"""Decoder progress and incorrect NVR container-duration regressions."""
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend import bundled_video_frames as frames, video_frames as adapter
from backend.handlers import RunContext
from backend.task_manager import TaskCancelled


class Capture:
    def __init__(self, actual=100, reported=100, bad=()):
        self.actual, self.reported, self.bad = actual, reported, set(bad)
        self.cursor, self.grabs, self.released = -1, 0, False

    def isOpened(self):
        return True

    def get(self, key):
        return {cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: self.reported,
                cv2.CAP_PROP_FRAME_WIDTH: 64, cv2.CAP_PROP_FRAME_HEIGHT: 48}.get(key, 0)

    def grab(self):
        self.grabs += 1
        if self.cursor + 1 >= self.actual:
            return False
        self.cursor += 1
        return True

    def retrieve(self):
        return (False, None) if self.cursor in self.bad else (True, np.full((48, 64, 3), self.cursor % 255, dtype=np.uint8))

    def set(self, *_):
        raise AssertionError('Repeated frame seeks must not be used')

    def release(self):
        self.released = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class VideoProgressTests(unittest.TestCase):
    def run_capture(self, capture, **kwargs):
        updates, status = [], []
        with tempfile.TemporaryDirectory() as scratch, patch.object(frames, 'open_video', return_value=capture), \
             patch.object(frames, '_stream_frame_count', side_effect=lambda path, count, fps, *_: (count, fps)):
            result = frames.vedio2img('fixture.mp4', scratch, target_fps=5,
                report_progress=lambda *args: updates.append(args), report_status=status.append, **kwargs)
        self.assertTrue(capture.released)
        return result, updates, status

    def test_sequential_decode_keeps_original_indices_and_sampling(self):
        capture = Capture()
        result, updates, _ = self.run_capture(capture)
        self.assertEqual(result['selected_frame_indices'], list(range(0, 100, 5)))
        self.assertEqual(result['success_count'], 20)
        self.assertEqual(updates[-1], (20, 20))
        self.assertEqual(capture.grabs, 96)

    def test_retrieve_failures_still_report_every_attempt(self):
        result, updates, status = self.run_capture(Capture(bad=range(100)))
        self.assertEqual(result['success_count'], 0)
        self.assertEqual(result['read_error_count'], 20)
        self.assertEqual(len(updates), 20)
        self.assertEqual(updates[-1], (20, 20))
        self.assertTrue(any('读取失败' in line for line in status))

    def test_saved_indices_exclude_failed_retrieves(self):
        result, updates, _ = self.run_capture(Capture(bad={0, 5}))
        self.assertEqual(result['selected_frame_indices'], list(range(10, 100, 5)))
        self.assertEqual(result['planned_count'], 20)
        self.assertEqual(result['success_count'], 18)
        self.assertEqual(len(updates), 20)

    def test_early_end_is_bounded_and_unprocessed_tail_is_not_a_fake_failure_storm(self):
        capture = Capture(actual=10, reported=10000)
        result, updates, status = self.run_capture(capture)
        self.assertEqual(result['success_count'], 2)
        self.assertEqual(result['read_error_count'], 1)
        self.assertEqual(result['skipped_count'], 1997)
        self.assertEqual(len(updates), 3)
        self.assertEqual(capture.grabs, 11)
        self.assertTrue(any('可能已提前到结尾或视频存在损坏' in line for line in status))

    def test_write_failures_are_separate_and_progress_continues(self):
        save_jpeg = frames._save_jpeg
        attempts = iter((PermissionError('拒绝访问'), cv2.error('JPEG error')))
        def write(path, frame):
            failure = next(attempts, None)
            if failure:
                raise failure
            save_jpeg(path, frame)
        with patch.object(frames, '_save_jpeg', side_effect=write):
            result, updates, status = self.run_capture(Capture())
        self.assertEqual(result['write_error_count'], 2)
        self.assertEqual(result['read_error_count'], 0)
        self.assertEqual(result['success_count'], 18)
        self.assertEqual(updates[-1], (20, 20))
        self.assertTrue(any('图片保存失败' in line and '拒绝访问' in line for line in status))

    def test_cancellation_during_skipped_frame_decoding_releases_capture(self):
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            if checks > 4:
                raise TaskCancelled('用户终止')
        capture = Capture()
        with self.assertRaises(TaskCancelled):
            self.run_capture(capture, check_cancelled=cancel)
        self.assertTrue(capture.released)


class StreamMetadataTests(unittest.TestCase):
    def process(self, stream):
        process = Mock(returncode=0)
        process.communicate.return_value = (json.dumps({'streams': [stream]}), '')
        process.poll.return_value = 0
        return process

    def test_video_duration_replaces_inflated_container_count_and_preserves_cap(self):
        process = self.process({'duration': '298.055133', 'avg_frame_rate': '25/1'})
        capture = Capture(actual=7452, reported=11131)
        status = []
        with tempfile.TemporaryDirectory() as scratch, patch.object(frames, 'open_video', return_value=capture), \
             patch.object(frames.shutil, 'which', return_value='ffprobe'), \
             patch.object(frames.subprocess, 'Popen', return_value=process) as popen:
            result = frames.vedio2img('fixture.mp4', scratch, target_fps=12.5, max_images=16,
                                     report_progress=lambda *_: None, report_status=status.append)
        self.assertEqual(result['end_frame'], 7452)
        self.assertEqual(result['success_count'], 16)
        self.assertEqual(result['error_count'], 0)
        self.assertEqual(result['selected_frame_indices'][-1], 7450)
        self.assertTrue(any('11131 → 7452' in line for line in status))
        self.assertNotIn('format=duration', ' '.join(popen.call_args.args[0]))

    def test_exact_stream_count_and_missing_fps(self):
        process = self.process({'nb_frames': '100', 'duration': '4', 'avg_frame_rate': '25/1'})
        with patch.object(frames.shutil, 'which', return_value='ffprobe'), patch.object(frames.subprocess, 'Popen', return_value=process):
            self.assertEqual(frames._stream_frame_count('fixture', 150, 0, None, Mock()), (100, 25))

    def test_inconsistent_nvr_duration_checks_actual_packet_count(self):
        metadata = self.process({'duration': '298.055133', 'avg_frame_rate': '25/1'})
        counted = self.process({'nb_read_packets': '7435'})
        with patch.object(frames.shutil, 'which', return_value='ffprobe'), \
             patch.object(frames.subprocess, 'Popen', side_effect=[metadata, counted]) as popen:
            self.assertEqual(frames._stream_frame_count('fixture', 11131, 25, None, Mock()), (7435, 25))
        self.assertIn('-count_packets', popen.call_args_list[1].args[0])

    def test_normal_metadata_does_not_need_full_packet_scan(self):
        metadata = self.process({'duration': '4', 'avg_frame_rate': '25/1'})
        with patch.object(frames.shutil, 'which', return_value='ffprobe'), patch.object(frames.subprocess, 'Popen', return_value=metadata) as popen:
            self.assertEqual(frames._stream_frame_count('fixture', 100, 25, None, Mock()), (100, 25))
        popen.assert_called_once()

    def test_missing_ffprobe_does_not_add_a_mandatory_dependency(self):
        with patch.object(frames.shutil, 'which', return_value=None):
            self.assertEqual(frames._stream_frame_count('fixture', 100, 25, None, Mock()), (100, 25))

    def test_probe_timeout_kills_child_and_falls_back(self):
        process = Mock()
        process.poll.return_value = None
        process.communicate.side_effect = [subprocess.TimeoutExpired('ffprobe', .2), ('', '')]
        with patch.object(frames.shutil, 'which', return_value='ffprobe'), patch.object(frames.subprocess, 'Popen', return_value=process), \
             patch.object(frames.time, 'monotonic', side_effect=[0, 9]):
            self.assertEqual(frames._stream_frame_count('fixture', 100, 25, None, Mock()), (100, 25))
        process.kill.assert_called_once()

    def test_probe_cancel_kills_child_and_propagates_stop(self):
        process = Mock()
        process.poll.return_value = None
        process.communicate.return_value = ('', '')
        with patch.object(frames.shutil, 'which', return_value='ffprobe'), patch.object(frames.subprocess, 'Popen', return_value=process):
            with self.assertRaises(TaskCancelled):
                frames._stream_frame_count('fixture', 100, 25, Mock(side_effect=TaskCancelled('用户终止')), Mock())
        process.kill.assert_called_once()


class AdapterProgressTests(unittest.TestCase):
    def test_chinese_video_and_output_paths_use_stream_reading_and_binary_jpeg_writes(self):
        if 'open(source, apiPreference, params)' not in cv2.VideoCapture.open.__doc__:
            self.skipTest('This OpenCV build predates the VideoCapture stream overload')
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            temporary = root / 'fixture.mp4'
            writer = cv2.VideoWriter(str(temporary), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
            writer.release()
            input_folder = root / '渗透线源视频'
            input_folder.mkdir()
            source = temporary.rename(input_folder / '渗透2号线-工位1_beike_20261116.mp4')
            destination = root / '抽帧结果'
            open_capture = frames.cv2.VideoCapture
            opened_streams = []
            def reject_filename_capture(source_or_path, *args):
                if isinstance(source_or_path, str):
                    raise AssertionError("The decoder must receive a stream for the Windows Unicode path")
                opened_streams.append(source_or_path)
                return open_capture(source_or_path, *args)
            with patch.object(frames, 'sys', SimpleNamespace(platform='win32')), \
                 patch.object(frames.cv2, 'VideoCapture', side_effect=reject_filename_capture), \
                 patch.object(frames.cv2, 'imwrite', side_effect=AssertionError("Use Python's Unicode file IO")):
                result = adapter.run_video_frames(RunContext('video_frames', {'video_file': source, 'output_folder': destination},
                    {'target_fps': 5, 'parallel_workers': 1}, lambda _: None))
            pictures = sorted(Path(result['outputFolders'][0]).glob('*.jpg'))
            self.assertEqual(result['imageCount'], 5)
            self.assertEqual(len(pictures), 5)
            self.assertTrue(opened_streams and all(stream.closed for stream in opened_streams))
            self.assertTrue(all('渗透2号线' in path.name and path.is_relative_to(destination) for path in pictures))
            for path in pictures:
                decoded = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (48, 64, 3))

    def test_decode_and_write_failures_have_different_final_errors(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / 'fixture.mp4'
            source.touch()
            context = RunContext('video_frames', {'video_file': source}, {'target_fps': 5, 'parallel_workers': 1}, lambda _: None)
            with patch.object(frames, '_stream_frame_count', side_effect=lambda path, count, fps, *_: (count, fps)):
                with patch.object(frames, 'open_video', side_effect=lambda _: Capture(bad=range(100))):
                    with self.assertRaisesRegex(RuntimeError, '未成功解码所选帧'):
                        adapter.run_video_frames(context)
                with patch.object(frames, 'open_video', side_effect=lambda _: Capture()), \
                     patch.object(frames, '_save_jpeg', side_effect=PermissionError('输出目录拒绝访问')):
                    with self.assertRaisesRegex(RuntimeError, '视频已解码.*输出目录拒绝访问'):
                        adapter.run_video_frames(context)

    def test_early_end_warning_reaches_summary_and_history_payload(self):
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / 'fixture.mp4'
            source.touch()
            messages = []
            with patch.object(frames, 'open_video', side_effect=lambda _: Capture(actual=10, reported=100)), \
                 patch.object(frames, '_stream_frame_count', side_effect=lambda path, count, fps, *_: (count, fps)):
                result = adapter.run_video_frames(RunContext('video_frames', {'video_file': source},
                    {'target_fps': 5, 'parallel_workers': 1}, messages.append))
            self.assertEqual(result['warningVideoCount'], 1)
            self.assertEqual(result['details'][0]['readErrors'], 1)
            self.assertEqual(result['details'][0]['skippedImages'], 17)
            self.assertIn('未必完成全部计划抽帧', result['message'])
            self.assertTrue(any('异常汇总' in line for line in messages))

    def test_slow_decoder_emits_heartbeat_and_monitor_exits(self):
        release = threading.Event()
        def decode(**_):
            self.assertTrue(release.wait(2))
            return {'success_count': 1, 'selected_frame_indices': []}
        def report(text):
            if '解码器尚未返回' in text:
                release.set()
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch) / 'fixture.mp4'
            source.touch()
            with patch.object(frames, 'open_video', side_effect=lambda _: Capture()), patch.object(frames, 'vedio2img', side_effect=decode), \
                 patch.object(adapter, 'DECODER_HEARTBEAT_SECONDS', .02):
                adapter.run_video_frames(RunContext('video_frames', {'video_file': source}, {'parallel_workers': 1}, report))
        self.assertTrue(release.is_set())
        self.assertFalse(any(thread.name == 'frame-heartbeat-1' for thread in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
