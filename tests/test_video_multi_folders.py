from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from backend import bundled_video_frames, server
from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.task_manager import PlatformTaskManager, TaskCancelled
from backend.video_frame_resources import VideoFrameResourcePool
from backend.video_frames import _resolve_videos, normalize_video_folder_paths, run_video_frames


class VideoMultiFolderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        resources = {"cpuCount": 8, "cpuPercent": 5, "availableMemoryMb": 16384, "maxWorkers": 4, "recommendedWorkers": 2, "warning": ""}
        self.resources = patch("backend.video_frame_resources.machine_resources", return_value=resources)
        self.resources.start()
        self.addCleanup(self.resources.stop)
        self.pool_patch = patch("backend.video_frames.resource_pool", VideoFrameResourcePool())
        self.pool_patch.start()
        self.addCleanup(self.pool_patch.stop)

    def video(self, relative, frame_count=10):
        source = self.root / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
        self.assertTrue(writer.isOpened())
        for index in range(frame_count):
            writer.write(np.full((48, 64, 3), index * 20 % 255, dtype=np.uint8))
        writer.release()
        return source

    def context(self, folders, *, source=None, output=True):
        paths = {"video_folder": folders[0]} if folders else {}
        if source:
            paths["video_file"] = source
        if output:
            paths["output_folder"] = self.root / "output"
        messages = []
        return RunContext("video_frames", paths, {"_video_folder_paths": [str(path) for path in folders], "parallel_workers": 0, "target_fps": 2}, messages.append), messages

    def test_multiline_quotes_spaces_and_normalized_duplicate_paths(self):
        first = self.root / "工位 1"
        second = self.root / "工位2"
        raw = f'  "{second}"\r\n\n\'{first}\'\n{first}/.\n'
        self.assertEqual(normalize_video_folder_paths(raw), sorted([first.resolve(), second.resolve()], key=lambda path: str(path).casefold()))
        self.assertEqual(normalize_video_folder_paths([str(first), str(first / ".")]), [first.resolve()])
        self.assertEqual(normalize_video_folder_paths(""), [])
        for invalid in ({}, [3], 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_video_folder_paths(invalid)

    def test_scan_parent_child_same_folder_and_single_video_once(self):
        first = self.video("A/one.MP4")
        second = self.video("A/nested/deep/two.mp4")
        third = self.video("B/three.mp4")
        context, messages = self.context([self.root / "A/nested", self.root / "A", self.root / "B", self.root / "A"], source=first)
        videos = _resolve_videos(context, bundled_video_frames)
        self.assertEqual(set(videos), {first, second, third})
        self.assertEqual(len(videos), 3)
        self.assertTrue(any("跳过重复扫描" in message for message in messages))

    def test_actual_multi_folder_extraction_shared_output_and_automatic_workers(self):
        self.video("A/deep/工位1.mp4")
        self.video("B/工位1.mp4")
        context, _ = self.context([self.root / "A", self.root / "B"])
        result = run_video_frames(context)
        self.assertEqual(result["videoCount"], 2)
        self.assertEqual(result["imageCount"], 4)
        self.assertEqual(result["parallelWorkers"], 2)
        self.assertEqual(len(set(result["outputFolders"])), 2)
        self.assertTrue(all(Path(folder).parent == self.root / "output" for folder in result["outputFolders"]))

    def test_empty_output_saves_beside_each_source(self):
        self.video("A/one.mp4")
        self.video("B/deep/two.mp4")
        context, _ = self.context([self.root / "A", self.root / "B"], output=False)
        result = run_video_frames(context)
        for detail in result["details"]:
            self.assertEqual(Path(detail["output"]).parent, Path(detail["video"]).parent)
            self.assertEqual(len(list(Path(detail["output"]).glob("*.jpg"))), 2)

    def test_same_leaf_folder_merges_videos_with_correct_per_video_counts(self):
        self.video("input/leaf/first.mp4")
        self.video("input/leaf/second.MP4", frame_count=20)
        context, _ = self.context([self.root / "input"])
        result = run_video_frames(context)
        self.assertEqual(result["imageCount"], 6)
        self.assertEqual([detail["images"] for detail in result["details"]], [2, 4])
        self.assertEqual(len(result["outputFolders"]), 1)
        self.assertEqual(len({detail["output"] for detail in result["details"]}), 1)
        output = Path(result["outputFolders"][0])
        self.assertTrue(output.name.startswith("leaf_frames_"))
        self.assertEqual(len(list(output.glob("*.jpg"))), 6)
        self.assertEqual(len(list((self.root / "output").iterdir())), 1)

    def test_same_named_leaf_folders_are_distinct_and_root_videos_form_their_own_group(self):
        self.video("input/clips/first.mp4")
        self.video("input/clips/second.mp4")
        self.video("other/clips/first.mp4")
        self.video("other/clips/second.mp4")
        self.video("input/root.mp4")
        context, _ = self.context([self.root / "input", self.root / "other"])
        result = run_video_frames(context)
        self.assertEqual(result["sourceFolderCount"], 3)
        self.assertEqual(len(result["outputFolders"]), 3)
        self.assertEqual(result["imageCount"], 10)
        groups = {}
        for detail in result["details"]:
            groups.setdefault(Path(detail["video"]).parent, set()).add(detail["output"])
        self.assertTrue(all(len(outputs) == 1 for outputs in groups.values()))
        self.assertEqual(len({next(iter(outputs)) for outputs in groups.values()}), 3)

    def test_grouped_outputs_default_beside_source_and_repeat_run_preserves_old_images(self):
        self.video("input/deep/one.mp4")
        self.video("input/deep/two.mp4")
        context, _ = self.context([self.root / "input"], output=False)
        first = run_video_frames(context)
        output = Path(first["outputFolders"][0])
        self.assertEqual(output.parent, self.root / "input/deep")
        original = {path.name: path.read_bytes() for path in output.glob('*.jpg')}
        self.assertEqual(len(original), 4)
        second = run_video_frames(context)
        self.assertNotEqual(first["outputFolders"], second["outputFolders"])
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.glob('*.jpg')})
        self.assertEqual(second["imageCount"], 4)

    def test_zero_frame_video_is_not_counted_as_success_when_shared_folder_has_images(self):
        self.video("input/a_good.mp4")
        self.video("input/z_empty.mp4")
        context, _ = self.context([self.root / "input"])
        context.parameters['parallel_workers'] = 1
        original = bundled_video_frames.vedio2img
        def decode(**arguments):
            if Path(arguments['vedio_path']).stem == 'z_empty':
                return {'success_count': 0, 'error_count': 1, 'selected_frame_indices': []}
            return original(**arguments)
        with patch.object(bundled_video_frames, 'vedio2img', side_effect=decode):
            result = run_video_frames(context)
        self.assertEqual(result['imageCount'], 2)
        self.assertEqual(len(result['details']), 1)
        self.assertEqual(len(result['failedVideos']), 1)
        self.assertIn('z_empty.mp4', result['failedVideos'][0]['video'])

    def test_image_limit_still_applies_per_video_in_shared_folder(self):
        self.video("input/one.mp4")
        self.video("input/two.mp4")
        context, _ = self.context([self.root / "input"])
        context.parameters['max_images_limit'] = 1
        result = run_video_frames(context)
        self.assertEqual(result['imageCount'], 2)
        self.assertEqual([detail['images'] for detail in result['details']], [1, 1])
        self.assertEqual(len(result['outputFolders']), 1)

    def test_merge_all_subfolders_and_multiple_inputs_without_name_collisions(self):
        self.video('A/deep/same.mp4')
        self.video('B/deep/same.mp4')
        extra = self.video('single/same.mp4')
        context, messages = self.context([self.root / 'A', self.root / 'B'], source=extra)
        context.parameters.update(output_layout='全部视频合并保存', project_code='project', scene_code='scene')
        result = run_video_frames(context)
        self.assertEqual(result['outputLayout'], '全部视频合并保存')
        self.assertEqual(result['sourceFolderCount'], 3)
        self.assertEqual(len(result['outputFolders']), 1)
        output = Path(result['outputFolders'][0])
        self.assertEqual(output.parent, self.root / 'output')
        self.assertEqual({d['output'] for d in result['details']}, {str(output)})
        names = [path.name for path in output.glob('*.jpg')]
        self.assertEqual((result['imageCount'], len(names), len(set(names))), (6, 6, 6))
        for index in range(1, 4):
            self.assertEqual(sum(f'project_scene_video{index:02d}_fps2_' in name for name in names), 2)
        self.assertTrue(any('全部视频合并保存' in message for message in messages))

    def test_merged_parallel_workers_reserve_exactly_one_folder(self):
        self.video('A/same.mp4')
        self.video('B/same.mp4')
        context, _ = self.context([self.root / 'A', self.root / 'B'])
        context.parameters['output_layout'] = '全部视频合并保存'
        barrier = threading.Barrier(2)
        original = bundled_video_frames.vedio2img
        def decode(**arguments):
            barrier.wait(timeout=5)
            return original(**arguments)
        with patch.object(bundled_video_frames, 'vedio2img', side_effect=decode):
            result = run_video_frames(context)
        self.assertEqual(result['parallelWorkers'], 2)
        self.assertEqual(len(list((self.root / 'output').iterdir())), 1)
        self.assertEqual(result['imageCount'], 4)
        self.assertEqual(len(list(Path(result['outputFolders'][0]).glob('*.jpg'))), 4)

    def test_merged_blank_output_uses_first_sorted_video_parent_and_repeat_run_preserves_images(self):
        first = self.video('A/deep/a.mp4')
        self.video('B/b.mp4')
        context, _ = self.context([self.root / 'B', self.root / 'A'], output=False)
        context.parameters['output_layout'] = '全部视频合并保存'
        result = run_video_frames(context)
        output = Path(result['outputFolders'][0])
        self.assertEqual(output.parent, first.parent)
        previous = {p.name: p.read_bytes() for p in output.glob('*.jpg')}
        again = run_video_frames(context)
        self.assertNotEqual(again['outputFolders'], result['outputFolders'])
        self.assertEqual(previous, {p.name: p.read_bytes() for p in output.glob('*.jpg')})
        self.assertEqual(again['imageCount'], 4)

    def test_merged_per_video_limits_and_legacy_empty_result_counting(self):
        self.video('A/good.mp4')
        self.video('B/empty.mp4')
        context, _ = self.context([self.root / 'A', self.root / 'B'])
        context.parameters.update(output_layout='全部视频合并保存', max_images_limit=1, parallel_workers=1)
        original = bundled_video_frames.vedio2img
        def decode(**arguments):
            if Path(arguments['vedio_path']).stem == 'empty':
                return {'selected_frame_indices': [], 'error_count': 1}
            return original(**arguments)
        with patch.object(bundled_video_frames, 'vedio2img', side_effect=decode):
            result = run_video_frames(context)
        self.assertEqual(result['imageCount'], 1)
        self.assertEqual(len(result['details']), 1)
        self.assertEqual(len(result['failedVideos']), 1)
        self.assertIn('empty.mp4', result['failedVideos'][0]['video'])
        # A normal merged run still caps each video independently, not the batch.
        again = run_video_frames(context)
        self.assertEqual(again['imageCount'], 2)
        self.assertEqual([d['images'] for d in again['details']], [1, 1])
        self.assertEqual(len(again['outputFolders']), 1)

    def test_invalid_output_layout_is_rejected_before_creating_outputs(self):
        context, _ = self.context([self.root / 'missing'])
        context.parameters['output_layout'] = 'invalid'
        with self.assertRaisesRegex(ValueError, '图片保存方式'):
            run_video_frames(context)
        self.assertFalse((self.root / 'output').exists())

    def test_output_layout_migration_preserves_custom_parameters(self):
        storage = self.root / 'functions.json'
        catalog = FunctionCatalog(storage)
        item = catalog.get('video_frames')
        item['parameters'] = [p for p in item['parameters'] if p['id'] != 'output_layout']
        next(p for p in item['parameters'] if p['id'] == 'target_fps')['default'] = 20
        storage.write_text(json.dumps([item]))
        for _ in range(2):
            migrated = FunctionCatalog(storage).get('video_frames')
            fields = {p['id']: p for p in migrated['parameters']}
            self.assertEqual(fields['target_fps']['default'], 20)
            self.assertEqual(fields['output_layout']['default'], '按源文件夹分别保存')
            self.assertEqual(fields['output_layout']['options'], ['按源文件夹分别保存', '全部视频合并保存'])
            self.assertEqual(len(fields), len(migrated['parameters']))

    def test_invalid_folder_is_reported_but_valid_folders_continue(self):
        self.video("valid/one.mp4")
        context, messages = self.context([self.root / "missing", self.root / "valid"])
        result = run_video_frames(context)
        self.assertEqual(result["videoCount"], 1)
        self.assertTrue(any("路径不存在或不是文件夹" in message for message in messages))
        context, _ = self.context([self.root / "missing"])
        with self.assertRaisesRegex(ValueError, "没有找到可处理"):
            run_video_frames(context)

    def test_scanning_can_be_cancelled(self):
        folder = self.root / "input"
        folder.mkdir()
        stop = threading.Event()
        context = RunContext("video_frames", {"video_folder": folder}, {}, lambda _: None, stop)
        def walking(*args, **kwargs):
            yield str(folder), [], []
            stop.set()
            yield str(folder / "nested"), [], []
        with patch("backend.bundled_video_frames.os.walk", side_effect=walking), self.assertRaises(TaskCancelled):
            _resolve_videos(context, bundled_video_frames)

    def test_catalog_upgrades_label_preserving_other_fields(self):
        storage = self.root / "functions.json"
        item = FunctionCatalog(storage).get("video_frames")
        item["name"] = "自用抽帧"
        item["pathFields"][0]["label"] = "视频文件夹（递归扫描子文件夹）"
        storage.write_text(json.dumps([item], ensure_ascii=False), encoding="utf-8")
        for _ in range(2):
            migrated = FunctionCatalog(storage).get("video_frames")
            self.assertEqual(migrated["name"], "自用抽帧")
            self.assertIn("每行一个", migrated["pathFields"][0]["label"])
            self.assertEqual(len(migrated["pathFields"]), 3)

    def test_api_normalizes_folder_list_and_blocks_reordered_duplicate_batch(self):
        manager = PlatformTaskManager()
        release = threading.Event()
        finished = threading.Event()
        captured = []
        def execute(_handler, context):
            captured.append(context)
            release.wait(5)
            return {"message": "done", "outputFolders": []}
        def request(folders):
            payload = {"paths": {"video_folder": folders}, "parameters": {"parallel_workers": 0}}
            body = json.dumps(payload).encode()
            raw = f"POST /api/functions/video_frames/run HTTP/1.0\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
            head, result = bytes(output).split(b"\r\n\r\n", 1)
            return int(head.split()[1]), json.loads(result)
        with patch.object(server, "platform_task_manager", manager), patch.object(server, "execute", side_effect=execute), \
             patch.object(server, "record_finished_run", side_effect=lambda *args, **kwargs: finished.set()), \
             patch.object(server.license_store, "status", return_value={"activated": True, "role": "user"}):
            try:
                status, result = request(f"{self.root}/B\n{self.root}/A\n{self.root}/A")
                self.assertEqual(status, 202)
                self.assertIn("2 个文件夹", result["task"]["name"])
                status, _ = request(f"{self.root}/A\n{self.root}/B")
                self.assertEqual(status, 400)
                self.assertEqual(captured[0].parameters["_video_folder_paths"], [str(self.root / "A"), str(self.root / "B")])
                self.assertEqual(captured[0].paths["video_folder"], self.root / "A")
            finally:
                release.set()
                self.assertTrue(finished.wait(3))
