from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.handlers import RunContext
from backend.catalog import FunctionCatalog
from backend.image_dedup import run_image_dedup
from backend.task_manager import TaskCancelled


class ImageCycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "images"
        self.folder.mkdir()

    def write(self, poses, labels=False):
        for i, pose in enumerate(poses):
            image = np.full((160, 160, 3), 30, np.uint8)
            cv2.rectangle(image, (10 + pose * 24, 35), (32 + pose * 24, 120), (210, 180, 90), -1)
            path = self.folder / f"frame_{i}.png"
            cv2.imwrite(str(path), image)
            if labels:
                path.with_suffix('.txt').write_text('0 0.5 0.5 0.2 0.2\n')

    def run_cycle(self, end=4, start=0, **parameters):
        return run_image_dedup(RunContext(
            function_id="image_dedup",
            paths={"images_folder": self.folder, "cycle_start_image": self.folder / f"frame_{start}.png",
                   "cycle_end_image": self.folder / f"frame_{end}.png"},
            parameters={"dedup_mode": "完整工序循环去重", "dedup_strength": "较保守", **parameters},
            report=lambda _: None))

    def test_variable_speed_whole_cycles_preview_execute_rollback(self):
        self.write([0, 1, 2, 3, 4] + [0, 0, 1, 2, 2, 3, 4, 4] + [0, 1, 2, 3, 4], labels=True)
        original = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        preview = self.run_cycle(operation="仅分析（不移动文件）")
        self.assertEqual(preview['proposedRemovedCount'], 13)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        result = self.run_cycle()
        self.assertEqual(result['removedCount'], 13)
        self.assertEqual(len(list(self.folder.glob('*.png'))), 5)
        rollback = self.run_cycle(operation="回滚最近一次")
        self.assertEqual(rollback['restoredCount'], 13)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_local_single_frame_anomaly_preserves_whole_candidate(self):
        self.write([0, 1, 2, 3, 4] * 2)
        path = self.folder / 'frame_7.png'
        image = cv2.imread(str(path))
        image[5:21, 5:21] = 255
        cv2.imwrite(str(path), image)
        result = self.run_cycle()
        self.assertEqual(result['removedCount'], 0)

    def test_incomplete_and_reverse_order_are_not_deleted(self):
        self.write([0, 1, 2, 3, 4] + [4, 3, 2, 1, 0] + [0, 1, 2])
        self.assertEqual(self.run_cycle()['removedCount'], 0)

    def test_annotation_geometry_changes_keep_cycle(self):
        self.write([0, 1, 2, 3, 4] * 2, labels=True)
        (self.folder / 'frame_7.txt').write_text('0 0.6 0.5 0.2 0.2\n')
        self.assertEqual(self.run_cycle()['removedCount'], 0)

    def test_static_template_rejected(self):
        self.write([0] * 10)
        with self.assertRaisesRegex(ValueError, '动作变化不足'):
            self.run_cycle()
        self.assertEqual(len(list(self.folder.glob('*.png'))), 10)

    def test_reference_and_prefix_never_removed(self):
        self.write([4, 4] + [0, 1, 2, 3, 4] * 2)
        result = self.run_cycle(start=2, end=6)
        self.assertEqual(result['removedCount'], 5)
        self.assertTrue(all((self.folder / f'frame_{i}.png').exists() for i in range(7)))

    def test_invalid_end_rejected(self):
        self.write([0, 1, 2, 3, 4] * 2)
        with self.assertRaisesRegex(ValueError, '结束图片'):
            self.run_cycle(start=4, end=0)

    def test_faster_cycle_and_closed_loop(self):
        self.write([0, 0, 1, 1, 2, 2, 3, 3, 4, 4] + [0, 1, 2, 3, 4])
        self.assertEqual(self.run_cycle(end=9)['removedCount'], 5)

    def test_same_start_end_pose(self):
        self.write([0, 1, 2, 3, 0] * 3)
        self.assertEqual(self.run_cycle()['removedCount'], 10)

    def test_large_template_uses_all_frames_not_only_coarse_anchors(self):
        poses = [pose for pose in range(5) for _ in range(30)]
        self.write(poses * 2)
        self.assertEqual(self.run_cycle(end=149, operation='仅分析（不移动文件）')['proposedRemovedCount'], 150)
        path = self.folder / 'frame_221.png'
        image = cv2.imread(str(path))
        image[5:21, 5:21] = 255
        cv2.imwrite(str(path), image)
        self.assertEqual(self.run_cycle(end=149)['removedCount'], 0)

    def test_duplicate_stems_are_rejected(self):
        self.write([0, 1, 2, 3, 4] * 2)
        cv2.imwrite(str(self.folder / 'frame_7.jpg'), cv2.imread(str(self.folder / 'frame_7.png')))
        with self.assertRaisesRegex(ValueError, '同名'):
            self.run_cycle()

    def test_different_sources_are_protected(self):
        self.write([0, 1, 2, 3, 4] * 2)
        for i in range(5, 10):
            (self.folder / f'frame_{i}.png').rename(self.folder / f'other_{i}.png')
        self.assertEqual(self.run_cycle()['removedCount'], 0)

    def test_source_changed_during_analysis_stops_before_moving(self):
        self.write([0, 1, 2, 3, 4] * 2)
        from backend.image_cycles import plan_cycles
        def changed(*args, **kwargs):
            result = plan_cycles(*args, **kwargs)
            (self.folder / 'frame_8.txt').write_text('1 0.5 0.5 0.2 0.2')
            return result
        with patch('backend.image_cycles.plan_cycles', side_effect=changed):
            with self.assertRaisesRegex(ValueError, '发生变化'):
                self.run_cycle()
        self.assertEqual(len(list(self.folder.glob('*.png'))), 10)

    def test_cancel_during_move_restores_every_file(self):
        self.write([0, 1, 2, 3, 4] * 2, labels=True)
        original = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        event = threading.Event()
        replace = Path.replace
        def cancel_after_move(source, target):
            result = replace(source, target)
            if 'files' in target.parts:
                event.set()
            return result
        with patch.object(Path, 'replace', cancel_after_move):
            with self.assertRaises(TaskCancelled):
                run_image_dedup(RunContext('image_dedup', {
                    'images_folder': self.folder, 'cycle_start_image': self.folder / 'frame_0.png',
                    'cycle_end_image': self.folder / 'frame_4.png'},
                    {'dedup_mode': '完整工序循环去重'}, lambda _: None, stop_event=event))
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_catalog_migrates_existing_settings_without_overwriting_defaults(self):
        storage = Path(self.temp.name) / 'functions.json'
        catalog = FunctionCatalog(storage)
        catalog._save()
        items = json.loads(storage.read_text())
        item = next(item for item in items if item.get('handlerId') == 'image.smart_dedup')
        item['pathFields'] = [f for f in item['pathFields'] if not f['id'].startswith('cycle_')]
        item['parameters'] = [f for f in item['parameters'] if not f['id'].startswith('cycle_') and f['id'] != 'dedup_mode']
        next(f for f in item['parameters'] if f['id'] == 'dedup_strength')['default'] = '保守'
        storage.write_text(json.dumps(items))
        storage.with_name(storage.name + '.image-cycle-dedup-v2').unlink()
        migrated = FunctionCatalog(storage).get('image_dedup')
        self.assertIn('cycle_start_image', [f['id'] for f in migrated['pathFields']])
        self.assertEqual(next(f for f in migrated['parameters'] if f['id'] == 'dedup_strength')['default'], '保守')
        self.assertIn('仅分析（不移动文件）', next(f for f in migrated['parameters'] if f['id'] == 'operation')['options'])
        self.assertEqual(migrated, FunctionCatalog(storage).get('image_dedup'))


if __name__ == '__main__':
    unittest.main()
