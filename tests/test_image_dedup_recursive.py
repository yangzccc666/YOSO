import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.image_dedup import _backup_root, run_image_dedup
from backend.task_manager import TaskCancelled


class RecursiveDedupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '场景'
        self.root.mkdir()
        self.logs = []

    def images(self, relative, count=3):
        folder = self.root / relative
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            image = folder / f'frame_{i}.png'
            self.assertTrue(cv2.imwrite(str(image), np.full((64, 80, 3), 60, np.uint8)))
            image.with_suffix('.txt').write_text('0 .5 .5 .2 .2')
        return folder

    def run_batch(self, **parameters):
        return run_image_dedup(RunContext('image_dedup', {'images_folder': self.root},
                                         {'recursive': True, **parameters}, self.logs.append))

    def snapshot(self, folders):
        return {str(path.relative_to(self.root)): path.read_bytes() for folder in folders for path in folder.iterdir() if path.is_file()}

    def test_arbitrary_depth_root_images_and_same_named_folders_are_isolated(self):
        folders = [self.images('.'), self.images('工位A/images'), self.images('工位B/deeper/images')]
        result = self.run_batch()
        self.assertEqual((result['folderCount'], result['imageCount'], result['keptCount'], result['removedCount']), (3, 9, 3, 6))
        self.assertTrue(all(len(list(folder.glob('*.png'))) == 1 for folder in folders))
        self.assertTrue(all(len(list(folder.glob('*.txt'))) == 1 for folder in folders))
        self.assertEqual({item['relativeFolder'] for item in result['folderResults']}, {'.', '工位A/images', '工位B/deeper/images'})

    def test_preview_reports_are_per_folder_and_do_not_move_images(self):
        folders = [self.images('A'), self.images('B')]
        original = self.snapshot(folders)
        result = self.run_batch(operation='仅分析（不移动文件）')
        self.assertEqual((result['proposedRemovedCount'], result['removedCount']), (4, 0))
        self.assertEqual(self.snapshot(folders), original)
        for item in result['folderResults']:
            report = json.loads(Path(item['analysisReport']).read_text())
            self.assertTrue(all((self.root / item['relativeFolder'] / r['reference']).exists() for r in report['removedImages']))
        with self.assertRaisesRegex(ValueError, '没有可以回滚'):
            self.run_batch(operation='回滚最近一次')

    def test_execute_preview_then_rollback_restores_images_and_labels_byte_for_byte(self):
        folders = [self.images('A'), self.images('B')]
        original = self.snapshot(folders)
        execution = self.run_batch()
        # A newer preview must not hide the most recent actual batch.
        self.run_batch(operation='仅分析（不移动文件）')
        restored = self.run_batch(operation='回滚最近一次')
        self.assertEqual(restored['restoredCount'], 4)
        self.assertEqual(self.snapshot(folders), original)
        self.assertEqual(json.loads(Path(execution['batchReport']).read_text())['status'], 'rolled_back')
        with self.assertRaisesRegex(ValueError, '没有可以回滚'):
            self.run_batch(operation='回滚最近一次')

    def test_rollback_conflict_preflights_every_directory_before_restoring_any(self):
        a, b = self.images('A'), self.images('B')
        result = self.run_batch()
        (b / 'frame_1.png').write_bytes(b'new user data')
        with self.assertRaisesRegex(RuntimeError, '同名文件'):
            self.run_batch(operation='回滚最近一次')
        self.assertEqual(len(list(a.glob('*.png'))), 1)
        self.assertEqual((b / 'frame_1.png').read_bytes(), b'new user data')
        self.assertEqual(json.loads(Path(result['batchReport']).read_text())['status'], 'completed')

    def test_does_not_restore_older_standalone_work(self):
        a = self.images('A')
        single = run_image_dedup(RunContext('image_dedup', {'images_folder': a}, {}, self.logs.append))
        self.images('B')
        self.run_batch()
        self.run_batch(operation='回滚最近一次')
        self.assertEqual(len(list(a.glob('*.png'))), 1)
        self.assertEqual(json.loads(Path(single['backupManifest']).read_text())['status'], 'active')

    def test_backup_directories_and_symlink_directories_are_not_scanned(self):
        self.images('A')
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'untouched.jpg').write_bytes(b'outside')
        (self.root / 'linked').symlink_to(outside, target_is_directory=True)
        self.run_batch()
        # Removed images live under a sibling hidden backup folder inside root.
        result = self.run_batch(operation='仅分析（不移动文件）')
        self.assertEqual((result['folderCount'], result['imageCount']), (1, 1))
        self.assertEqual((outside / 'untouched.jpg').read_bytes(), b'outside')

    def test_failed_directory_does_not_block_other_workstations(self):
        bad = self.images('A')
        cv2.imwrite(str(bad / 'frame_0.jpg'), np.zeros((10, 10, 3), np.uint8))
        good = self.images('B')
        result = self.run_batch()
        self.assertEqual((result['successfulFolderCount'], result['failedFolderCount']), (1, 1))
        self.assertIn('共用标注', result['folderResults'][0]['error'])
        self.assertEqual(len(list(good.glob('*.png'))), 1)
        self.assertEqual(len(list(bad.glob('*.png'))), 3)

    def test_cancel_preserves_completed_batch_scope_and_allows_rollback(self):
        a, b = self.images('A'), self.images('B')
        original = self.snapshot([a, b])
        event = threading.Event()
        def report(message):
            self.logs.append(message)
            if '[1/2 A] 详细去重与回滚清单' in message:
                event.set()
        with self.assertRaises(TaskCancelled):
            run_image_dedup(RunContext('image_dedup', {'images_folder': self.root}, {'recursive': True}, report, event))
        self.assertEqual(len(list(a.glob('*.png'))), 1)
        self.assertEqual(len(list(b.glob('*.png'))), 3)
        result = self.run_batch(operation='回滚最近一次')
        self.assertEqual(result['restoredCount'], 2)
        self.assertEqual(self.snapshot([a, b]), original)

    def test_cancel_mid_batch_rollback_reverts_all_restores(self):
        a, b = self.images('A'), self.images('B')
        result = self.run_batch()
        event = threading.Event()
        original_replace = Path.replace
        def move(source, target):
            value = original_replace(source, target)
            if target.parent == a:
                event.set()
            return value
        with patch.object(Path, 'replace', move), self.assertRaises(TaskCancelled):
            run_image_dedup(RunContext('image_dedup', {'images_folder': self.root},
                                      {'recursive': True, 'operation': '回滚最近一次'}, self.logs.append, event))
        self.assertEqual([len(list(f.glob('*.png'))) for f in [a, b]], [1, 1])
        self.assertEqual(json.loads(Path(result['batchReport']).read_text())['status'], 'completed')
        self.assertEqual(self.run_batch(operation='回滚最近一次')['restoredCount'], 4)

    def test_empty_root_and_cycle_batch_rejected_before_backup_creation(self):
        with self.assertRaisesRegex(ValueError, '没有支持的图片'):
            self.run_batch()
        self.images('A')
        with self.assertRaisesRegex(ValueError, '各工位单独指定'):
            self.run_batch(dedup_mode='完整工序循环去重')
        self.assertFalse(_backup_root(self.root).exists())

    def test_recursive_disabled_keeps_existing_single_folder_behavior(self):
        self.images('A')
        with self.assertRaisesRegex(ValueError, '没有支持的图片'):
            self.run_batch(recursive=False)

    def test_exact_and_continuous_modes_also_remain_directory_isolated(self):
        folders = [self.images('A', count=4), self.images('B', count=4)]
        exact = self.run_batch(dedup_mode='仅精确重复')
        self.assertEqual((exact['removedCount'], exact['keptCount']), (6, 2))
        self.run_batch(operation='回滚最近一次')
        adjacent = self.run_batch(dedup_mode='连续相似帧去重')
        self.assertEqual((adjacent['removedCount'], adjacent['keptCount']), (4, 4))
        self.assertTrue(all(len(list(folder.glob('*.png'))) == 2 for folder in folders))

    def test_batch_rollback_rejects_a_workstation_directory_replaced_by_a_link(self):
        folder = self.images('A')
        self.run_batch()
        outside = Path(self.temp.name) / 'relocated'
        folder.rename(outside)
        folder.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, '变成链接'):
            self.run_batch(operation='回滚最近一次')
        self.assertEqual(len(list(outside.glob('*.png'))), 1)

    def test_rollback_of_individual_child_does_not_restore_its_older_standalone_batch(self):
        a, b = self.images('A'), self.images('B')
        self.run_batch()
        run_image_dedup(RunContext('image_dedup', {'images_folder': a}, {'operation': '回滚最近一次'}, self.logs.append))
        result = self.run_batch(operation='回滚最近一次')
        self.assertEqual((result['folderCount'], result['restoredCount']), (1, 2))
        self.assertTrue(all(len(list(folder.glob('*.png'))) == 3 for folder in [a, b]))

    def test_migration_adds_switch_once_and_preserves_custom_defaults(self):
        path = Path(self.temp.name) / 'functions.json'
        item = FunctionCatalog(path).get('image_dedup')
        item['name'] = '我的去重'
        item['parameters'] = [p for p in item['parameters'] if p['id'] != 'recursive']
        next(p for p in item['parameters'] if p['id'] == 'dedup_mode')['default'] = '仅精确重复'
        path.write_text(json.dumps([item]))
        for _ in range(2):
            migrated = FunctionCatalog(path).get('image_dedup')
            self.assertEqual(migrated['name'], '我的去重')
            self.assertEqual(len([p for p in migrated['parameters'] if p['id'] == 'recursive']), 1)
            self.assertEqual(next(p for p in migrated['parameters'] if p['id'] == 'dedup_mode')['default'], '仅精确重复')
        changed = FunctionCatalog(path)
        migrated = changed.get('image_dedup')
        next(p for p in migrated['parameters'] if p['id'] == 'recursive')['default'] = True
        path.write_text(json.dumps([migrated]))
        self.assertTrue(next(p for p in FunctionCatalog(path).get('image_dedup')['parameters'] if p['id'] == 'recursive')['default'])
