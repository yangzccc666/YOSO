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
from backend.image_dedup import run_image_dedup
from backend.image_quality_dedup import AUTO_MODE, EXACT_MODE, verify_pair
from backend.task_manager import TaskCancelled


class QualityDedupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / 'images'
        self.folder.mkdir()

    def scene(self, phase=0):
        image = np.full((160, 192, 3), 50, np.uint8)
        cv2.rectangle(image, (10 + phase * 30, 35), (35 + phase * 30, 120), (200, 150, 90), -1)
        return image

    def write(self, index, image):
        path = self.folder / f'frame_{index:04d}.png'
        self.assertTrue(cv2.imwrite(str(path), image))
        return path

    def run_dedup(self, **parameters):
        return run_image_dedup(RunContext('image_dedup', {'images_folder': self.folder},
                                         {'dedup_mode': AUTO_MODE, **parameters}, lambda _: None))

    def test_many_similar_variable_repeats_keep_different_action_samples(self):
        rng = np.random.default_rng(42)
        for i, phase in enumerate([0, 1, 2, 3] * 6):
            noise = rng.integers(-1, 2, self.scene().shape)
            self.write(i, np.clip(self.scene(phase).astype(int) + noise, 0, 255).astype(np.uint8))
        result = self.run_dedup(minimum_keep_ratio=1, max_consecutive_skips=1, dedup_strength='not-used')
        self.assertEqual(result['keptCount'], 4)
        self.assertEqual(result['removedCount'], 20)
        self.assertTrue(all((self.folder / f'frame_{i:04d}.png').exists() for i in range(4)))
        manifest = json.loads(Path(result['backupManifest']).read_text())
        self.assertFalse(manifest['settings']['quantityQuotaApplied'])
        self.assertIsNone(manifest['settings']['similarityThreshold'])
        self.assertTrue(all(item['reference'] in {f'frame_{i:04d}.png' for i in range(4)} for item in manifest['removedImages']))
        self.assertTrue(any(item['matchType'] == 'near' for item in manifest['removedImages']))

    def test_not_adjacent_and_not_same_video_prefix(self):
        for i, phase in enumerate([0, 1, 2, 0]):
            path = self.write(i, self.scene(phase))
            path.rename(self.folder / f'video{i}_fps5_t000000000ms_f00000000.png')
        self.assertEqual(self.run_dedup()['removedCount'], 1)

    def test_slow_drift_does_not_chain_through_discarded_samples(self):
        base = self.scene()
        for i in range(11): self.write(i, np.clip(base.astype(int) + i * 2, 0, 255).astype(np.uint8))
        result = self.run_dedup()
        self.assertGreaterEqual(result['keptCount'], 3)
        self.assertGreater(result['removedCount'], 0)

    def test_legacy_continuous_mode_no_longer_honors_quantity_quotas(self):
        for i in range(30): self.write(i, self.scene())
        result = self.run_dedup(dedup_mode='连续相似帧去重', minimum_keep_ratio=1, max_consecutive_skips=1)
        self.assertEqual((result['keptCount'], result['removedCount']), (2, 28))

    def test_many_identical_images_have_no_floor_or_deletion_cap(self):
        for i in range(60): self.write(i, self.scene())
        result = self.run_dedup(minimum_keep_ratio='ignored', max_consecutive_skips='ignored')
        self.assertEqual((result['keptCount'], result['removedCount']), (1, 59))

    def test_one_unique_transient_event_survives_among_many_duplicates(self):
        for i in range(21): self.write(i, self.scene(3 if i == 10 else 0))
        result = self.run_dedup()
        self.assertEqual(result['keptCount'], 2)
        self.assertTrue((self.folder / 'frame_0010.png').exists())

    def test_local_small_defect_is_not_hidden_by_full_image_similarity(self):
        base = np.full((1024, 1024, 3), 60, np.uint8)
        altered = base.copy()
        altered[500:504, 500:504] = 220
        self.write(0, base)
        self.write(1, altered)
        self.write(2, base)
        result = self.run_dedup()
        self.assertEqual(result['keptCount'], 2)
        self.assertTrue((self.folder / 'frame_0001.png').exists())

    def test_exact_only_does_not_remove_sensor_noise(self):
        base = self.scene()
        noisy = base.copy()
        noisy[1, 1] += 1
        for i, image in enumerate([base, noisy, base]): self.write(i, image)
        result = self.run_dedup(dedup_mode=EXACT_MODE)
        self.assertEqual((result['keptCount'], result['removedCount']), (2, 1))

    def test_alpha_bit_depth_dimensions_and_lighting_changes_are_preserved(self):
        image = self.scene()
        alpha = np.dstack((image, np.full(image.shape[:2], 255, np.uint8)))
        alpha2 = alpha.copy()
        alpha2[0:20, 0:20, 3] = 0
        for i, value in enumerate([image, np.clip(image.astype(int) + 25, 0, 255).astype(np.uint8),
                                   image.astype(np.uint16) * 257, alpha, alpha2, image[:100]]):
            self.write(i, value)
        self.assertEqual(self.run_dedup()['removedCount'], 0)

    def test_annotation_class_count_and_geometry_are_protected(self):
        labels = ['0 .5 .5 .2 .2', '0 .6 .5 .2 .2', '1 .5 .5 .2 .2', '0 .5 .5 .2 .2\n0 .2 .2 .1 .1']
        for i, label in enumerate(labels): self.write(i, self.scene()).with_suffix('.txt').write_text(label)
        self.assertEqual(self.run_dedup()['removedCount'], 0)

    def test_invalid_labels_and_unreadable_images_are_never_deleted(self):
        for i in range(3): self.write(i, self.scene()).with_suffix('.txt').write_text('invalid annotation')
        (self.folder / 'broken.jpg').write_bytes(b'broken')
        result = self.run_dedup()
        self.assertEqual((result['removedCount'], result['unreadableCount']), (0, 4))
        self.assertEqual(len(list(self.folder.glob('*'))), 7)

    def test_shared_sidecar_is_rejected_before_moving_anything(self):
        path = self.write(0, self.scene())
        cv2.imwrite(str(path.with_suffix('.jpg')), self.scene())
        path.with_suffix('.txt').write_text('0 .5 .5 .2 .2')
        with self.assertRaisesRegex(ValueError, '共用标注'):
            self.run_dedup()

    def test_analysis_does_not_move_data_and_execute_rollback_restores_every_byte(self):
        for i in range(4): self.write(i, self.scene()).with_suffix('.txt').write_text('0 .5 .5 .2 .2')
        original = {p.name: p.read_bytes() for p in self.folder.iterdir()}
        preview = self.run_dedup(operation='仅分析（不移动文件）')
        self.assertEqual(preview['proposedRemovedCount'], 3)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.folder.iterdir()})
        self.assertTrue(Path(preview['analysisReport']).exists())
        self.assertEqual(self.run_dedup()['removedCount'], 3)
        self.assertEqual(self.run_dedup(operation='回滚最近一次')['restoredCount'], 3)
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.folder.iterdir()})

    def test_change_during_analysis_aborts_the_entire_plan(self):
        for i in range(3): self.write(i, self.scene())
        from backend.image_quality_dedup import read_original
        modified = False
        def read(path):
            nonlocal modified
            image = read_original(path)
            if path.name == 'frame_0002.png' and not modified:
                modified = True
                (self.folder / 'frame_0000.txt').write_text('0 .5 .5 .2 .2')
            return image
        with patch('backend.image_quality_dedup.read_original', side_effect=read), self.assertRaisesRegex(ValueError, '发生变化'):
            self.run_dedup()
        self.assertEqual(len(list(self.folder.glob('*.png'))), 3)

    def test_cancel_during_move_restores_data(self):
        for i in range(3): self.write(i, self.scene())
        event = threading.Event()
        replace = Path.replace
        def move(source, target):
            result = replace(source, target)
            if 'files' in target.parts: event.set()
            return result
        with patch.object(Path, 'replace', move), self.assertRaises(TaskCancelled):
            run_image_dedup(RunContext('image_dedup', {'images_folder': self.folder}, {}, lambda _: None, event))
        self.assertEqual(len(list(self.folder.glob('*.png'))), 3)

    def test_migration_removes_quotas_and_preserves_custom_fields(self):
        path = Path(self.temp.name) / 'functions.json'
        item = FunctionCatalog(path).get('image_dedup')
        item['parameters'].extend([{'id': 'minimum_keep_ratio', 'type': 'number', 'default': .9},
                                   {'id': 'max_consecutive_skips', 'type': 'number', 'default': 1}])
        field = next(p for p in item['parameters'] if p['id'] == 'dedup_mode')
        field['default'] = '完整工序循环去重'
        path.write_text(json.dumps([item]))
        for _ in range(2):
            updated = FunctionCatalog(path).get('image_dedup')
            fields = {p['id']: p for p in updated['parameters']}
            self.assertNotIn('minimum_keep_ratio', fields)
            self.assertNotIn('max_consecutive_skips', fields)
            self.assertIn(AUTO_MODE, fields['dedup_mode']['options'])
            self.assertEqual(fields['dedup_mode']['default'], '完整工序循环去重')
