import json
import io
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from backend.catalog import FunctionCatalog
from backend.handlers import RunContext
from backend.image_dedup import _backup_root, run_image_dedup
from backend.image_process_dedup import PROCESS_MODE, load_regions, preview_regions, region_for
from backend.task_manager import TaskCancelled


class ProcessDedupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / '场景'
        self.root.mkdir()
        self.logs = []
        self.region = [0., 0., .5, 1.]

    def config(self, regions=None):
        return json.dumps({'version': 1, 'root': str(self.root), 'regions': regions if regions is not None else {'.': self.region}})

    def scene(self, phase, index, abnormal=False, jitter=False):
        rng = np.random.default_rng(index + 500)
        image = np.full((160, 256, 3), 45, np.uint8)
        shift = index % 3 - 1 if jitter else 0
        cv2.rectangle(image, (12 + phase*29 + shift, 45), (34 + phase*29 + shift, 115), (190, 130, 75), -1)
        cv2.circle(image, (25 + phase*29, 28), 10, (60, 190, 220), -1)
        image[:, 128:] = rng.integers(0, 255, (160, 128, 3), dtype=np.uint8)
        image[:, :128] = np.clip(image[:, :128].astype(int) + rng.integers(-2, 3, (160, 128, 3)), 0, 255).astype(np.uint8)
        if abnormal:
            cv2.rectangle(image, (40, 120), (80, 150), (20, 20, 240), -1)
        return image

    def write(self, folder, index, phase=0, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'frame_{index:04d}.png'
        self.assertTrue(cv2.imwrite(str(path), self.scene(phase, index, **kwargs)))
        return path

    def run_dedup(self, **parameters):
        return run_image_dedup(RunContext('image_dedup', {'images_folder': self.root},
                              {'dedup_mode': PROCESS_MODE, 'process_regions': self.config(), **parameters}, self.logs.append))

    def test_background_changes_and_variable_cycle_lengths_do_not_block_process_duplicates(self):
        # Same actions, different dwell times/hand position and unrelated workstation activity.
        phases = [0, 0, 1, 1, 2, 2, 0, 1, 1, 1, 2, 0, 0, 0, 1, 2, 2, 2]
        for i, phase in enumerate(phases):
            self.write(self.root, i, phase, jitter=True)
        old = self.run_dedup(dedup_mode='差异优先去重（自动）', operation='仅分析（不移动文件）')
        self.assertEqual(old['proposedRemovedCount'], 0)
        result = self.run_dedup()
        self.assertGreaterEqual(result['removedCount'], 12)
        self.assertGreaterEqual(result['keptCount'], 3)
        manifest = json.loads(Path(result['backupManifest']).read_text())
        self.assertEqual(manifest['settings']['workstation']['region'], self.region)
        self.assertTrue(all((self.root / item['reference']).exists() for item in manifest['removedImages']))
        self.assertTrue(all(item['matchType'] == 'process' for item in manifest['removedImages']))

    def test_new_action_and_visible_abnormality_in_roi_are_preserved(self):
        for i, phase in enumerate([0, 1, 2, 0, 1, 2, 0, 1, 2]):
            self.write(self.root, i, phase, abnormal=i == 6)
        result = self.run_dedup()
        self.assertGreater(result['removedCount'], 0)
        self.assertTrue((self.root / 'frame_0006.png').exists())
        self.assertTrue(all((self.root / f'frame_{i:04d}.png').exists() for i in range(3)))

    def test_bbox_jitter_allowed_but_target_class_and_count_changes_protected(self):
        labels = ['0 .09 .5 .10 .3', '0 .095 .5 .10 .3', '1 .09 .5 .10 .3',
                  '0 .09 .5 .10 .3\n0 .35 .5 .1 .2', '0 .4 .5 .1 .3']
        for i, label in enumerate(labels):
            self.write(self.root, i).with_suffix('.txt').write_text(label)
        result = self.run_dedup()
        self.assertEqual(result['removedCount'], 1)
        self.assertFalse((self.root / 'frame_0001.png').exists())
        self.assertTrue(all((self.root / f'frame_{i:04d}.png').exists() for i in [0, 2, 3, 4]))

    def test_unrelated_workstation_labels_outside_roi_are_explicitly_ignored(self):
        for i in range(3):
            self.write(self.root, i).with_suffix('.txt').write_text(f'0 .09 .5 .1 .3\n{i+1} .8 .5 .1 .2')
        self.assertEqual(self.run_dedup()['removedCount'], 2)

    def test_invalid_or_non_detection_labels_and_broken_images_preserved(self):
        for i, label in enumerate(['invalid', '0 .1 .1 .2 .2 .3 .3', '0 .1 .1 -.2 .3']):
            self.write(self.root, i).with_suffix('.txt').write_text(label)
        (self.root / 'broken.jpg').write_bytes(b'bad image')
        result = self.run_dedup()
        self.assertEqual((result['removedCount'], result['unreadableCount']), (0, 4))

    def test_preview_execute_rollback_restores_all_image_and_sidecar_bytes(self):
        for i in range(5):
            self.write(self.root, i).with_suffix('.txt').write_text('0 .1 .5 .1 .3')
        snapshot = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = self.run_dedup(operation='仅分析（不移动文件）')
        self.assertEqual(result['proposedRemovedCount'], 4)
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.root.iterdir()})
        self.assertEqual(self.run_dedup()['removedCount'], 4)
        self.assertEqual(self.run_dedup(operation='回滚最近一次', process_regions='')['restoredCount'], 4)
        self.assertEqual(snapshot, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_recursive_missing_roi_rejected_before_any_directory_is_modified(self):
        for name in ['A', 'B']:
            for i in range(3): self.write(self.root / name, i)
        with self.assertRaisesRegex(ValueError, 'B 尚未框选'):
            self.run_dedup(recursive=True, process_regions=self.config({'A': self.region}))
        self.assertTrue(all(len(list((self.root / name).glob('*.png'))) == 3 for name in ['A', 'B']))
        self.assertFalse(_backup_root(self.root).exists())

    def test_recursive_each_workstation_uses_its_own_roi(self):
        for name in ['A', 'B']:
            for i in range(3): self.write(self.root / name, i)
        # B deliberately selects the unrelated moving region, which is NOT repeated.
        result = self.run_dedup(recursive=True, process_regions=self.config({'A': self.region, 'B': [.5, 0, 1, 1]}))
        self.assertEqual(result['folderResults'][0]['removedCount'], 2)
        self.assertEqual(result['folderResults'][1]['removedCount'], 0)
        self.assertEqual(self.run_dedup(recursive=True, operation='回滚最近一次')['restoredCount'], 2)

    def test_roi_validation_and_folder_scope(self):
        self.write(self.root, 0)
        for region in [[0, 0, 0, 1], [-.1, 0, .5, 1], [0, 0, 2, 1], [0, 0, float('nan'), 1], [False, 0, .5, 1]]:
            with self.assertRaises(ValueError): load_regions(self.config({'.': region}))
        with self.assertRaisesRegex(ValueError, '请先读取'):
            self.run_dedup(process_regions='')
        config = load_regions(self.config())
        with self.assertRaisesRegex(ValueError, '图片路径已改变'):
            region_for(config, self.root.parent)

    def test_preview_reads_actual_directory_without_writing_and_rejects_traversal(self):
        image = self.write(self.root / '工位1', 0)
        before = list(self.root.rglob('*'))
        result = preview_regions({'imagesFolder': str(self.root), 'recursive': True})
        self.assertEqual(result['folders'], ['工位1'])
        self.assertEqual(result['sampleName'], image.name)
        self.assertTrue(result['image'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(before, list(self.root.rglob('*')))
        with self.assertRaisesRegex(ValueError, '扫描范围'):
            preview_regions({'imagesFolder': str(self.root), 'recursive': True, 'folderKey': '..'})
        with self.assertRaisesRegex(ValueError, '没有可预览'):
            preview_regions({'imagesFolder': str(self.root / '工位1'), 'sampleName': '../secret.jpg'})

    def test_symlink_sidecars_and_symlink_preview_images_are_protected(self):
        a, b = self.write(self.root, 0), self.write(self.root, 1)
        outside = self.root.parent / 'outside.txt'
        outside.write_text('0 .1 .5 .1 .3')
        for p in [a, b]: p.with_suffix('.txt').symlink_to(outside)
        self.assertEqual(self.run_dedup()['removedCount'], 0)
        linked = self.root / 'linked.png'
        linked.symlink_to(a)
        with self.assertRaises(ValueError):
            preview_regions({'imagesFolder': str(self.root), 'sampleName': linked.name})

    def test_new_mode_migration_preserves_explicit_mode_and_saved_roi(self):
        path = self.root.parent / 'functions.json'
        item = FunctionCatalog(path).get('image_dedup')
        next(p for p in item['parameters'] if p['id'] == 'dedup_mode')['default'] = '完整工序循环去重'
        item['parameters'] = [p for p in item['parameters'] if p['id'] not in {'process_regions', 'process_strength'}]
        path.write_text(json.dumps([item]))
        for _ in range(2):
            fields = {p['id']: p for p in FunctionCatalog(path).get('image_dedup')['parameters']}
            self.assertEqual(fields['dedup_mode']['default'], '完整工序循环去重')
            self.assertIn(PROCESS_MODE, fields['dedup_mode']['options'])
            self.assertEqual(fields['process_strength']['default'], '平衡（推荐）')

    def test_cancel_does_not_apply_planned_moves(self):
        for i in range(4): self.write(self.root, i)
        event = threading.Event()
        def log(message):
            if '建议移除' in message: event.set()
        with self.assertRaises(TaskCancelled):
            run_image_dedup(RunContext('image_dedup', {'images_folder': self.root},
                            {'dedup_mode': PROCESS_MODE, 'process_regions': self.config()}, log, event))
        self.assertEqual(len(list(self.root.glob('*.png'))), 4)

    def test_preview_api_requires_activation_and_surfaces_validation_errors(self):
        from backend import server
        def request(activated, error=None):
            body = json.dumps({'imagesFolder': str(self.root)}).encode()
            raw = b'POST /api/image-dedup/region-preview HTTP/1.0\r\nContent-Length: ' + str(len(body)).encode() + b'\r\n\r\n' + body
            output = bytearray()
            connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
            with patch.object(server.license_store, 'status', return_value={'activated': activated, 'role': 'user', 'message': 'activation required'}), \
                 patch('backend.image_process_dedup.preview_regions', return_value={'folders': ['.']}, side_effect=error) as preview:
                server.AppHandler(connection, ('127.0.0.1', 1), SimpleNamespace())
                if not activated: preview.assert_not_called()
            head, payload = bytes(output).split(b'\r\n\r\n', 1)
            return int(head.split()[1]), json.loads(payload)
        self.assertEqual(request(False)[0], 403)
        self.assertEqual(request(True), (200, {'folders': ['.']}))
        self.assertEqual(request(True, ValueError('bad folder'))[0], 400)
