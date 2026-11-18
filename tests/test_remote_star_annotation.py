import copy
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.auto_annotation import run_auto_annotation
from backend.auto_annotation_worker import run
from backend.catalog import AUTO_ANNOTATE_FUNCTION, FunctionCatalog
from backend.handlers import RunContext
from backend.function_defaults import FunctionDefaultsStore
from backend.remote_inference import RemoteInferenceManager
from backend.remote_star_annotation import StarSession, model_labels, run_remote_annotation, validate_prediction
from backend.task_manager import TaskCancelled


def prediction(identity='one', **extra):
    return dict(type='result', id=identity, shape=[20, 30],
                detections=[{'class_id': 1, 'box': [-2.5, 3.2, 40.1, 18.3], 'score': .8}], **extra)


class ProtocolTests(unittest.TestCase):
    def test_mapping_and_filter(self):
        result = validate_prediction(prediction(), 'one', ['MARK', '工件 & 合格'], [])
        self.assertEqual(result.boxes.cls.tolist(), [1])
        self.assertEqual(result.orig_shape, [20, 30])
        self.assertEqual(validate_prediction(prediction(), 'one', ['a', 'b'], [0]).boxes.cls.tolist(), [])

    def test_bad_results_never_become_labels(self):
        cases = [dict(id='another'), dict(type='ready'), dict(shape=[0, 30]), dict(shape=[20.5, 30]),
                 dict(detections=None), dict(detections=[{'class_id': 5}]),
                 dict(detections=[{'class_id': 1, 'box': [0, 0, float('nan'), 10], 'score': .8}]),
                 dict(detections=[{'class_id': 1, 'box': [10, 0, 1, 10], 'score': .8}])]
        for fields in cases:
            with self.subTest(fields=fields), self.assertRaises(RuntimeError):
                data = prediction()
                data.update(fields)
                validate_prediction(data, 'one', ['a', 'b'], [])

    def test_remote_image_error(self):
        with self.assertRaisesRegex(ValueError, '无法解码'):
            validate_prediction({'type': 'result', 'id': 'one', 'error': '无法解码'}, 'one', ['a'], [])

    def test_ordered_labels_preserve_duplicate_class_names(self):
        self.assertEqual(model_labels('0: MARK\n1: 合格\n2: MARK', 'unused'), ['MARK', '合格', 'MARK'])
        self.assertEqual(model_labels('MARK，合格', 'unused'), ['MARK', '合格'])
        with self.assertRaisesRegex(ValueError, '连续'):
            model_labels('1: wrong', 'unused')
        with patch('backend.remote_star_annotation.detect_star_labels', return_value={'labels': ['a']}) as detect:
            self.assertEqual(model_labels('', 'a.star'), ['a'])
            detect.assert_called_once_with('a.star')


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.weights = self.folder / '模型.star'
        self.weights.write_bytes(b'model')
        self.image = self.folder / '中文.jpg'
        self.image.write_bytes(b'original-image')
        self.context = RunContext('auto_annotation', {'image_folder': self.folder, 'weights_file': self.weights},
                                  {'star_labels': 'a\n工件 & 合格'}, MagicMock(), threading.Event())
        self.config = dict(folder=str(self.folder), weights=str(self.weights), conf=.5, iou=.7, imgsz=640,
                           classes=[], recursive=True, existing='跳过', empty_xml=True, device='')

    def session(self):
        return StarSession(self.context, self.config, ['a', '工件 & 合格'])

    def test_real_batch_policies_with_remote_predictions(self):
        session = self.session()
        session.sftp = MagicMock()
        session.channel = MagicMock()
        session.receive = lambda *a: prediction(json.loads(session.channel.sendall.call_args.args[0])['id'])
        result = run(self.config, session, self.context.report, session.check)
        self.assertEqual((result['written'], result['objects']), (1, 1))
        xml = self.image.with_suffix('.xml')
        root = ET.parse(xml).getroot()
        self.assertEqual(root.findtext('filename'), '中文.jpg')
        self.assertEqual(root.findtext('object/name'), '工件 & 合格')
        self.assertEqual([root.findtext('object/bndbox/' + k) for k in ('xmin', 'ymin', 'xmax', 'ymax')], ['0', '3', '30', '19'])
        self.assertEqual(self.image.read_bytes(), b'original-image')
        session.sftp.reset_mock()
        self.assertEqual(run(self.config, session)['skipped'], 1)
        session.sftp.put.assert_not_called()
        xml.write_text('old-xml')
        self.config['existing'] = '备份后覆盖'
        result = run(self.config, session)
        self.assertEqual((Path(result['backup']) / xml.name).read_text(), 'old-xml')
        self.assertEqual(ET.parse(xml).findtext('object/name'), '工件 & 合格')

    def test_empty_policy_preserves_old_xml(self):
        session = self.session()
        session.sftp, session.channel = MagicMock(), MagicMock()
        def receive(*_):
            data = prediction(json.loads(session.channel.sendall.call_args.args[0])['id'])
            data['detections'] = []
            return data
        session.receive = receive
        xml = self.image.with_suffix('.xml')
        xml.write_text('old-xml')
        self.config.update(existing='备份后覆盖', empty_xml=False)
        self.assertEqual(run(self.config, session)['skipped'], 1)
        self.assertEqual(xml.read_text(), 'old-xml')
        self.config['empty_xml'] = True
        result = run(self.config, session)
        self.assertEqual(result['empty'], 1)
        self.assertEqual(ET.parse(xml).findall('object'), [])

    def test_stop_during_inference_does_not_write_or_swallow_cancellation(self):
        session = self.session()
        session.sftp, session.channel = MagicMock(), MagicMock()
        def receive(*_):
            self.context.stop_event.set()
            return prediction(json.loads(session.channel.sendall.call_args.args[0])['id'])
        session.receive = receive
        with self.assertRaises(TaskCancelled):
            run(self.config, session, check_cancelled=session.check)
        self.assertFalse(self.image.with_suffix('.xml').exists())

    def test_protocol_failure_aborts_instead_of_creating_empty_xml(self):
        session = self.session()
        session.sftp, session.channel = MagicMock(), MagicMock()
        session.receive = lambda *_: prediction('wrong-image')
        with self.assertRaisesRegex(RuntimeError, '不匹配'):
            run(self.config, session, check_cancelled=session.check)
        self.assertFalse(self.image.with_suffix('.xml').exists())

    def test_fragmented_json_and_runtime_diagnostics(self):
        session = self.session()
        session.channel = MagicMock(closed=False)
        session.channel.recv_stderr_ready.return_value = True
        session.channel.recv_stderr.return_value = b'CUDA ready\n'
        session.channel.recv_ready.return_value = True
        session.channel.recv.side_effect = [b'{"type":"rea', b'dy","version":1}\n']
        self.assertEqual(session.receive(), {'type': 'ready', 'version': 1})
        self.context.report.assert_any_call('CUDA ready')

    def test_receive_reports_fatal_exit_and_timeout(self):
        session = self.session()
        session.channel = MagicMock(closed=False)
        session.channel.recv_stderr_ready.return_value = False
        session.channel.recv_ready.return_value = False
        session.channel.exit_status_ready.return_value = False
        with self.assertRaisesRegex(RuntimeError, '超时'):
            session.receive(timeout=0)
        session.buffer = b'{"type":"fatal","error":"missing TensorRT"}\n'
        with self.assertRaisesRegex(RuntimeError, 'missing TensorRT'):
            session.receive()
        session.buffer = b'not-json\n'
        with self.assertRaisesRegex(RuntimeError, '有效 JSON'):
            session.receive()
        session.channel.closed = True
        with self.assertRaisesRegex(RuntimeError, '退出'):
            session.receive()

    def test_cleanup_is_scoped_to_uploaded_files_and_exact_worker(self):
        session = self.session()
        session.client, session.sftp, session.channel = MagicMock(), MagicMock(), MagicMock()
        session.created, session.python = True, '/python with space/bin/python'
        session.sftp.open.return_value.__enter__.return_value.read.return_value = b'12345'
        stdout = MagicMock()
        stdout.channel.exit_status_ready.return_value = True
        session.client.exec_command.return_value = (None, stdout, None)
        session.close()
        removed = [c.args[0] for c in session.sftp.remove.call_args_list]
        self.assertEqual(removed, [session.root + '/' + name for name in session.files])
        command = session.client.exec_command.call_args.args[0]
        self.assertIn('/proc/%d/cmdline', command)
        self.assertIn(session.root + '/remote_star_annotation_worker.py', command)
        self.assertNotIn('pkill', command)
        session.channel.shutdown_write.assert_called_once()
        session.client.close.assert_called_once()

    def test_start_uploads_runtime_model_and_no_local_image_batch(self):
        session = self.session()
        paramiko = MagicMock()
        client = paramiko.SSHClient.return_value
        client.get_transport.return_value.open_session.return_value = MagicMock()
        with patch('backend.remote_star_annotation.resolve_tested_yolo_connection', return_value=dict(host='box', port=22, username='u', password='secret')), \
             patch.object(RemoteInferenceManager, '_load_paramiko', return_value=paramiko), \
             patch.object(session, 'receive', return_value={'type': 'ready', 'version': 1, 'input_shape': [1, 3, 640, 640]}):
            session.start()
        uploaded = [c.args[1] for c in session.sftp.put.call_args_list]
        self.assertEqual(len(uploaded), 3)
        self.assertTrue(uploaded[-1].endswith('/model.star'))
        self.assertFalse(any(v.endswith('/image.bin') for v in uploaded))

    def test_coordinator_closes_on_start_failure(self):
        with patch('backend.remote_star_annotation.StarSession') as factory:
            factory.return_value.start.side_effect = RuntimeError('No CUDA')
            with self.assertRaisesRegex(RuntimeError, 'No CUDA'):
                run_remote_annotation(self.context, self.config)
            factory.return_value.close.assert_called_once()

    def test_preflight_rejects_collisions_before_uploading_a_model(self):
        self.image.with_suffix('.png').touch()
        with patch('backend.remote_star_annotation.StarSession') as factory:
            with self.assertRaisesRegex(ValueError, '同名 XML'):
                run_remote_annotation(self.context, self.config)
            factory.assert_not_called()

    def test_mode_dispatch_never_resolves_local_python_for_star(self):
        self.context.parameters.update(annotation_mode='远程 STAR 标注', imgsz='not-used')
        with patch('backend.remote_star_annotation.run_remote_annotation', return_value={'message': 'ok'}) as remote, \
             patch('backend.auto_annotation._resolve_python') as python:
            self.assertEqual(run_auto_annotation(self.context), {'message': 'ok'})
            python.assert_not_called()
            self.assertEqual(remote.call_args.args[1]['weights'], str(self.weights))


class MigrationTests(unittest.TestCase):
    def test_existing_feature_migrates_without_resetting_custom_defaults(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'functions.json'
            item = copy.deepcopy(AUTO_ANNOTATE_FUNCTION)
            item['name'] = '自定义标注'
            item['parameters'] = [v for v in item['parameters'] if v['id'] in ('conf', 'imgsz', 'recursive')]
            item['parameters'][0]['default'] = .8
            defaults = FunctionDefaultsStore(Path(temp) / 'defaults.json')
            record = defaults.save(item, {'paths': {'image_folder': '/my/dataset'}, 'parameters': {'conf': .9}})
            path.write_text(json.dumps([item]))
            for _ in range(2):
                loaded = next(v for v in FunctionCatalog(path).list() if v['id'] == 'auto_annotation')
                self.assertEqual(loaded['name'], '自定义标注')
                fields = {v['id']: v for v in loaded['parameters']}
                self.assertEqual(fields['conf']['default'], .8)
                self.assertEqual(fields['annotation_mode']['default'], '本地 PT 标注')
                self.assertEqual(fields['remote_host']['visibleWhen']['equals'], '远程 STAR 标注')
                self.assertEqual(fields['imgsz']['visibleWhen']['equals'], '本地 PT 标注')
                self.assertEqual(len(fields), len(loaded['parameters']))
                self.assertEqual(defaults.get(item['id']), record)


class UploadedWorkerTests(unittest.TestCase):
    """Run the actual uploaded protocol executable, substituting only CUDA runtime."""
    def invoke(self, num_classes=2, corrupt=False):
        import cv2
        import numpy as np
        from backend import remote_star_annotation_worker
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(remote_star_annotation_worker.__file__, root / 'remote_star_annotation_worker.py')
            (root / 'config.json').write_text(json.dumps(dict(conf=.5, iou=.7, num_classes=num_classes, cuda_backend='auto')))
            image = b'not-an-image' if corrupt else cv2.imencode('.png', np.zeros((20, 30, 3), np.uint8))[1].tobytes()
            (root / 'image.bin').write_bytes(image)
            (root / 'bundled_run_star_video.py').write_text('''import os
import numpy as np
def read_engine_bytes(path): return b'engine', 0
class TensorRTRunner:
    input_shape = (1, 3, 640, 640)
    def __init__(self, *a, **k): os.write(1, b'NATIVE LOAD\\n')
    def infer(self, tensor):
        os.write(1, b'NATIVE INFER\\n')
        return {}
    def close(self): os.write(1, b'NATIVE CLOSE\\n')
def preprocess(frame, shape): return frame, 1, (0, 0)
def pair_yolo26_outputs(*a, **k): return [(None, np.zeros((1, 1, 2)))]
def decode_yolo26(*a, **k):
    return np.array([[1., 2., 10., 15.]]), np.array([.9]), np.array([1])
''')
            process = subprocess.run([sys.executable, '-u', str(root / 'remote_star_annotation_worker.py')],
                                     input='{"id":"first"}\n{"id":"second"}\n', capture_output=True, text=True, timeout=20)
            records = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertTrue((root / 'pid').is_file())
            return process, records

    def test_load_once_infer_every_image_and_native_logs_do_not_corrupt_json(self):
        process, records = self.invoke()
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual([v['type'] for v in records], ['ready', 'result', 'result'])
        self.assertEqual([v['id'] for v in records[1:]], ['first', 'second'])
        self.assertEqual(records[1]['shape'], [20, 30])
        self.assertEqual(records[1]['detections'][0]['class_id'], 1)
        self.assertEqual(process.stderr.count('NATIVE LOAD'), 1)
        self.assertEqual(process.stderr.count('NATIVE INFER'), 2)
        self.assertIn('NATIVE CLOSE', process.stderr)

    def test_class_count_mismatch_fails_before_xml_results(self):
        process, records = self.invoke(num_classes=1)
        self.assertNotEqual(process.returncode, 0)
        self.assertEqual([v['type'] for v in records], ['ready', 'fatal'])
        self.assertIn('数量不一致', records[-1]['error'])
        self.assertIn('NATIVE CLOSE', process.stderr)

    def test_bad_images_are_not_empty_annotations(self):
        process, records = self.invoke(corrupt=True)
        self.assertEqual(process.returncode, 0)
        self.assertTrue(all('error' in r and 'detections' not in r for r in records[1:]))
