import json
from pathlib import Path
import tempfile
import unittest

from backend.function_defaults import FunctionDefaultsStore


class FunctionDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'defaults.json'
        self.store = FunctionDefaultsStore(self.path)
        self.definition = {'id': 'video_frames', 'handlerId': 'video.extract_frames',
            'pathFields': [{'id': 'video_folder'}], 'parameters': [
                {'id': 'target_fps', 'type': 'number', 'default': 5},
                {'id': 'enabled', 'type': 'boolean', 'default': True},
                {'id': 'credential', 'type': 'password', 'default': ''}]}

    def test_defaults_survive_restart_preserve_empty_zero_false_and_are_independent(self):
        first = self.store.save(self.definition, {'paths': {'video_folder': '/数据/工位1\n/数据/工位2'},
            'parameters': {'target_fps': 20, 'enabled': False, 'max_images': '', 'offset': 0}})
        other = {**self.definition, 'id': 'another_function'}
        self.store.save(other, {'paths': {}, 'parameters': {'target_fps': 1}})
        restarted = FunctionDefaultsStore(self.path)
        self.assertEqual(restarted.get('video_frames'), first)
        self.assertEqual(restarted.get('another_function')['values']['parameters']['target_fps'], 1)
        self.assertEqual(first['values']['parameters']['offset'], 0)
        self.assertEqual(first['values']['parameters']['enabled'], False)
        self.assertEqual(first['values']['parameters']['max_images'], '')
        changed = self.store.save(self.definition, {'paths': {}, 'parameters': {'target_fps': 8}})
        self.assertNotEqual(first['revision'], changed['revision'])
        self.assertEqual(restarted.get('video_frames')['values']['parameters']['target_fps'], 8)

    def test_passwords_and_connection_tokens_never_enter_defaults_file(self):
        result = self.store.save(self.definition, {'paths': {}, 'parameters': {
            'credential': 'sensitive-credential', 'remote_password': 'sensitive-password',
            'remote_connection_token': 'sensitive-token', 'remember_password': True,
            'star_models': '[{"path":"model.star","labels":"MARK"}]'}})
        content = self.path.read_text()
        self.assertNotIn('sensitive-', content)
        self.assertTrue(result['values']['parameters']['remember_password'])
        self.assertIn('star_models', result['values']['parameters'])

    def test_training_snapshot_saves_editable_values_but_requires_device_each_time(self):
        definition = {**self.definition, 'handlerId': 'yolo.train'}
        result = self.store.save(definition, {'paths': {}, 'parameters': {'device': '3', 'training_parameters': json.dumps({
            'epochs': 350, 'batch': 8, 'device': '3', 'remote_password': 'sensitive-password', 'data': '/dataset/data.yaml'})}})
        parameters = result['values']['parameters']
        self.assertEqual(parameters['device'], '')
        training = json.loads(parameters['training_parameters'])
        self.assertEqual(training['device'], '')
        self.assertEqual(training['epochs'], 350)
        self.assertEqual(training['data'], '/dataset/data.yaml')
        self.assertNotIn('sensitive-password', self.path.read_text())

    def test_invalid_input_leaves_previously_saved_configuration_unchanged(self):
        self.store.save(self.definition, {'paths': {}, 'parameters': {'target_fps': 2}})
        previous = self.path.read_bytes()
        for raw in [{'paths': []}, {'paths': {'bad': 42}, 'parameters': {}},
                    {'paths': {}, 'parameters': {'target_fps': float('nan')}},
                    {'paths': {}, 'parameters': {'target_fps': {'nested': 3}}}]:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.store.save(self.definition, raw)
            self.assertEqual(self.path.read_bytes(), previous)


if __name__ == '__main__':
    unittest.main()
