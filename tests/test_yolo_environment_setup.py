import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'setup/yolo-runtime'
spec = importlib.util.spec_from_file_location('yolo_environment_setup', SOURCE / 'install_yolo_env.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class YoloEnvironmentSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='yolo-setup-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.python = self.root / 'python.exe'
        self.python.touch()
        (self.root / 'requirements-yolo.txt').write_text('ultralytics>=8.4,<9\n')
        self.info = {'python': str(self.python), 'torch': '2.test', 'ultralytics': '8.4.test', 'cudaAvailable': False}
        self.commands = []
        self.existing = False

    def fake_run(self, command, capture=False):
        self.commands.append(command)
        if command[1:4] == ['env', 'list', '--json']:
            return SimpleNamespace(stdout=json.dumps({'envs': ['/conda/envs/yolo26'] if self.existing else []}))
        return SimpleNamespace(stdout='other output\n' + installer.RESULT_MARKER + json.dumps(self.info) + '\n')

    def test_new_environment_uses_python311_cpu_index_then_runtime_requirements(self):
        with patch.object(installer, 'ROOT', self.root), patch.object(installer, 'run', self.fake_run):
            result = installer.install('/conda', 'yolo26')
        self.assertEqual(result, self.info)
        self.assertIn(['/conda', 'create', '-y', '-n', 'yolo26', 'python=3.11', 'pip'], self.commands)
        torch = next(command for command in self.commands if installer.CPU_INDEX in command)
        self.assertIn('torch>=2.2,<3', torch)
        requirements = next(command for command in self.commands if '-r' in command)
        self.assertLess(self.commands.index(torch), self.commands.index(requirements))
        self.assertTrue(any(command[-1] == 'check' for command in self.commands))
        self.assertFalse(any('remove' in command for command in self.commands))

    def test_healthy_existing_environment_is_verified_without_installs(self):
        self.existing = True
        with patch.object(installer, 'ROOT', self.root), patch.object(installer, 'run', self.fake_run):
            self.assertEqual(installer.install('/conda'), self.info)
        self.assertEqual(len(self.commands), 2)
        self.assertFalse(any('install' in command or 'create' in command for command in self.commands))

    def test_broken_existing_environment_is_not_modified_without_explicit_repair(self):
        self.existing = True
        with patch.object(installer, 'ROOT', self.root), patch.object(installer, 'run', self.fake_run), \
             patch.object(installer, 'verify', side_effect=ValueError('broken')):
            with self.assertRaisesRegex(ValueError, '--env yolo26_platform'):
                installer.install('/conda')
        self.assertEqual(len(self.commands), 1)

    def test_explicit_repair_updates_packages_but_never_recreates_environment(self):
        self.existing = True
        with patch.object(installer, 'ROOT', self.root), patch.object(installer, 'run', self.fake_run):
            installer.install('/conda', repair=True)
        self.assertFalse(any('create' in command or 'remove' in command for command in self.commands))
        self.assertTrue(any('install' in command for command in self.commands))

    def test_base_root_and_invalid_names_are_rejected_before_any_command(self):
        with patch.object(installer, 'run') as run:
            for name in ('base', 'BASE', 'root', '../env', 'name with spaces', 'x&whoami', ''):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    installer.install('/conda', name)
            run.assert_not_called()

    def test_installation_failure_stops_without_false_success_or_removal(self):
        def fail_on_torch(command, capture=False):
            if installer.CPU_INDEX in command:
                raise subprocess.CalledProcessError(1, command, stderr='offline')
            return self.fake_run(command, capture)
        with patch.object(installer, 'ROOT', self.root), patch.object(installer, 'run', fail_on_torch):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.install('/conda')
        self.assertFalse(any('-r' in command or 'remove' in command for command in self.commands))

    def test_verification_checks_imports_nms_and_yolo26_without_downloaded_weights(self):
        self.assertIn('torchvision.ops.nms', installer.VERIFY_CODE)
        self.assertIn('yolo26n.yaml', installer.VERIFY_CODE)
        self.assertNotIn('yolo26n.pt', installer.VERIFY_CODE)
        with patch.object(installer, 'run', return_value=SimpleNamespace(stdout='no valid result')):
            with self.assertRaises(ValueError):
                installer.verify('/conda', 'yolo26')

    def test_installer_assets_and_batch_are_portable_and_packaged(self):
        batch = (SOURCE / 'install_yolo_env.bat').read_text()
        self.assertIn('%~dp0install_yolo_env.py', batch)
        self.assertIn('where conda.exe', batch)
        self.assertNotIn('/home/anson', batch)
        spec = (SOURCE.parents[1] / 'packaging/yolo_data_platform.spec').read_text()
        self.assertIn('AI-Environment-Setup', spec)
        self.assertTrue((SOURCE / 'requirements-yolo.txt').is_file())
