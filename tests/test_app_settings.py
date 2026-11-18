import tempfile
import json
import sys
import unittest
from pathlib import Path

from backend.app_settings import AppSettingsStore


class AppSettingsTests(unittest.TestCase):
    def test_default_only_contains_xanylabeling_and_legacy_editor_is_hidden_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            store = AppSettingsStore(path)
            self.assertEqual([app['id'] for app in store.payload()['externalApps']], ['xanylabeling'])
            legacy = {'externalApps': {'xanylabeling': {'executable': '/opt/python', 'arguments': 'app.py'},
                                       'antigravity': {'executable': '/opt/editor'}}, 'aiPython': '/opt/ai'}
            path.write_text(json.dumps(legacy))
            payload = store.payload()
            self.assertEqual(len(payload['externalApps']), 1)
            self.assertEqual(payload['externalApps'][0]['executable'], '/opt/python')
            self.assertEqual(json.loads(path.read_text()), legacy)
            store.update(payload)
            self.assertEqual(json.loads(path.read_text())['hiddenLegacyExternalApps']['antigravity']['executable'], '/opt/editor')
            self.assertEqual(store.payload()['aiPython'], '/opt/ai')

    def test_custom_tools_names_descriptions_and_order_persist_and_delete_stays_deleted(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            store = AppSettingsStore(path)
            apps = [{'id': 'tool_a', 'name': '图片查看器', 'description': '检查图片', 'executable': '/opt/viewer'},
                    {'id': 'xanylabeling', 'name': '我的标注工具', 'arguments': 'app.py'}]
            saved = store.update({'externalApps': apps})
            self.assertEqual([app['name'] for app in saved['externalApps']], ['图片查看器', '我的标注工具'])
            self.assertEqual(AppSettingsStore(path).payload(), saved)
            store.update({'externalApps': []})
            self.assertEqual(AppSettingsStore(path).payload()['externalApps'], [])
            with self.assertRaisesRegex(ValueError, '未配置或已删除'):
                store.launch('tool_a')

    def test_invalid_ids_duplicate_and_empty_names_do_not_overwrite_saved_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / 'settings.json')
            store.update({'externalApps': []})
            before = store.storage_file.read_bytes()
            for apps in [[{'id': '../outside', 'name': 'bad'}], [{'id': '/abs', 'name': 'bad'}],
                         [{'id': 'a', 'name': ''}], [{'id': 'a', 'name': 'a\nb'}],
                         [{'id': 'a', 'name': 'one'}, {'id': 'a', 'name': 'two'}],
                         [{'id': 'a', 'name': 'a', 'executable': 'bad\x00path'}], [None]]:
                with self.subTest(apps=apps), self.assertRaises(ValueError):
                    store.update({'externalApps': apps})
                self.assertEqual(store.storage_file.read_bytes(), before)
            with self.assertRaisesRegex(ValueError, '未知'):
                store.launch('../outside')

    def test_custom_python_tool_can_launch_with_separate_arguments_without_shell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = AppSettingsStore(root / 'settings.json')
            store.update({'externalApps': [{'id': 'custom_python', 'name': '测试程序',
                                          'executable': sys.executable, 'workingDirectory': directory,
                                          'arguments': '-c\nimport sys; print(sys.argv[1])\nfile with spaces;$(literal)'}]})
            result = store.launch('custom_python')
            self.assertIn('测试程序', result['message'])
            self.assertEqual((root / 'logs/custom_python.log').read_text().strip(), 'file with spaces;$(literal)')

    def test_settings_are_saved_per_store_and_merged_with_app_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = AppSettingsStore(Path(directory) / "settings.json")
            saved = store.update({"externalApps": [{
                "id": "xanylabeling", "executable": "/opt/python", "workingDirectory": "/workspace",
                "arguments": "app.py", "condaEnvironment": "xlabel", "condaExecutable": "/opt/conda",
            }]})
            app = next(item for item in saved["externalApps"] if item["id"] == "xanylabeling")
            self.assertEqual(app["executable"], "/opt/python")
            self.assertEqual(app["name"], "X-AnyLabeling")
            self.assertEqual(saved["configFile"], str(Path(directory) / "settings.json"))

    def test_configured_program_launches_without_machine_specific_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / "sample-tool"
            executable.write_text("#!/bin/sh\nprintf 'started:%s' \"$1\"\n", encoding="utf-8")
            executable.chmod(0o755)
            store = AppSettingsStore(root / "settings.json")
            store.update({"externalApps": [{
                "id": "antigravity", "executable": str(executable), "workingDirectory": str(root),
                "arguments": "portable", "condaEnvironment": "", "condaExecutable": "",
            }]})
            result = store.launch("antigravity")
            self.assertIn("已发送", result["message"])
            self.assertEqual((root / "logs" / "antigravity.log").read_text(encoding="utf-8"), "started:portable")


if __name__ == "__main__":
    unittest.main()
