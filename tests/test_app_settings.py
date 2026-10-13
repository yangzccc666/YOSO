import tempfile
import unittest
from pathlib import Path

from backend.app_settings import AppSettingsStore


class AppSettingsTests(unittest.TestCase):
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
