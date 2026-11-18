from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import desktop_input_method as ime


class DesktopInputMethodTests(unittest.TestCase):
    def test_non_linux_does_not_override_native_input_method(self):
        with patch.object(ime.sys, "platform", "win32"), patch.dict(os.environ, {
            "QT_IM_MODULE": "fcitx", "PROCESSING_VIEW_QT_IM_MODULE": "xim"}, clear=True):
            self.assertEqual(ime.configure_input_method(), ime.InputMethodReport())
            self.assertEqual(os.environ["QT_IM_MODULE"], "fcitx")

    def test_explicit_ibus_is_respected(self):
        with patch.object(ime.sys, "platform", "linux"), patch.dict(os.environ, {
            "QT_IM_MODULE": "fcitx", "PROCESSING_VIEW_QT_IM_MODULE": "ibus"}, clear=True):
            report = ime.configure_input_method()
            self.assertEqual(report.module, "ibus")
            self.assertEqual(os.environ["QT_IM_MODULE"], "ibus")

    def test_wayland_default_is_not_replaced_using_xmodifiers_alone(self):
        with patch.object(ime.sys, "platform", "linux"), patch.dict(os.environ, {
            "XDG_SESSION_TYPE": "wayland", "XMODIFIERS": "@im=fcitx"}, clear=True):
            self.assertEqual(ime.configure_input_method(), ime.InputMethodReport())
            self.assertNotIn("QT_IM_MODULE", os.environ)

    def test_xim_override_does_not_claim_fcitx_support(self):
        with patch.object(ime.sys, "platform", "linux"), patch.dict(os.environ, {
            "PROCESSING_VIEW_QT_IM_MODULE": "xim"}, clear=True):
            report = ime.configure_input_method()
            self.assertIn("不是搜狗/Fcitx", report.warning)
            self.assertNotIn("已加载", report.describe())

    def test_private_manifest_rejects_stale_qt_arch_and_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plugin = root / "platforminputcontexts" / "fcitx.so"
            plugin.parent.mkdir()
            plugin.write_bytes(b"fixture")
            manifest = {"qt": "6.9.2", "arch": "testarch", "plugin": plugin.name,
                        "sha256": hashlib.sha256(plugin.read_bytes()).hexdigest()}
            def save():
                (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            save()
            self.assertEqual(ime.validate_private_plugin(root, "6.9.2", "testarch"), plugin)
            with self.assertRaisesRegex(ValueError, "不匹配"):
                ime.validate_private_plugin(root, "6.10.0", "testarch")
            with self.assertRaisesRegex(ValueError, "不匹配"):
                ime.validate_private_plugin(root, "6.9.2", "other")
            manifest["plugin"] = "../outside.so"
            save()
            with self.assertRaisesRegex(ValueError, "文件名无效"):
                ime.validate_private_plugin(root, "6.9.2", "testarch")
            manifest["plugin"] = plugin.name
            manifest["sha256"] = "bad"
            save()
            with self.assertRaisesRegex(ValueError, "校验失败"):
                ime.validate_private_plugin(root, "6.9.2", "testarch")

    def _qt_mocks(self, temporary: str):
        return (patch("PySide6.QtCore.QLibraryInfo.path", return_value=temporary),
                patch("PySide6.QtCore.QCoreApplication.libraryPaths", return_value=[]),
                patch("PySide6.QtCore.QCoreApplication.addLibraryPath"),
                patch("desktop_input_method.private_plugin_root", return_value=Path(temporary) / "private"))

    def test_missing_plugin_never_falls_back_to_xim(self):
        with tempfile.TemporaryDirectory() as temporary:
            qpath, paths, addpath, private = self._qt_mocks(temporary)
            with qpath, paths, addpath as added, private, patch.dict(os.environ, {
                "QT_IM_MODULE": "fcitx", "XMODIFIERS": "@im=fcitx", "CUDA_VISIBLE_DEVICES": "3"}, clear=True):
                report = ime.configure_input_method()
                self.assertIn("缺少可加载", report.warning)
                self.assertEqual(os.environ["QT_IM_MODULE"], "fcitx")
                self.assertEqual(os.environ["XMODIFIERS"], "@im=fcitx")
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "3")
                self.assertNotIn("QT_PLUGIN_PATH", os.environ)
                added.assert_not_called()

    def test_working_private_plugin_is_registered_before_app(self):
        with tempfile.TemporaryDirectory() as temporary:
            qpath, paths, addpath, private = self._qt_mocks(temporary)
            root = Path(temporary) / "private"
            root.mkdir()
            plugin = root / "platforminputcontexts" / "fcitx.so"
            with qpath, paths, addpath as added, private, \
                 patch.object(ime, "validate_private_plugin", return_value=plugin), \
                 patch.object(ime, "load_fcitx_plugin", return_value=(True, "", ("fcitx",))), \
                 patch.dict(os.environ, {"QT_IM_MODULE": "fcitx5", "GTK_IM_MODULE": "fcitx"}, clear=True):
                report = ime.configure_input_method()
                self.assertEqual(report.plugin, str(plugin))
                self.assertFalse(report.warning)
                self.assertEqual(os.environ["QT_IM_MODULE"], "fcitx")
                self.assertEqual(os.environ["GTK_IM_MODULE"], "fcitx")
                added.assert_called_once_with(str(root))

    def test_incompatible_native_plugin_warns_instead_of_reporting_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            qpath, paths, addpath, private = self._qt_mocks(temporary)
            directory = Path(temporary) / "platforminputcontexts"
            directory.mkdir()
            (directory / "libfcitx.so").touch()
            with qpath, paths, addpath, private, \
                 patch.object(ime, "load_fcitx_plugin", return_value=(False, "Qt_6_PRIVATE_API mismatch", ("fcitx",))), \
                 patch.dict(os.environ, {"QT_IM_MODULE": "fcitx"}, clear=True):
                report = ime.configure_input_method()
                self.assertIn("Qt_6_PRIVATE_API", report.warning)
                self.assertFalse(report.plugin)

    def test_recover_previous_xim_fallback_when_fcitx_is_available(self):
        with tempfile.TemporaryDirectory() as temporary:
            qpath, paths, addpath, private = self._qt_mocks(temporary)
            directory = Path(temporary) / "platforminputcontexts"
            directory.mkdir()
            (directory / "libfcitx.so").touch()
            with qpath, paths, addpath, private, \
                 patch.object(ime, "load_fcitx_plugin", return_value=(True, "", ("fcitx",))), \
                 patch.dict(os.environ, {"QT_IM_MODULE": "xim", "XMODIFIERS": "@im=fcitx"}, clear=True):
                self.assertEqual(ime.configure_input_method().module, "fcitx")
                self.assertEqual(os.environ["QT_IM_MODULE"], "fcitx")

    def test_compose_alias_does_not_count_as_a_chinese_plugin(self):
        loader = Mock()
        loader.metaData.return_value = {"MetaData": {"Keys": ["compose", "xim"]}}
        with patch("PySide6.QtCore.QPluginLoader", return_value=loader):
            loaded, error, _ = ime.load_fcitx_plugin(Path("compose.so"))
        self.assertFalse(loaded)
        self.assertIn("不是 Fcitx", error)
        loader.load.assert_not_called()

    def test_service_probe_is_read_only_and_reports_unreachable_bus(self):
        bus = Mock()
        bus.isConnected.return_value = False
        bus.lastError.return_value.message.return_value = "permission denied"
        with patch("PySide6.QtDBus.QDBusConnection.sessionBus", return_value=bus):
            self.assertIn("permission denied", ime.fcitx_service_warning())
        bus.interface.assert_not_called()

    def test_service_probe_checks_current_display_without_starting_services(self):
        bus = Mock()
        bus.isConnected.return_value = True
        reply = bus.interface.return_value.isServiceRegistered.return_value
        reply.isValid.return_value = True
        reply.value.return_value = True
        with patch("PySide6.QtDBus.QDBusConnection.sessionBus", return_value=bus), \
             patch.dict(os.environ, {"DISPLAY": ":1.0"}, clear=True):
            self.assertEqual(ime.fcitx_service_warning(), "")
        bus.interface.return_value.isServiceRegistered.assert_called_once_with("org.fcitx.Fcitx-1")
        bus.interface.return_value.startService.assert_not_called()


if __name__ == "__main__":
    unittest.main()
