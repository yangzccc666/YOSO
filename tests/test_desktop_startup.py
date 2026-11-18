from __future__ import annotations

import os
import unittest
from unittest.mock import Mock, patch

import desktop


class DesktopStartupTests(unittest.TestCase):
    def test_failed_linux_opengl_selects_both_software_renderers(self):
        probe = Mock(return_value=False)
        with patch.dict(os.environ, {}, clear=True), patch.object(desktop.sys, "platform", "linux"):
            mode = desktop.configure_desktop_rendering(probe)
            self.assertIn("OpenGL", mode)
            self.assertEqual(os.environ["QT_QUICK_BACKEND"], "software")
            self.assertEqual(os.environ["QTWEBENGINE_CHROMIUM_FLAGS"], "--disable-gpu")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            self.assertNotIn("LIBGL_ALWAYS_SOFTWARE", os.environ)
        probe.assert_called_once_with()

    def test_working_linux_opengl_keeps_acceleration(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(desktop.sys, "platform", "linux"):
            self.assertEqual(desktop.configure_desktop_rendering(lambda: True), "硬件加速模式")
            self.assertNotIn("QT_QUICK_BACKEND", os.environ)
            self.assertNotIn("QTWEBENGINE_CHROMIUM_FLAGS", os.environ)

    def test_manual_software_preserves_flags_input_method_and_cuda(self):
        env = {"PROCESSING_VIEW_RENDERER": "software", "QTWEBENGINE_CHROMIUM_FLAGS": "--enable-logging",
               "QT_IM_MODULE": "xim", "CUDA_VISIBLE_DEVICES": "2"}
        probe = Mock(side_effect=AssertionError("must not probe"))
        with patch.dict(os.environ, env, clear=True):
            desktop.configure_desktop_rendering(probe)
            desktop.configure_desktop_rendering(probe)
            self.assertEqual(os.environ["QTWEBENGINE_CHROMIUM_FLAGS"], "--enable-logging --disable-gpu")
            self.assertEqual(os.environ["QT_IM_MODULE"], "xim")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "2")

    def test_explicit_software_flags_align_both_renderers(self):
        for env in ({"QTWEBENGINE_CHROMIUM_FLAGS": "--disable-gpu"}, {"QT_QUICK_BACKEND": "software"}):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                desktop.configure_desktop_rendering(Mock(side_effect=AssertionError("must not probe")))
                self.assertEqual(os.environ["QT_QUICK_BACKEND"], "software")
                self.assertEqual(os.environ["QTWEBENGINE_CHROMIUM_FLAGS"], "--disable-gpu")

    def test_hardware_override_custom_backend_and_other_platforms_do_not_probe(self):
        cases = [("linux", {"PROCESSING_VIEW_RENDERER": "hardware"}),
                 ("linux", {"QSG_RHI_BACKEND": "vulkan"}),
                 ("linux", {"QT_QUICK_BACKEND": "custom"}), ("darwin", {}), ("win32", {})]
        for platform, env in cases:
            with self.subTest(platform=platform, env=env), patch.dict(os.environ, env, clear=True), \
                 patch.object(desktop.sys, "platform", platform):
                desktop.configure_desktop_rendering(Mock(side_effect=AssertionError("must not probe")))
                self.assertNotIn("QTWEBENGINE_CHROMIUM_FLAGS", os.environ)

    def test_invalid_renderer_is_explained(self):
        with patch.dict(os.environ, {"PROCESSING_VIEW_RENDERER": "typo"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "auto、software 或 hardware"):
                desktop.configure_desktop_rendering()

    def test_system_bus_uses_existing_socket_without_changing_session_bus(self):
        with patch.dict(os.environ, {"DBUS_SESSION_BUS_ADDRESS": "unix:path=/custom/session"}, clear=True), \
             patch.object(desktop.sys, "platform", "linux"), patch("desktop.Path.is_socket", return_value=True):
            desktop.configure_linux_system_bus()
            self.assertEqual(os.environ["DBUS_SYSTEM_BUS_ADDRESS"], "unix:path=/run/dbus/system_bus_socket")
            self.assertEqual(os.environ["DBUS_SESSION_BUS_ADDRESS"], "unix:path=/custom/session")

    def test_system_bus_preserves_explicit_address(self):
        with patch.dict(os.environ, {"DBUS_SYSTEM_BUS_ADDRESS": "unix:path=/custom/system"}, clear=True), \
             patch.object(desktop.sys, "platform", "linux"), patch("desktop.Path.is_socket") as exists:
            desktop.configure_linux_system_bus()
            exists.assert_not_called()
            self.assertEqual(os.environ["DBUS_SYSTEM_BUS_ADDRESS"], "unix:path=/custom/system")

    def test_system_bus_missing_socket_is_not_fabricated(self):
        for result in (False, OSError("not accessible")):
            with self.subTest(result=result), patch.dict(os.environ, {}, clear=True), \
                 patch.object(desktop.sys, "platform", "linux"), \
                 patch("desktop.Path.is_socket", side_effect=result if isinstance(result, Exception) else None,
                       return_value=False):
                desktop.configure_linux_system_bus()
                self.assertNotIn("DBUS_SYSTEM_BUS_ADDRESS", os.environ)

    def test_system_bus_does_not_touch_other_platforms(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(desktop.sys, "platform", "win32"), \
             patch("desktop.Path.is_socket") as exists:
            desktop.configure_linux_system_bus()
            exists.assert_not_called()


if __name__ == "__main__":
    unittest.main()
