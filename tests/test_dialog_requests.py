"""Concurrent picker requests must not create another modal or reuse its answer."""
import threading
import unittest
from unittest.mock import patch

from backend import server


class DialogRequestTests(unittest.TestCase):
    def test_browser_fallback_ignores_duplicate_and_accepts_next_request(self):
        entered, release = threading.Event(), threading.Event()
        replies = []

        def pick(mode):
            entered.set()
            release.wait(timeout=3)
            return [mode]

        with patch.object(server, "_path_chooser", None), patch.object(server, "_choose_tk_path", side_effect=pick):
            worker = threading.Thread(target=lambda: replies.append(server.choose_path("file")))
            worker.start()
            try:
                self.assertTrue(entered.wait(2))
                self.assertEqual(server.choose_path("directory"), [])
            finally:
                release.set()
                worker.join(3)
            self.assertEqual(replies, [["file"]])
            self.assertEqual(server.choose_path("multiple"), ["multiple"])

    def test_shutdown_releases_desktop_request_before_qt_processes_its_event(self):
        try:
            from desktop_dialogs import DesktopPathChooser
            from PySide6.QtCore import Qt
        except ImportError:
            self.skipTest("PySide6 unavailable")

        chooser = DesktopPathChooser(None)
        queued = threading.Event()
        chooser.requested.connect(lambda *_: queued.set(), Qt.ConnectionType.DirectConnection)
        replies = []
        worker = threading.Thread(target=lambda: replies.append(chooser.choose("file")), daemon=True)
        worker.start()
        try:
            self.assertTrue(queued.wait(2))
            self.assertTrue(chooser.busy)
            self.assertEqual(chooser.choose("multiple"), [])
        finally:
            chooser.stop()
            worker.join(2)
        self.assertEqual(replies, [[]])
        self.assertFalse(chooser.busy)
        self.assertEqual(chooser.choose("directory"), [])
