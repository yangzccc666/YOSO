"""Desktop close guards must leave the server untouched on cancellation."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import desktop


class CloseConfirmationTests(unittest.TestCase):
    def test_native_dialog_requires_explicit_close_and_defaults_to_cancel(self):
        for choice in ("confirm", "cancel", "dismiss"):
            with self.subTest(choice=choice):
                dialog, confirm, cancel = Mock(), object(), object()
                dialog.addButton.side_effect = [confirm, cancel]
                dialog.clickedButton.return_value = {"confirm": confirm, "cancel": cancel, "dismiss": None}[choice]
                message_box = Mock(return_value=dialog)
                with patch.dict("sys.modules", {"PySide6.QtWidgets": SimpleNamespace(QMessageBox=message_box)}):
                    self.assertEqual(desktop.confirm_desktop_close(), choice == "confirm")
                dialog.setDefaultButton.assert_called_once_with(cancel)
                dialog.setEscapeButton.assert_called_once_with(cancel)
                dialog.exec.assert_called_once()

    def test_cancel_leaves_window_and_server_running(self):
        event, parent, shutdown = Mock(), Mock(), Mock()
        with patch.object(desktop, "confirm_desktop_close", return_value=False) as confirm:
            self.assertFalse(desktop.handle_desktop_close(event, parent, shutdown))
        confirm.assert_called_once_with(parent)
        event.ignore.assert_called_once()
        event.accept.assert_not_called()
        shutdown.assert_not_called()

    def test_accept_closes_only_after_confirmation(self):
        event, shutdown = Mock(), Mock()
        order = []
        event.ignore.side_effect = lambda: order.append("ignore")
        event.accept.side_effect = lambda: order.append("accept")
        shutdown.side_effect = lambda: order.append("shutdown")
        with patch.object(desktop, "confirm_desktop_close", side_effect=lambda _: order.append("confirm") or True):
            self.assertTrue(desktop.handle_desktop_close(event, None, shutdown))
        self.assertEqual(order, ["ignore", "confirm", "shutdown", "accept"])

    def test_confirmation_failure_does_not_stop_server(self):
        event, shutdown = Mock(), Mock()
        with patch.object(desktop, "confirm_desktop_close", side_effect=RuntimeError("dialog error")):
            with self.assertRaises(RuntimeError):
                desktop.handle_desktop_close(event, None, shutdown)
        event.ignore.assert_called_once()
        shutdown.assert_not_called()
        event.accept.assert_not_called()


if __name__ == "__main__":
    unittest.main()
