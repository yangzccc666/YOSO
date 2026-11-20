"""Persistent, non-modal desktop reminders for overdue countdowns."""

from __future__ import annotations

import sqlite3
from typing import Callable

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QApplication, QDialog, QLabel, QPushButton, QVBoxLayout

from backend.countdowns import Countdown


class CountdownPopup(QDialog):
    # The HTTP server sends snapshots from its thread. All widgets stay on the Qt thread.
    reminders_changed = Signal(object)

    def __init__(self, acknowledge: Callable[[str], None]) -> None:
        super().__init__(None, Qt.WindowType.Dialog | Qt.WindowType.CustomizeWindowHint
                         | Qt.WindowType.WindowTitleHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("倒计时提醒")
        self.setModal(False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_QuitOnClose, False)
        self.setFixedWidth(420)
        self.setStyleSheet("""
            CountdownPopup { background-color: #fff4f4; }
            QLabel { color: #51424a; }
            QLabel#countdownTitle { color: #a11c2f; font-size: 22px; font-weight: bold; }
            QLabel#countdownNote { color: #28232b; font-size: 16px; }
            QLabel#countdownError { color: #a11c2f; }
            QPushButton#countdownAcknowledge {
                background-color: #ad2437; color: white; border: 1px solid #ad2437;
                border-radius: 5px; font-weight: bold; padding: 8px 18px;
            }
            QPushButton#countdownAcknowledge:hover { background-color: #931a2c; }
            QPushButton#countdownAcknowledge:focus { border: 1px solid #181827; }
            QPushButton#countdownAcknowledge:disabled { background-color: #825b61; }
        """)
        self._acknowledge = acknowledge
        self._current_id: str | None = None
        self.reminders_changed.connect(self.show_reminders, Qt.ConnectionType.QueuedConnection)

        title = QLabel("倒计时到了")
        title.setObjectName("countdownTitle")
        self.note = QLabel()
        self.note.setObjectName("countdownNote")
        self.note.setTextFormat(Qt.TextFormat.PlainText)
        self.note.setWordWrap(True)
        self.remaining = QLabel()
        self.remaining.hide()
        self.error = QLabel("暂时无法确认，请重试。")
        self.error.setObjectName("countdownError")
        self.error.hide()
        self.button = QPushButton("知道了")
        self.button.setObjectName("countdownAcknowledge")
        self.button.clicked.connect(self._confirm)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        for label in (title, self.note, self.remaining, self.error):
            layout.addWidget(label)
        layout.addWidget(self.button, alignment=Qt.AlignmentFlag.AlignRight)

    @Slot(object)
    def show_reminders(self, timers: list[Countdown]) -> None:
        if not timers:
            self._current_id = None
            self.hide()
            return

        current = timers[0]
        is_new = current["id"] != self._current_id
        if is_new:
            self._current_id = current["id"]
            self.note.setText(current["note"])
            self.error.hide()
            self.button.setEnabled(True)
        waiting = len(timers) - 1
        self.remaining.setText(f"还有 {waiting} 项已到期")
        self.remaining.setVisible(waiting > 0)
        self.adjustSize()
        if not self.isVisible():
            screen = QGuiApplication.screenAt(QCursor.pos()) or QGuiApplication.primaryScreen()
            if screen is not None:
                center = screen.availableGeometry().center()
                self.move(center.x() - self.width() // 2, center.y() - self.height() // 2)
            self.show()
            self.raise_()
        if is_new:
            QApplication.beep()

    @Slot()
    def _confirm(self) -> None:
        if self._current_id is None:
            return
        try:
            self._acknowledge(self._current_id)
        except (sqlite3.Error, OSError):
            self.error.show()
            return
        self.error.hide()
        self.button.setEnabled(False)

    def reject(self) -> None:
        # Escape or window-manager shortcuts leave the reminder pending.
        pass
