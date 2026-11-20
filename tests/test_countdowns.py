import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.countdowns import CountdownStore


class CountdownStoreTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "config" / "countdowns.sqlite3"
        self.store = CountdownStore(self.path)
        clock = patch("backend.countdowns.epoch_ms", return_value=1000)
        self.clock = clock.start()
        self.addCleanup(clock.stop)

    def test_overdue_reminders_survive_restart_until_acknowledged(self):
        self.store.start(" 检查训练结果 ", 2)
        self.store.start("复查抽帧", 5)
        reopened = CountdownStore(self.path)
        self.assertEqual([(row["note"], row["deadlineMs"]) for row in reopened.list()],
                         [("检查训练结果", 3000), ("复查抽帧", 6000)])
        self.clock.return_value = 2999
        self.assertEqual(reopened.list(overdue=True), [])
        self.clock.return_value = 6000
        overdue = reopened.list(overdue=True)
        self.assertEqual([row["note"] for row in overdue], ["检查训练结果", "复查抽帧"])
        self.assertEqual(CountdownStore(self.path).list(overdue=True), overdue)
        reopened.remove(overdue[0]["id"])
        self.assertEqual(CountdownStore(self.path).list(overdue=True), overdue[1:])

    def test_timer_already_sent_to_the_tray_still_gets_a_visible_reminder(self):
        self.path.parent.mkdir()
        with sqlite3.connect(self.path) as db:
            db.execute("""
                CREATE TABLE countdowns (
                    id TEXT PRIMARY KEY, note TEXT NOT NULL, deadline_ms INTEGER NOT NULL,
                    notified INTEGER NOT NULL DEFAULT 0
                )
            """)
            db.execute("INSERT INTO countdowns VALUES (?, ?, ?, ?)", ("older", "检查队列", 1000, 1))
        db.close()
        self.assertEqual(self.store.list(overdue=True)[0]["note"], "检查队列")
        self.store.start("新版计时", 5)
        self.assertEqual(len(self.store.list()), 2)

    def test_cancelled_timer_never_notifies(self):
        self.store.start("待取消", 1)
        key = self.store.list()[0]["id"]
        self.store.remove(key)
        self.clock.return_value = 3000
        self.assertEqual(self.store.list(overdue=True), [])
        self.assertEqual(self.store.list(), [])

    def test_invalid_creation_does_not_write_storage(self):
        for note, duration in [(" ", 1), ("字" * 121, 1), ("备注", 0), ("备注", True), ("备注", 0.5)]:
            with self.subTest(note=note, duration=duration), self.assertRaises(ValueError):
                self.store.start(note, duration)
        self.assertFalse(self.path.exists())
