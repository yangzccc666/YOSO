"""Persistent countdowns shared by the desktop and browser interfaces."""

from __future__ import annotations

import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import TypedDict
from uuid import uuid4


MAX_DURATION_SECONDS = 9999 * 3600 + 59 * 60 + 59
MAX_NOTE_LENGTH = 120


class Countdown(TypedDict):
    id: str
    note: str
    deadlineMs: int


def epoch_ms() -> int:
    return time.time_ns() // 1_000_000


class CountdownStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connection(self):
        with closing(sqlite3.connect(self.path, timeout=0.15)) as db, db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS countdowns (
                    id TEXT PRIMARY KEY,
                    note TEXT NOT NULL,
                    deadline_ms INTEGER NOT NULL
                )
            """)
            yield db

    def list(self, *, overdue: bool = False) -> list[Countdown]:
        if not self.path.exists():
            return []
        condition, parameters = ("WHERE deadline_ms <= ?", (epoch_ms(),)) if overdue else ("", ())
        with self._connection() as db:
            rows = db.execute(f"SELECT id, note, deadline_ms FROM countdowns {condition} ORDER BY deadline_ms, id",
                              parameters)
            return [{"id": key, "note": note, "deadlineMs": deadline} for key, note, deadline in rows]

    def start(self, note: object, duration_seconds: object) -> None:
        if not isinstance(note, str) or not note.strip() or len(note.strip()) > MAX_NOTE_LENGTH:
            raise ValueError(f"请填写 1～{MAX_NOTE_LENGTH} 字的倒计时备注。")
        if type(duration_seconds) is not int or not 1 <= duration_seconds <= MAX_DURATION_SECONDS:
            raise ValueError("倒计时时长必须大于 0，小时最多 9999，分钟和秒最多 59。")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as db:
            db.execute("INSERT INTO countdowns (id, note, deadline_ms) VALUES (?, ?, ?)",
                       (uuid4().hex, note.strip(), epoch_ms() + duration_seconds * 1000))

    def remove(self, key: str) -> None:
        if self.path.exists():
            with self._connection() as db:
                db.execute("DELETE FROM countdowns WHERE id = ?", (key,))
