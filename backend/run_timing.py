"""Shared wall-clock timing for legacy history and restored remote tasks."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any


def elapsed_seconds(started_at: Any, finished_at: Any) -> float | None:
    """Old naive timestamps are local time, as in the original desktop logger."""
    try:
        start = datetime.fromisoformat(str(started_at))
        end = datetime.fromisoformat(str(finished_at))
        seconds = end.astimezone().timestamp() - start.astimezone().timestamp()
        return round(seconds, 3) if math.isfinite(seconds) and seconds >= 0 else None
    except (ValueError, TypeError, OSError, OverflowError):
        return None


def valid_elapsed(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return round(float(value), 3) if math.isfinite(value) and value >= 0 else None
