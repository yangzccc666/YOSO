"""Cross-platform writable directories for packaged and source builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "YOLODataPlatform"


def _expanded_environment_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(os.path.expandvars(value)).expanduser() if value else None


def platform_config_dir() -> Path:
    override = _expanded_environment_path("PROCESSING_VIEW_CONFIG_DIR")
    if override is not None:
        return override
    if sys.platform == "win32":
        root = _expanded_environment_path("APPDATA") or Path.home() / "AppData/Roaming"
        return root / APP_DIRECTORY_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / APP_DIRECTORY_NAME
    root = _expanded_environment_path("XDG_CONFIG_HOME") or Path.home() / ".config"
    return root / "yolo-data-platform"


def platform_data_dir() -> Path:
    override = _expanded_environment_path("PROCESSING_VIEW_DATA_DIR")
    if override is not None:
        return override
    # Source checkouts keep their existing ignored runtime directory. Frozen
    # applications must write outside the signed/read-only application bundle.
    if not getattr(sys, "frozen", False):
        return Path(__file__).resolve().parents[1] / "runtime"
    if sys.platform == "win32":
        root = _expanded_environment_path("LOCALAPPDATA") or Path.home() / "AppData/Local"
        return root / APP_DIRECTORY_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support" / APP_DIRECTORY_NAME
    root = _expanded_environment_path("XDG_DATA_HOME") or Path.home() / ".local/share"
    return root / "yolo-data-platform"
