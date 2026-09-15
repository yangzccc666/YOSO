"""Small encrypted local credential vault for remembered SSH passwords."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken


class CredentialStore:
    """Encrypt credentials at rest and restrict the vault to the current user."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            config_value = os.environ.get("XDG_CONFIG_HOME", "").strip()
            config_root = Path(config_value).expanduser() if config_value else Path.home() / ".config"
            directory = config_root / "yolo-data-platform"
        self.directory = directory
        self.key_file = directory / "credentials.key"
        self.data_file = directory / "ssh_credentials.json"
        self._lock = threading.RLock()

    @staticmethod
    def _entry_id(host: str, port: int, username: str) -> str:
        identity = f"{username}@{host}:{port}".encode("utf-8")
        return hashlib.sha256(identity).hexdigest()

    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    def _fernet(self, create: bool) -> Fernet | None:
        if self.key_file.is_file():
            return Fernet(self.key_file.read_bytes().strip())
        if not create:
            return None
        self._ensure_directory()
        key = Fernet.generate_key()
        descriptor = os.open(self.key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, key)
        finally:
            os.close(descriptor)
        return Fernet(key)

    def _load(self) -> dict[str, str]:
        if not self.data_file.is_file():
            return {}
        try:
            content = json.loads(self.data_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return {
            str(key): str(value)
            for key, value in content.items()
            if isinstance(key, str) and isinstance(value, str)
        } if isinstance(content, dict) else {}

    def _save(self, entries: dict[str, str]) -> None:
        self._ensure_directory()
        temporary = self.data_file.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        temporary.replace(self.data_file)
        os.chmod(self.data_file, 0o600)

    def get(self, host: str, port: int, username: str) -> str | None:
        with self._lock:
            fernet = self._fernet(create=False)
            token = self._load().get(self._entry_id(host, port, username))
            if fernet is None or not token:
                return None
            try:
                return fernet.decrypt(token.encode("ascii")).decode("utf-8")
            except (InvalidToken, UnicodeDecodeError, ValueError):
                return None

    def set(self, host: str, port: int, username: str, password: str) -> None:
        if not password:
            raise ValueError("不能保存空密码。")
        with self._lock:
            fernet = self._fernet(create=True)
            assert fernet is not None
            entries = self._load()
            entries[self._entry_id(host, port, username)] = (
                fernet.encrypt(password.encode("utf-8")).decode("ascii")
            )
            self._save(entries)

    def delete(self, host: str, port: int, username: str) -> bool:
        with self._lock:
            entries = self._load()
            removed = entries.pop(self._entry_id(host, port, username), None) is not None
            if removed:
                self._save(entries)
            return removed
