"""Small encrypted local credential vault for remembered SSH passwords."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from .platform_paths import platform_config_dir


class CredentialStore:
    """Encrypt credentials at rest and restrict the vault to the current user."""

    def __init__(self, directory: Path | None = None) -> None:
        if directory is None:
            directory = platform_config_dir()
        self.directory = directory
        self.key_file = directory / "credentials.key"
        self.data_file = directory / "ssh_credentials.json"
        self.history_file = directory / "ssh_connection_history.json"
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

    def _load_history(self) -> list[dict[str, object]]:
        if not self.history_file.is_file():
            return []
        try:
            content = json.loads(self.history_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(content, list):
            return []
        records: list[dict[str, object]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            host = str(item.get("host", "")).strip()
            username = str(item.get("username", "")).strip()
            try:
                port = int(item.get("port", 22))
            except (TypeError, ValueError):
                continue
            if host and username and 1 <= port <= 65535:
                records.append({
                    "host": host,
                    "port": port,
                    "username": username,
                    "lastUsedAt": str(item.get("lastUsedAt", "")),
                })
        return records

    def _save_history(self, records: list[dict[str, object]]) -> None:
        self._ensure_directory()
        temporary = self.history_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.history_file)
        os.chmod(self.history_file, 0o600)

    def record_connection(self, host: str, port: int, username: str) -> None:
        """Remember a successfully tested SSH identity without storing a plaintext password."""
        with self._lock:
            identity = self._entry_id(host, port, username)
            records = [
                item for item in self._load_history()
                if self._entry_id(str(item["host"]), int(item["port"]), str(item["username"])) != identity
            ]
            records.insert(0, {
                "host": host,
                "port": port,
                "username": username,
                "lastUsedAt": datetime.now().isoformat(timespec="seconds"),
            })
            self._save_history(records[:20])

    def list_connections(self) -> list[dict[str, object]]:
        with self._lock:
            return [{
                **item,
                "passwordRemembered": self.get(
                    str(item["host"]), int(item["port"]), str(item["username"])
                ) is not None,
            } for item in self._load_history()]
