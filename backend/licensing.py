"""Offline signed activation. Only the issuer holds the Ed25519 private key."""

from __future__ import annotations

import base64
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from .platform_paths import platform_data_dir


PUBLIC_FILE = Path(__file__).with_name("license_public.json")
PRODUCT = "YOLODataPlatform"


def encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def decode(value: str) -> bytes:
    return base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)


def canonical(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, indent=2)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class LicenseStore:
    def __init__(self, storage_file: Path | None = None, public_file: Path | None = None) -> None:
        self.storage_file = storage_file or platform_data_dir() / "activation.json"
        self.public_file = public_file or PUBLIC_FILE
        self._lock = threading.RLock()

    def verify(self, key: str) -> dict[str, Any]:
        if len(key) > 16384:
            raise ValueError("激活密钥格式不正确。")
        try:
            prefix, payload, signature = "".join(key.split()).split(".")
            if prefix != "YOLO1":
                raise ValueError()
            raw = decode(payload)
            claims = json.loads(raw)
            if not isinstance(claims, dict):
                raise ValueError()
            public = json.loads(self.public_file.read_text(encoding="utf-8"))
            verifier = Ed25519PublicKey.from_public_bytes(decode(public["publicKey"]))
            verifier.verify(decode(signature), raw)
            if claims["product"] != PRODUCT or claims["version"] != 1 or not claims["id"] or not claims["customer"]:
                raise ValueError()
            if claims.get("role", "user") not in {"admin", "user"}:
                raise ValueError()
            expiry = claims.get("expiresAt")
            if expiry and datetime.fromisoformat(expiry).astimezone(timezone.utc) <= datetime.now(timezone.utc):
                raise ValueError("激活密钥已过期，请联系提供者续期。")
            revocations = [public.get("revocations")]
            local_revocations = self.storage_file.parent / "license_revocations.json"
            if local_revocations.is_file():
                revocations.append(json.loads(local_revocations.read_text(encoding="utf-8")))
            for revoked in revocations:
                if not revoked:
                    continue
                verifier.verify(decode(revoked["signature"]), canonical(revoked["payload"]))
                if claims["id"] in revoked["payload"]["ids"]:
                    raise ValueError("激活密钥已被撤销，请联系提供者。")
            return claims
        except FileNotFoundError as exc:
            raise ValueError("程序缺少授权验证配置，请联系提供者。") from exc
        except (InvalidSignature, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("激活密钥无效或已被修改，请检查后重试。") from exc
        except ValueError as exc:
            if str(exc).startswith("激活密钥"):
                raise
            raise ValueError("激活密钥格式不正确，请粘贴完整密钥。") from exc

    def status(self) -> dict[str, Any]:
        with self._lock:
            try:
                saved = json.loads(self.storage_file.read_text(encoding="utf-8"))
                claims = self.verify(saved["key"])
                return {"activated": True, "customer": claims["customer"], "licenseId": claims["id"],
                        "expiresAt": claims.get("expiresAt"), "message": "平台已激活。"}
            except (OSError, ValueError, KeyError, TypeError) as exc:
                message = str(exc) if isinstance(exc, ValueError) else "首次使用，请输入提供者发给你的激活密钥。"
                return {"activated": False, "message": message}

    def activate(self, key: str) -> dict[str, Any]:
        with self._lock:
            self.verify(key)
            write_json(self.storage_file, {"key": "".join(key.split())})
            return self.status()

license_store = LicenseStore()
