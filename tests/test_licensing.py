from __future__ import annotations

import json
import tempfile
import io
import unittest
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from backend.licensing import LicenseStore, PRODUCT, canonical, encode, write_json
from backend import server


class LicensingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = Ed25519PrivateKey.generate()
        self.public = self.root / "public.json"
        write_json(self.public, {"publicKey": encode(self.private.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw))})
        self.store = LicenseStore(self.root / "activation.json", self.public)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def key(self, **overrides) -> str:
        claims = {"version": 1, "product": PRODUCT, "id": "test-license", "customer": "测试用户", "expiresAt": None, **overrides}
        payload = canonical(claims)
        return f"YOLO1.{encode(payload)}.{encode(self.private.sign(payload))}"

    def request(self, method: str, path: str, payload=None):
        body = json.dumps(payload).encode() if payload is not None else b""
        raw = f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        output = bytearray()
        connection = SimpleNamespace(makefile=lambda *_args: io.BytesIO(raw), sendall=output.extend)
        with patch.object(server, "license_store", self.store):
            server.AppHandler(connection, ("127.0.0.1", 12345), SimpleNamespace())
        headers, content = bytes(output).split(b"\r\n\r\n", 1)
        return int(headers.split(b" ")[1]), json.loads(content)

    def test_activation_persists_and_wrong_key_does_not_replace_it(self) -> None:
        self.assertFalse(self.store.status()["activated"])
        self.assertTrue(self.store.activate(self.key())["activated"])
        restarted = LicenseStore(self.store.storage_file, self.public)
        self.assertEqual(restarted.status()["customer"], "测试用户")
        with self.assertRaises(ValueError):
            self.store.activate("wrong-key")
        self.assertTrue(restarted.status()["activated"])

    def test_forged_expired_and_wrong_product_licenses_are_rejected(self) -> None:
        valid = self.key()
        prefix, payload, signature = valid.split(".")
        forged_payload = canonical({"version": 1, "product": PRODUCT, "id": "test-license", "customer": "伪造"})
        for key in (f"{prefix}.{encode(forged_payload)}.{signature}", self.key(product="other"), "YOLO1.@@@@.@@@@"):
            with self.assertRaises(ValueError):
                self.store.activate(key)
        with self.assertRaisesRegex(ValueError, "已过期"):
            self.store.activate(self.key(expiresAt=(datetime.now(timezone.utc) - timedelta(days=1)).isoformat()))

    def test_signed_revocation_disables_existing_activation(self) -> None:
        self.store.activate(self.key())
        public = json.loads(self.public.read_text())
        payload = {"ids": ["test-license"]}
        public["revocations"] = {"payload": payload, "signature": encode(self.private.sign(canonical(payload)))}
        write_json(self.public, public)
        self.assertFalse(self.store.status()["activated"])
        self.assertIn("撤销", self.store.status()["message"])

    def test_missing_or_corrupted_activation_fails_closed(self) -> None:
        self.store.storage_file.write_text("not json")
        self.assertFalse(self.store.status()["activated"])
        self.public.unlink()
        with self.assertRaisesRegex(ValueError, "缺少授权验证配置"):
            self.store.activate(self.key())

    def test_api_blocks_every_method_until_activated(self) -> None:
        for method, path in (("GET", "/api/workspace"), ("POST", "/api/groups"),
                             ("PUT", "/api/settings"), ("DELETE", "/api/groups/test")):
            code, data = self.request(method, path)
            self.assertEqual(code, 403)
            self.assertEqual(data["code"], "activation_required")
        self.assertFalse(self.request("GET", "/api/license/status")[1]["activated"])
        self.assertTrue(self.request("POST", "/api/license/activate", {"key": self.key()})[1]["activated"])
        self.assertEqual(self.request("GET", "/api/workspace")[0], 200)

    def test_user_distribution_never_exposes_owner_endpoints(self) -> None:
        for role in ("user", "admin"):
            self.store.activate(self.key(role=role))
            for method, endpoint in (("GET", ""), ("POST", "/issue"), ("POST", "/configure"), ("POST", "/revoke")):
                code, _data = self.request(method, "/api/license/admin" + endpoint, {})
                self.assertEqual(code, 404)
            self.assertEqual(self.request("GET", "/api/workspace")[0], 200)


if __name__ == "__main__":
    unittest.main()
