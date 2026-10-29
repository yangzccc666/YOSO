from __future__ import annotations

import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.star_model_history import StarModelHistoryStore


class StarModelHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = StarModelHistoryStore(self.root / "history.json")
        self.model_file = self.root / "中文模型.star"
        self.model_file.write_bytes(b"fixture")
        self.model = {"path": str(self.model_file), "labels": "OK,NG", "conf": 0.5}

    def test_persistent_all_models_without_five_record_limit(self):
        for index in range(13):
            self.store.remember([{**self.model, "labels": f"stage_{index},NG"}], used=True)
        records = StarModelHistoryStore(self.store.path).list()
        self.assertEqual(len(records), 13)
        self.assertEqual(records[0]["labels"], "stage_12,NG")
        self.assertTrue(all(row["available"] for row in records))
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_merge_keeps_id_counts_and_order_sensitive_variants(self):
        self.store.remember([self.model])
        original = self.store.list()[0]
        self.assertEqual(original["useCount"], 0)
        self.assertIsNone(original["lastUsedAt"])
        self.store.remember([{**self.model, "labels": " OK, NG ", "conf": 0.7}], used=True)
        self.store.remember([self.model, self.model], used=True)
        record = self.store.list()[0]
        self.assertEqual(record["id"], original["id"])
        self.assertEqual(record["createdAt"], original["createdAt"])
        self.assertEqual(record["useCount"], 2)
        self.assertEqual(record["conf"], 0.5)
        self.assertIsNotNone(record["lastUsedAt"])
        self.store.remember([{**self.model, "labels": "NG,OK"}], used=True)
        self.assertEqual([row["labels"] for row in self.store.list()], ["NG,OK", "OK,NG"])
        self.store.remember([{**self.model, "labels": "OK,,NG"}])
        self.assertEqual(self.store.list()[0]["labels"], "OK,,NG")

    def test_payload_legacy_and_multiple_models_keep_only_model_metadata(self):
        payload = {"parameters": {"star_models": json.dumps([self.model, {**self.model, "labels": "other"}]),
                                   "host": "private-host", "password": "not-for-history"},
                   "paths": {"local_video_file": "/some/video.mp4"}}
        self.store.remember_payload(payload)
        self.store.remember_payload({"parameters": {"labels": "legacy", "conf": 0.3}, "paths": {"local_model_file": str(self.model_file)}})
        self.assertEqual(len(self.store.list()), 3)
        saved = self.store.path.read_text()
        self.assertNotIn("not-for-history", saved)
        self.assertNotIn("private-host", saved)
        self.assertNotIn("video.mp4", saved)

    def test_missing_file_retained_and_delete_never_deletes_model(self):
        self.store.remember([self.model], require_files=True)
        self.model_file.rename(self.root / "moved.star")
        self.assertFalse(self.store.list()[0]["available"])
        self.assertEqual(len(self.store.list()), 1)
        self.store.delete(self.store.list()[0]["id"])
        self.assertEqual(self.store.list(), [])
        self.assertTrue((self.root / "moved.star").is_file())
        with self.assertRaises(ValueError):
            self.store.delete("nonexistent")

    def test_invalid_batch_does_not_partially_save(self):
        for invalid in ([], {}, [None], [{**self.model, "path": ""}], [{**self.model, "labels": ""}],
                        [{**self.model, "path": "invalid.txt"}], [{**self.model, "conf": float("nan")}],
                        [{**self.model, "conf": 2}], [{**self.model, "conf": "wrong"}],
                        [self.model, {**self.model, "path": str(self.root / "missing.plan")} ]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.store.remember(invalid, require_files=True)
            self.assertEqual(self.store.list(), [])

    def test_corrupt_history_is_not_overwritten(self):
        for original in ('{"unexpected":"schema"}', '[false]', 'broken json'):
            self.store.path.write_text(original)
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                self.store.remember([self.model])
            self.assertEqual(self.store.path.read_text(), original)

    def test_concurrent_runs_preserve_use_counts(self):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: self.store.remember([self.model], used=True), range(20)))
        self.assertEqual(len(self.store.list()), 1)
        self.assertEqual(self.store.list()[0]["useCount"], 20)

    def request(self, method, path, payload=None, activated=True):
        from backend import server
        body = json.dumps(payload or {}).encode()
        raw = f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body
        response = bytearray()
        connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=response.extend)
        with patch.object(server, "star_model_history", self.store), patch.object(server.license_store, "status", return_value={"activated": activated, "role": "user", "message": "activation required"}):
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
        head, body = bytes(response).split(b"\r\n\r\n", 1)
        return int(head.split()[1]), json.loads(body)

    def test_user_api_save_list_delete_and_license_guard(self):
        self.assertEqual(self.request("POST", "/api/star-model-history", {"models": [self.model]}, activated=False)[0], 403)
        self.assertEqual(self.store.list(), [])
        self.assertEqual(self.request("POST", "/api/star-model-history", {"models": [self.model]})[0], 200)
        status, payload = self.request("GET", "/api/star-model-history")
        self.assertEqual(status, 200)
        self.assertEqual(payload["models"][0]["labels"], "OK,NG")
        self.assertEqual(self.request("DELETE", f"/api/star-model-history/{payload['models'][0]['id']}")[0], 200)
        self.assertTrue(self.model_file.exists())
        self.assertEqual(self.store.list(), [])

    def test_start_records_all_models_and_failure_to_write_does_not_stop_inference(self):
        from backend import server
        payload = {"parameters": {"star_models": json.dumps([self.model, {**self.model, "labels": "second"}])}, "paths": {}}
        with patch.object(server.platform_task_manager, "start", return_value=SimpleNamespace(id="task")), \
             patch.object(server.platform_task_manager, "set_stop_callback"), \
             patch.object(server.remote_inference_manager, "start", side_effect=lambda *args, **kwargs: {"id": "session"}):
            status, _ = self.request("POST", "/api/remote-inference/start", payload)
            self.assertEqual(status, 201)
            self.assertEqual(len(self.store.list()), 2)
            with patch.object(self.store, "remember_payload", side_effect=OSError("disk full")):
                status, result = self.request("POST", "/api/remote-inference/start", payload)
            self.assertEqual(status, 201)
            self.assertIn("disk full", result["modelHistoryWarning"])

    def test_rejected_start_does_not_add_history(self):
        from backend import server
        with patch.object(server.platform_task_manager, "start", return_value=SimpleNamespace(id="task")), \
             patch.object(server.platform_task_manager, "finish"), patch.object(server, "record_finished_run"), \
             patch.object(server.remote_inference_manager, "start", side_effect=ValueError("invalid model")):
            status, _ = self.request("POST", "/api/remote-inference/start", {"parameters": {}, "paths": {}})
        self.assertEqual(status, 400)
        self.assertEqual(self.store.list(), [])
