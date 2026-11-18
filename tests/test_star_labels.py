from __future__ import annotations

import io
import json
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend.star_labels import detect_star_labels


def star_fixture(labels, *, size_prefixed=True):
    """Build forward-offset FlatBuffers fixtures without a runtime dependency."""
    data = bytearray(8 if size_prefixed else 4)
    def alloc(size):
        data.extend(b"\0" * (-len(data) % 4))
        start = len(data)
        data.extend(b"\0" * size)
        return start
    def u32(offset, value):
        struct.pack_into("<I", data, offset, value)
    def table(slots):
        vt = alloc(4 + 2 * slots)
        struct.pack_into("<HH", data, vt, 4 + 2 * slots, 4 + 4 * slots)
        for slot in range(slots):
            struct.pack_into("<H", data, vt + 4 + slot * 2, 4 + slot * 4)
        start = alloc(4 + 4 * slots)
        struct.pack_into("<i", data, start, start - vt)
        return start
    def text(value):
        encoded = value.encode("utf-8")
        start = alloc(4 + len(encoded) + 1)
        u32(start, len(encoded))
        data[start + 4:start + 4 + len(encoded)] = encoded
        return start
    def link(offset, target):
        u32(offset, target - offset)
    root = table(4)
    link(4 if size_prefixed else 0, root)
    metadata = table(1)
    link(root + 4, metadata)
    link(metadata + 4, text("fixture"))
    hardware = table(1)
    link(root + 8, hardware)
    link(hardware + 4, text("TensorRT"))
    model = table(5)
    link(root + 12, model)
    vector = alloc(4 + 4 * len(labels))
    link(model + 20, vector)
    u32(vector, len(labels))
    # Store strings backwards physically to ensure class order follows offsets.
    for index in reversed(range(len(labels))):
        link(vector + 4 + index * 4, text(labels[index]))
    engine = b"ftrtNOT_A_REAL_ENGINE_ignore_fake_class_names"
    payload = alloc(4 + len(engine))
    link(root + 16, payload)
    u32(payload, len(engine))
    data[payload + 4:payload + 4 + len(engine)] = engine
    if size_prefixed:
        u32(0, len(data) - 4)
    return data, {"root": root, "model": model, "vector": vector, "payload": payload}


class StarLabelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "中文模型.star"

    def write(self, labels=("MARK", "start_okp", "hold_inspect"), **kwargs):
        data, offsets = star_fixture(list(labels), **kwargs)
        self.path.write_bytes(data)
        return data, offsets

    def test_actual_vector_order_unicode_duplicates_and_no_sidecar_guessing(self):
        names = ["MARK", "中文", "NGO", "MARK"]
        self.write(names)
        (self.path.parent / "classes.txt").write_text("WRONG\nOTHER", encoding="utf-8")
        before = self.path.read_bytes()
        result = detect_star_labels(self.path)
        self.assertEqual(result["labels"], names)
        self.assertEqual(result["count"], 4)
        self.assertIn("STAR 内嵌", result["source"])
        self.assertEqual(self.path.read_bytes(), before)

    def test_non_size_prefixed_flatbuffer(self):
        self.write(["OK", "NG"], size_prefixed=False)
        self.assertEqual(detect_star_labels(self.path)["labels"], ["OK", "NG"])

    def test_missing_file_and_plan_or_engine_are_explained(self):
        with self.assertRaisesRegex(ValueError, "不存在"):
            detect_star_labels(self.path)
        for suffix in (".plan", ".engine", ".txt"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, "仅支持"):
                detect_star_labels(self.path.with_suffix(suffix))

    def test_truncated_and_engine_disguised_as_star_are_not_guessed(self):
        data, _ = self.write()
        for invalid in (b"", b"ftrtMARK\0OK\0NG\0", data[:20], data[:-1]):
            self.path.write_bytes(invalid)
            with self.subTest(length=len(invalid)), self.assertRaises(ValueError):
                detect_star_labels(self.path)

    def test_missing_labels_field_and_empty_vector(self):
        self.write([])
        with self.assertRaisesRegex(ValueError, "类别列表为空"):
            detect_star_labels(self.path)
        data, offsets = self.write()
        vt = offsets["model"] - struct.unpack_from("<i", data, offsets["model"])[0]
        struct.pack_into("<H", data, vt + 12, 0)
        self.path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, "缺少有效"):
            detect_star_labels(self.path)

    def test_bounds_and_string_validation(self):
        for invalid in ("vector-pointer", "vector-count", "string-pointer", "string-length", "terminator", "utf8", "vtable"):
            data, offsets = self.write()
            vector = offsets["vector"]
            first = vector + 4 + struct.unpack_from("<I", data, vector + 4)[0]
            if invalid == "vector-pointer":
                struct.pack_into("<I", data, offsets["model"] + 20, 0xFFFFFFFF)
            elif invalid == "vector-count":
                struct.pack_into("<I", data, vector, 0xFFFFFFFF)
            elif invalid == "string-pointer":
                struct.pack_into("<I", data, vector + 4, 0xFFFFFFFF)
            elif invalid == "string-length":
                struct.pack_into("<I", data, first, 0xFFFFFFFF)
            elif invalid == "terminator":
                data[first + 4 + len("MARK")] = 1
            elif invalid == "utf8":
                data[first + 4] = 0xFF
            else:
                struct.pack_into("<i", data, offsets["root"], 0x7FFFFFFF)
            self.path.write_bytes(data)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                detect_star_labels(self.path)

    def test_unsafe_csv_names_rejected_without_changing_model(self):
        for label in ("", " ", "bad,name", "bad，name", "bad\nname", "bad\0name"):
            self.write([label])
            original = self.path.read_bytes()
            with self.subTest(label=label), self.assertRaises(ValueError):
                detect_star_labels(self.path)
            self.assertEqual(self.path.read_bytes(), original)

    def request(self, activated=True, model=None):
        from backend import server
        body = json.dumps({"modelFile": str(model or self.path)}).encode()
        raw = b"POST /api/remote-inference/detect-labels HTTP/1.0\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body
        output = bytearray()
        connection = SimpleNamespace(makefile=lambda *_: io.BytesIO(raw), sendall=output.extend)
        with patch.object(server.license_store, "status", return_value={"activated": activated, "role": "user", "message": "activation required"}), \
             patch.object(server.remote_inference_manager, "test_connection") as ssh, \
             patch.object(server.remote_inference_manager, "start") as start:
            server.AppHandler(connection, ("127.0.0.1", 1), SimpleNamespace())
            ssh.assert_not_called()
            start.assert_not_called()
        head, payload = bytes(output).split(b"\r\n\r\n", 1)
        return int(head.split()[1]), json.loads(payload)

    def test_api_license_guard_success_and_clear_invalid_model_error(self):
        self.write()
        self.assertEqual(self.request(activated=False)[0], 403)
        status, response = self.request()
        self.assertEqual(status, 200)
        self.assertEqual(response["labels"], ["MARK", "start_okp", "hold_inspect"])
        status, response = self.request(model=self.path.with_suffix(".plan"))
        self.assertEqual(status, 400)
        self.assertIn("仅支持", response["error"])


if __name__ == "__main__":
    unittest.main()
