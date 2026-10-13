from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.handlers import RunContext
from backend.negative_samples import run_extract_negative_samples


class NegativeSampleTests(unittest.TestCase):
    def test_moves_only_images_with_valid_empty_xml_and_deletes_that_xml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "negative"
            source.mkdir()
            (source / "empty.jpg").write_bytes(b"empty-image")
            (source / "empty.xml").write_text("<annotation></annotation>", encoding="utf-8")
            (source / "labeled.jpg").write_bytes(b"labeled-image")
            (source / "labeled.xml").write_text(
                "<annotation><object><name>part</name></object></annotation>", encoding="utf-8"
            )
            (source / "broken.jpg").write_bytes(b"broken-image")
            (source / "broken.xml").write_text("<annotation>", encoding="utf-8")

            messages: list[str] = []
            result = run_extract_negative_samples(RunContext(
                function_id="extract_negative_samples",
                paths={"source_folder": source, "negative_folder": destination},
                parameters={"recursive": False},
                report=messages.append,
            ))

            self.assertEqual(result["negativeCount"], 1)
            self.assertEqual(result["labeledCount"], 1)
            self.assertEqual(result["skippedCount"], 1)
            self.assertFalse((source / "empty.jpg").exists())
            self.assertFalse((source / "empty.xml").exists())
            self.assertEqual((destination / "empty.jpg").read_bytes(), b"empty-image")
            self.assertTrue((source / "labeled.jpg").exists())
            self.assertTrue((source / "labeled.xml").exists())
            self.assertTrue((source / "broken.jpg").exists())
            self.assertTrue((source / "broken.xml").exists())
            self.assertTrue(any("XML 无法解析" in message for message in messages))

    def test_existing_destination_file_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "negative"
            source.mkdir()
            destination.mkdir()
            (source / "sample.jpg").write_bytes(b"new")
            (source / "sample.xml").write_text("<annotation />", encoding="utf-8")
            (destination / "sample.jpg").write_bytes(b"existing")

            result = run_extract_negative_samples(RunContext(
                function_id="extract_negative_samples",
                paths={"source_folder": source, "negative_folder": destination},
                parameters={},
                report=lambda _message: None,
            ))

            self.assertEqual(result["negativeCount"], 0)
            self.assertEqual(result["skippedCount"], 1)
            self.assertEqual((destination / "sample.jpg").read_bytes(), b"existing")
            self.assertTrue((source / "sample.jpg").exists())
            self.assertTrue((source / "sample.xml").exists())


if __name__ == "__main__":
    unittest.main()
