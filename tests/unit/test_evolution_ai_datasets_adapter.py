from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from financial_vlm.integrations.evolution_ai_datasets_adapter import load_ocr_tokens


class LoadOcrTokensTest(unittest.TestCase):
    """`load_ocr_tokens` reads the on-disk sidecar file directly rather than
    going through the private package's `load_dataset`, so -- unlike the
    other loaders in this module -- it's testable without
    `evolution_ai_datasets` installed."""

    def test_missing_file_returns_empty_tuple(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            result = load_ocr_tokens(Path(tmp), "doc-1", "page-1")
            self.assertEqual(result, ())

    def test_parses_valid_tokens_and_skips_malformed_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_dir = root / "files" / "doc-1" / "pages" / "page-1"
            page_dir.mkdir(parents=True)
            (page_dir / "ocr.json").write_text(
                json.dumps(
                    [
                        {"c": [10, 20, 30, 40], "cf": 99, "e": 4, "t": "hello"},
                        {"c": [1, 2, 3], "t": "bad_coords"},  # wrong length, skipped
                        {"c": [1, 2, 3, 4], "t": ""},  # empty text, skipped
                        {"t": "no_coords"},  # missing coords, skipped
                    ]
                )
            )
            result = load_ocr_tokens(root, "doc-1", "page-1")
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].text, "hello")
            self.assertEqual((result[0].top, result[0].left, result[0].bottom, result[0].right), (10, 20, 30, 40))
            self.assertEqual(result[0].confidence, 99)

    def test_different_document_or_page_id_reads_a_different_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page_dir = root / "files" / "doc-1" / "pages" / "page-1"
            page_dir.mkdir(parents=True)
            (page_dir / "ocr.json").write_text(json.dumps([{"c": [0, 0, 1, 1], "t": "x"}]))
            self.assertEqual(len(load_ocr_tokens(root, "doc-1", "page-1")), 1)
            self.assertEqual(load_ocr_tokens(root, "doc-1", "page-2"), ())
            self.assertEqual(load_ocr_tokens(root, "doc-2", "page-1"), ())


if __name__ == "__main__":
    unittest.main()
