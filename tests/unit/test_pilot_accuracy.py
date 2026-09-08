from __future__ import annotations

import unittest

try:
    from PIL import Image
except ImportError:
    Image = None

from financial_vlm.evaluation.pilot_accuracy import (
    build_prediction_record,
    is_exact_correct,
    is_numeric_correct,
    make_full_page_hires,
    make_oracle_packet,
    make_oracle_crop,
    make_oracle_highlight,
    make_packet_noheader_crop,
    normalize_numeric_answer,
    summarize_accuracy,
)


class PilotAccuracyTests(unittest.TestCase):
    def test_numeric_normalization_handles_common_formats(self) -> None:
        self.assertEqual(normalize_numeric_answer("Answer: $1,590.00"), "1590")
        self.assertEqual(normalize_numeric_answer("(159)"), "-159")
        self.assertTrue(is_numeric_correct("The value is 1,590.", "1590"))
        self.assertFalse(is_numeric_correct("1,591", "1590"))

    def test_numeric_normalization_handles_european_thousands_dots(self) -> None:
        self.assertEqual(normalize_numeric_answer("39.039"), "39039")
        self.assertEqual(normalize_numeric_answer("(8.828.072)"), "-8828072")
        self.assertEqual(normalize_numeric_answer("6,054.049"), "6054049")
        self.assertEqual(normalize_numeric_answer("1.25"), "1.25")
        self.assertTrue(is_numeric_correct("6,054.049", "6,054,049"))

    def test_exact_match_is_case_insensitive_after_answer_prefix(self) -> None:
        self.assertTrue(is_exact_correct("Answer: 159", "159"))
        self.assertTrue(is_exact_correct(" VALUE: EBITDA ", "ebitda"))

    def test_summarizes_by_setting_and_variant(self) -> None:
        records = [
            {"setting": "full_page", "variant": "clean", "exact_correct": True, "numeric_correct": True},
            {"setting": "full_page", "variant": "clean", "exact_correct": False, "numeric_correct": True},
            {"setting": "oracle_evidence", "variant": "clean", "exact_correct": True, "numeric_correct": True},
        ]

        summary = summarize_accuracy(records)

        self.assertEqual(summary["overall"]["total"], 3)
        self.assertAlmostEqual(summary["by_setting"]["full_page"]["numeric_accuracy"], 1.0)
        self.assertAlmostEqual(summary["by_setting_variant"]["full_page/clean"]["exact_accuracy"], 0.5)

    def test_counterfactual_test_metrics(self) -> None:
        records = [
            {"setting": "full_page", "variant": "clean", "exact_correct": True, "numeric_correct": True},
            {
                "setting": "full_page",
                "variant": "target_value_replacement",
                "exact_correct": True,
                "numeric_correct": True,
            },
            {
                "setting": "full_page",
                "variant": "target_value_replacement",
                "exact_correct": False,
                "numeric_correct": False,
            },
            {"setting": "full_page", "variant": "header_address_swap", "exact_correct": True, "numeric_correct": True},
            {
                "setting": "full_page",
                "variant": "irrelevant_value_replacement",
                "exact_correct": True,
                "numeric_correct": True,
            },
        ]

        metrics = summarize_accuracy(records)["counterfactual_test_by_setting"]["full_page"]

        self.assertAlmostEqual(metrics["clean_accuracy_numeric"], 1.0)
        self.assertAlmostEqual(metrics["target_value_following_rate_numeric"], 0.5)
        self.assertAlmostEqual(metrics["header_following_rate_numeric"], 1.0)
        self.assertAlmostEqual(metrics["irrelevant_stability_rate_numeric"], 1.0)
        self.assertAlmostEqual(metrics["counterfactual_grounding_score_numeric"], 0.5 ** (1.0 / 3.0))

    def test_build_prediction_record_scores_correct_prediction(self) -> None:
        flat_record = {
            "group_id": "syn_000001_abc",
            "document_id": "doc-1",
            "question": "What was revenue in 2022?",
            "variant": "target_value_replacement",
            "image_path": "images/dev/syn_000001_abc__target_value_replacement.png",
            "answer": {"raw": "111", "metric": "Revenue", "period": "2022", "scale": None},
        }

        row = build_prediction_record(flat_record, setting="full_page", prediction="Answer: 111")

        self.assertEqual(row["setting"], "full_page")
        self.assertEqual(row["variant"], "target_value_replacement")
        self.assertEqual(row["group_id"], "syn_000001_abc")
        self.assertEqual(row["target_answer"], "111")
        self.assertTrue(row["exact_correct"])
        self.assertTrue(row["numeric_correct"])
        self.assertIsNone(row["error"])

    def test_build_prediction_record_counts_failure_without_dropping_sample(self) -> None:
        flat_record = {
            "group_id": "syn_000002_def",
            "question": "What was revenue in 2022?",
            "variant": "clean",
            "answer": {"raw": "100"},
        }

        row = build_prediction_record(
            flat_record,
            setting="full_page",
            prediction=None,
            error="FileNotFoundError: missing image",
        )

        self.assertFalse(row["exact_correct"])
        self.assertFalse(row["numeric_correct"])
        self.assertEqual(row["error"], "FileNotFoundError: missing image")
        self.assertIsNone(row["prediction"])

        summary = summarize_accuracy([row])
        self.assertEqual(summary["overall"]["total"], 1)
        self.assertEqual(summary["overall"]["numeric_correct"], 0)

    def test_summary_recomputes_correctness_from_prediction_text(self) -> None:
        records = [
            {
                "setting": "full_page",
                "variant": "clean",
                "prediction": "Answer: 1,590",
                "target_answer": "1590",
                "exact_correct": False,
                "numeric_correct": False,
            }
        ]

        summary = summarize_accuracy(records)

        self.assertEqual(summary["overall"]["numeric_correct"], 1)
        self.assertEqual(summary["overall"]["exact_correct"], 0)

    def test_oracle_crop_modes_return_images(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (100, 80), "white")
        bbox = (40, 30, 60, 45)

        cell = make_oracle_crop(image, bbox, mode="cell", padding=5)
        context = make_oracle_crop(image, bbox, mode="header_context", padding=5)

        self.assertEqual(cell.size, (30, 25))
        self.assertGreater(context.width, 0)
        self.assertGreater(context.height, cell.height)

    def test_oracle_highlight_marks_full_page_bbox(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (100, 80), "white")
        bbox = (40, 30, 60, 45)

        highlighted = make_oracle_highlight(image, bbox, padding=0, width=3)

        self.assertEqual(highlighted.size, image.size)
        self.assertEqual(highlighted.getpixel((40, 30)), (220, 0, 0))
        self.assertEqual(image.getpixel((40, 30)), (255, 255, 255))

    def test_packet_noheader_crop_enlarges_target_cell_only(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (100, 80), "white")
        bbox = (40, 30, 60, 45)

        crop = make_packet_noheader_crop(image, bbox, padding=5, scale=3)

        self.assertEqual(crop.size, (90, 75))

    def test_full_page_hires_upscales_whole_page_without_cropping(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (100, 80), "white")

        hires = make_full_page_hires(image, scale=3)
        unscaled = make_full_page_hires(image, scale=1)

        self.assertEqual(hires.size, (300, 240))
        self.assertEqual(unscaled.size, image.size)

    def test_full_page_hires_rejects_scale_below_one(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (100, 80), "white")

        with self.assertRaises(ValueError):
            make_full_page_hires(image, scale=0)

    def test_oracle_packet_renders_compact_context(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (300, 180), "white")
        packet = {
            "value_cell": {"cell_id": "r02c02", "text": "159", "bbox": [200, 90, 260, 120]},
            "row_headers": [{"cell_id": "r02c00", "text": "Adjusted EBITDA", "bbox": [0, 90, 100, 120]}],
            "column_headers": [{"cell_id": "r00c02", "text": "2022", "bbox": [200, 0, 260, 30]}],
            "unit_cells": [{"cell_id": "r01c02", "text": "USD millions", "bbox": [200, 30, 260, 60]}],
            "spanning_headers": [],
        }

        rendered = make_oracle_packet(image, packet, patches=[], padding=4, scale=1)

        self.assertGreater(rendered.width, 0)
        self.assertGreater(rendered.height, 0)
        self.assertLess(rendered.width, image.width)

    def test_oracle_packet_directly_crops_rendered_page_by_default(self) -> None:
        if Image is None:
            self.skipTest("Pillow is not installed")
        image = Image.new("RGB", (120, 80), "white")
        for x in range(40, 70):
            for y in range(30, 50):
                image.putpixel((x, y), (10, 20, 30))
        packet = {
            "value_cell": {"cell_id": "r01c01", "text": "old", "bbox": [40, 30, 70, 50]},
            "row_headers": [],
            "column_headers": [],
            "unit_cells": [],
            "spanning_headers": [],
        }
        patches = [{"cell": {"cell_id": "r01c01"}, "new_text": "999"}]

        rendered = make_oracle_packet(image, packet, patches=patches, padding=0, gap=0, scale=1)

        self.assertIn((10, 20, 30), rendered.getdata())


if __name__ == "__main__":
    unittest.main()
