from __future__ import annotations

import unittest

from financial_vlm.data.grounding_target import (
    EVIDENCE_FIRST_PROMPT_INSTRUCTION,
    GROUNDING_PROMPT_INSTRUCTION,
    bbox01_to_pixels,
    bbox_iou,
    build_grounding_example,
    categorize_hfr_errors,
    check_split_leakage,
    evidence_first_prompt,
    format_evidence_first_target,
    format_grounding_target,
    grounding_prompt,
    normalize_bbox_1000,
    normalize_header_text,
    parse_evidence_first_prediction,
    parse_grounding_prediction,
    resolve_cell_at_point,
    source_index_from_group_id,
    summarize_evidence_first_grounding_metrics,
    summarize_grounding_metrics,
    validate_and_build_examples,
    validate_grounding_payload,
)


def make_payload(**overrides) -> dict:
    payload = {
        "image_path": "images/train/syn_000001_abc__clean.png",
        "answer": {"raw": "125.4"},
        "evidence": {
            "target_id": "cell_12_2",
            "bbox_normalised": [0.40, 0.27, 0.49, 0.31],
            "row_header_ids": ["cell_12_0"],
            "column_header_ids": ["cell_0_2"],
        },
        "evidence_units": [
            {"id": "cell_12_0", "text": "Revenue", "bbox_normalised": [0.0, 0.27, 0.15, 0.31]},
            {"id": "cell_0_2", "text": "FY2023", "bbox_normalised": [0.40, 0.0, 0.49, 0.05]},
            {"id": "cell_12_2", "text": "125.4", "bbox_normalised": [0.40, 0.27, 0.49, 0.31]},
        ],
    }
    payload.update(overrides)
    return payload


class GroundingPromptTests(unittest.TestCase):
    def test_prompt_includes_fixed_instruction_and_question_only(self) -> None:
        prompt = grounding_prompt("What was revenue in FY2023?")

        self.assertIn(GROUNDING_PROMPT_INSTRUCTION, prompt)
        self.assertIn("What was revenue in FY2023?", prompt)
        # no oracle hints leak into the prompt
        self.assertNotIn("Revenue", prompt.replace("What was revenue in FY2023?", ""))
        self.assertNotIn("125.4", prompt)

    def test_prompt_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            grounding_prompt("   ")


class HeaderNormalizationTests(unittest.TestCase):
    def test_normalizes_case_and_whitespace(self) -> None:
        self.assertEqual(normalize_header_text("  Revenue  "), "revenue")
        self.assertEqual(normalize_header_text("FY   2023"), "fy 2023")
        self.assertEqual(normalize_header_text("Revenue"), normalize_header_text("REVENUE"))

    def test_none_normalizes_to_empty_string(self) -> None:
        self.assertEqual(normalize_header_text(None), "")


class BboxConversionTests(unittest.TestCase):
    def test_normalize_bbox_1000_matches_pixel_formula(self) -> None:
        # x1/W=0.403, y1/H=0.271, x2/W=0.492, y2/H=0.309 -- same as the
        # worked example in the task's own formula (403/1000 style values)
        result = normalize_bbox_1000([0.403, 0.271, 0.492, 0.309])
        self.assertEqual(result, (403, 271, 492, 309))

    def test_normalize_bbox_1000_clamps_to_range(self) -> None:
        result = normalize_bbox_1000([-0.01, 0.0, 1.0, 1.005])
        self.assertEqual(result, (0, 0, 1000, 1000))

    def test_normalize_bbox_1000_rejects_wrong_length(self) -> None:
        with self.assertRaises(ValueError):
            normalize_bbox_1000([0.1, 0.2, 0.3])

    def test_bbox01_to_pixels_recovers_original_coordinates(self) -> None:
        pixels = bbox01_to_pixels([0.1, 0.2, 0.5, 0.6], width=1000, height=500)
        self.assertEqual(pixels, (100, 100, 500, 300))


class TargetFormattingTests(unittest.TestCase):
    def test_format_matches_the_specified_template_exactly(self) -> None:
        text = format_grounding_target("125.4", "Revenue", "FY2023", (403, 271, 492, 309))
        self.assertEqual(
            text,
            "<answer>125.4</answer>\n"
            "<row>Revenue</row>\n"
            "<column>FY2023</column>\n"
            "<bbox>403 271 492 309</bbox>",
        )


class ParsePredictionTests(unittest.TestCase):
    def test_parses_well_formed_prediction(self) -> None:
        text = "<answer>125.4</answer>\n<row>Revenue</row>\n<column>FY2023</column>\n<bbox>403 271 492 309</bbox>"
        parsed = parse_grounding_prediction(text)

        self.assertEqual(parsed.answer, "125.4")
        self.assertEqual(parsed.row, "Revenue")
        self.assertEqual(parsed.column, "FY2023")
        self.assertEqual(parsed.bbox, (403, 271, 492, 309))
        self.assertTrue(parsed.bbox_valid)

    def test_missing_tags_become_none_without_raising(self) -> None:
        parsed = parse_grounding_prediction("125.4, somewhere in the Revenue row")

        self.assertIsNone(parsed.answer)
        self.assertIsNone(parsed.row)
        self.assertIsNone(parsed.column)
        self.assertIsNone(parsed.bbox)
        self.assertFalse(parsed.bbox_valid)

    def test_bbox_with_reversed_coordinates_is_invalid(self) -> None:
        parsed = parse_grounding_prediction("<bbox>500 500 100 100</bbox>")
        self.assertFalse(parsed.bbox_valid)
        self.assertIsNone(parsed.bbox)

    def test_bbox_out_of_range_is_invalid(self) -> None:
        parsed = parse_grounding_prediction("<bbox>0 0 1500 300</bbox>")
        self.assertFalse(parsed.bbox_valid)

    def test_bbox_non_integer_tokens_do_not_match(self) -> None:
        parsed = parse_grounding_prediction("<bbox>a b c d</bbox>")
        self.assertIsNone(parsed.bbox)
        self.assertFalse(parsed.bbox_valid)

    def test_whitespace_tolerant_parsing(self) -> None:
        text = "<answer> 125.4 </answer><row> Revenue </row><column>FY2023</column><bbox>  403   271  492 309 </bbox>"
        parsed = parse_grounding_prediction(text)
        self.assertEqual(parsed.answer, "125.4")
        self.assertEqual(parsed.row, "Revenue")
        self.assertEqual(parsed.bbox, (403, 271, 492, 309))


class BboxIoUTests(unittest.TestCase):
    def test_identical_boxes_have_iou_one(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 100, 100), (0, 0, 100, 100)), 1.0)

    def test_disjoint_boxes_have_iou_zero(self) -> None:
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_partial_overlap(self) -> None:
        # a: [0,0,10,10] area 100; b: [5,5,15,15] area 100; intersection [5,5,10,10] area 25
        # union = 100+100-25=175 -> iou=25/175
        self.assertAlmostEqual(bbox_iou((0, 0, 10, 10), (5, 5, 15, 15)), 25 / 175)


class ResolveCellAtPointTests(unittest.TestCase):
    def test_point_inside_single_cell_resolves(self) -> None:
        units = [{"id": "cell_a", "bbox_normalised": [0.0, 0.0, 0.5, 0.5]}]
        self.assertEqual(resolve_cell_at_point(100, 100, units), "cell_a")

    def test_point_outside_any_cell_returns_none(self) -> None:
        units = [{"id": "cell_a", "bbox_normalised": [0.0, 0.0, 0.1, 0.1]}]
        self.assertIsNone(resolve_cell_at_point(900, 900, units))

    def test_overlapping_cells_prefer_target_id(self) -> None:
        units = [
            {"id": "cell_wide_header", "bbox_normalised": [0.0, 0.0, 1.0, 1.0]},
            {"id": "cell_target", "bbox_normalised": [0.4, 0.4, 0.6, 0.6]},
        ]
        result = resolve_cell_at_point(500, 500, units, target_id="cell_target")
        self.assertEqual(result, "cell_target")

    def test_overlapping_cells_without_target_match_return_first(self) -> None:
        units = [
            {"id": "cell_a", "bbox_normalised": [0.0, 0.0, 1.0, 1.0]},
            {"id": "cell_b", "bbox_normalised": [0.4, 0.4, 0.6, 0.6]},
        ]
        result = resolve_cell_at_point(500, 500, units, target_id="cell_not_present")
        self.assertEqual(result, "cell_a")


class ValidateGroundingPayloadTests(unittest.TestCase):
    def test_valid_payload_returns_none(self) -> None:
        self.assertIsNone(validate_grounding_payload(make_payload()))

    def test_missing_evidence(self) -> None:
        payload = make_payload()
        del payload["evidence"]
        self.assertEqual(validate_grounding_payload(payload), "missing_evidence")

    def test_missing_target_cell_id(self) -> None:
        payload = make_payload()
        payload["evidence"]["target_id"] = ""
        self.assertEqual(validate_grounding_payload(payload), "missing_target_cell")

    def test_target_id_not_in_evidence_units(self) -> None:
        payload = make_payload()
        payload["evidence"]["target_id"] = "cell_does_not_exist"
        self.assertEqual(validate_grounding_payload(payload), "missing_target_cell")

    def test_empty_row_header_ids(self) -> None:
        payload = make_payload()
        payload["evidence"]["row_header_ids"] = []
        self.assertEqual(validate_grounding_payload(payload), "missing_row_header")

    def test_ambiguous_multiple_row_header_ids(self) -> None:
        payload = make_payload()
        payload["evidence"]["row_header_ids"] = ["cell_12_0", "cell_11_0"]
        self.assertEqual(validate_grounding_payload(payload), "ambiguous_row_header")

    def test_empty_column_header_ids(self) -> None:
        payload = make_payload()
        payload["evidence"]["column_header_ids"] = []
        self.assertEqual(validate_grounding_payload(payload), "missing_column_header")

    def test_ambiguous_multiple_column_header_ids(self) -> None:
        payload = make_payload()
        payload["evidence"]["column_header_ids"] = ["cell_0_2", "cell_0_3"]
        self.assertEqual(validate_grounding_payload(payload), "ambiguous_column_header")

    def test_invalid_bbox_reversed_coordinates(self) -> None:
        payload = make_payload()
        payload["evidence"]["bbox_normalised"] = [0.5, 0.5, 0.1, 0.1]
        self.assertEqual(validate_grounding_payload(payload), "invalid_bbox")

    def test_missing_answer(self) -> None:
        payload = make_payload()
        payload["answer"] = {"raw": ""}
        self.assertEqual(validate_grounding_payload(payload), "missing_answer")

    def test_unsafe_text_rejected(self) -> None:
        payload = make_payload()
        payload["evidence_units"][0]["text"] = "Rev<enue>"
        self.assertEqual(validate_grounding_payload(payload), "unsafe_text_for_serialization")


class BuildGroundingExampleTests(unittest.TestCase):
    def test_builds_example_with_resolved_headers_and_bbox(self) -> None:
        example = build_grounding_example("syn_000001_abc", "clean", "What was revenue in FY2023?", make_payload())

        self.assertEqual(example.row_header, "Revenue")
        self.assertEqual(example.column_header, "FY2023")
        self.assertEqual(example.answer_raw, "125.4")
        self.assertEqual(example.target_cell_id, "cell_12_2")
        self.assertEqual(example.bbox_1000, normalize_bbox_1000([0.40, 0.27, 0.49, 0.31]))
        self.assertIn("<answer>125.4</answer>", example.target_text)
        self.assertIn("<row>Revenue</row>", example.target_text)
        self.assertIn("<column>FY2023</column>", example.target_text)

    def test_raises_with_group_id_and_reason_in_message(self) -> None:
        payload = make_payload()
        payload["evidence"]["row_header_ids"] = []
        with self.assertRaises(ValueError) as ctx:
            build_grounding_example("syn_000001_abc", "clean", "Q?", payload)
        self.assertIn("syn_000001_abc", str(ctx.exception))
        self.assertIn("missing_row_header", str(ctx.exception))


class SummarizeGroundingMetricsTests(unittest.TestCase):
    def test_aggregates_accuracy_and_iou_per_variant(self) -> None:
        rows = [
            {
                "variant": "clean",
                "answer_correct": True,
                "row_correct": True,
                "column_correct": True,
                "cell_correct": True,
                "bbox_iou": 0.9,
            },
            {
                "variant": "clean",
                "answer_correct": True,
                "row_correct": False,
                "column_correct": True,
                "cell_correct": False,
                "bbox_iou": 0.1,
            },
            {
                "variant": "header_address_swap",
                "answer_correct": False,
                "row_correct": True,
                "column_correct": False,
                "cell_correct": False,
                "bbox_iou": 0.0,
            },
        ]

        summary = summarize_grounding_metrics(rows)

        self.assertEqual(set(summary), {"clean", "header_address_swap"})
        clean = summary["clean"]
        self.assertEqual(clean["total"], 2)
        self.assertAlmostEqual(clean["row_header_accuracy"], 0.5)
        self.assertAlmostEqual(clean["column_header_accuracy"], 1.0)
        self.assertAlmostEqual(clean["cell_acc_at_1"], 0.5)
        self.assertAlmostEqual(clean["bbox_iou_mean"], 0.5)
        self.assertAlmostEqual(clean["bbox_iou_median"], 0.5)
        self.assertEqual(clean["answer_correct_count"], 2)
        # 1 of 2 is both answer-correct and cell-correct
        self.assertAlmostEqual(clean["p_answer_and_cell_correct"], 0.5)
        # of the 2 answer-correct rows, 1 is also cell-correct
        self.assertAlmostEqual(clean["p_cell_correct_given_answer_correct"], 0.5)

    def test_p_cell_given_answer_is_none_when_no_answer_correct(self) -> None:
        rows = [
            {
                "variant": "clean",
                "answer_correct": False,
                "row_correct": False,
                "column_correct": False,
                "cell_correct": False,
                "bbox_iou": 0.0,
            }
        ]
        summary = summarize_grounding_metrics(rows)
        self.assertIsNone(summary["clean"]["p_cell_correct_given_answer_correct"])
        self.assertEqual(summary["clean"]["answer_correct_count"], 0)


def make_group_record(group_id: str, *, question: str = "What was revenue?", answer: str = "125.4") -> dict:
    payload = make_payload(answer={"raw": answer})
    return {"group_id": group_id, "question": question, "variants": {"clean": payload}}


class SourceIndexFromGroupIdTests(unittest.TestCase):
    def test_extracts_source_index(self) -> None:
        self.assertEqual(source_index_from_group_id("syn_000938_qDFEJXHk3Pm"), "000938")

    def test_rejects_unrecognized_format(self) -> None:
        with self.assertRaises(ValueError):
            source_index_from_group_id("not-a-group-id")


class CheckSplitLeakageTests(unittest.TestCase):
    def test_passes_and_reports_counts_for_disjoint_splits(self) -> None:
        train = [make_group_record("syn_000001_a"), make_group_record("syn_000002_a")]
        dev = [make_group_record("syn_000003_a")]
        test = [make_group_record("syn_000004_a")]

        counts = check_split_leakage(train, dev, test)

        self.assertEqual(counts, {"train_source_tables": 2, "dev_source_tables": 1, "test_source_tables": 1})

    def test_raises_on_train_dev_overlap(self) -> None:
        train = [make_group_record("syn_000001_a")]
        dev = [make_group_record("syn_000001_b")]  # same source_index 000001, different question slug
        test: list[dict] = []

        with self.assertRaises(ValueError) as ctx:
            check_split_leakage(train, dev, test)
        self.assertIn("train_dev", str(ctx.exception))

    def test_raises_on_dev_test_overlap(self) -> None:
        train: list[dict] = []
        dev = [make_group_record("syn_000005_a")]
        test = [make_group_record("syn_000005_b")]

        with self.assertRaises(ValueError) as ctx:
            check_split_leakage(train, dev, test)
        self.assertIn("dev_test", str(ctx.exception))


class ValidateAndBuildExamplesTests(unittest.TestCase):
    def test_builds_examples_for_all_valid_groups(self) -> None:
        records = [make_group_record("syn_000001_a"), make_group_record("syn_000002_a")]

        examples, stats = validate_and_build_examples(records)

        self.assertEqual(len(examples), 2)
        self.assertEqual(stats["total_groups"], 2)
        self.assertEqual(stats["valid_groups"], 2)

    def test_counts_invalid_groups_by_reason_without_raising(self) -> None:
        good = make_group_record("syn_000001_a")
        bad = make_group_record("syn_000002_a")
        bad["variants"]["clean"]["evidence"]["row_header_ids"] = []

        examples, stats = validate_and_build_examples([good, bad])

        self.assertEqual(len(examples), 1)
        self.assertEqual(stats["total_groups"], 2)
        self.assertEqual(stats["valid_groups"], 1)
        self.assertEqual(stats["invalid__missing_row_header"], 1)

    def test_duplicate_group_id_with_identical_annotations_is_counted_not_raised(self) -> None:
        record = make_group_record("syn_000001_a")
        duplicate = make_group_record("syn_000001_a")

        examples, stats = validate_and_build_examples([record, duplicate])

        self.assertEqual(len(examples), 1)
        self.assertEqual(stats["duplicate_group_id_skipped"], 1)

    def test_duplicate_group_id_with_contradictory_annotations_raises(self) -> None:
        record = make_group_record("syn_000001_a", answer="125.4")
        contradictory = make_group_record("syn_000001_a", answer="999.9")

        with self.assertRaises(ValueError) as ctx:
            validate_and_build_examples([record, contradictory])
        self.assertIn("syn_000001_a", str(ctx.exception))
        self.assertIn("contradictory", str(ctx.exception))


class EvidenceFirstPromptTests(unittest.TestCase):
    def test_prompt_includes_fixed_instruction_and_question_only(self) -> None:
        prompt = evidence_first_prompt("What was revenue in FY2023?")
        self.assertIn(EVIDENCE_FIRST_PROMPT_INSTRUCTION, prompt)
        self.assertIn("What was revenue in FY2023?", prompt)

    def test_prompt_rejects_empty_question(self) -> None:
        with self.assertRaises(ValueError):
            evidence_first_prompt("  ")


class FormatEvidenceFirstTargetTests(unittest.TestCase):
    def test_bbox_appears_before_answer(self) -> None:
        text = format_evidence_first_target("125.4", (403, 271, 492, 309))
        self.assertEqual(text, "<bbox>403 271 492 309</bbox>\n<answer>125.4</answer>")
        self.assertLess(text.index("<bbox>"), text.index("<answer>"))

    def test_no_row_or_column_tags_present(self) -> None:
        text = format_evidence_first_target("125.4", (403, 271, 492, 309))
        self.assertNotIn("<row>", text)
        self.assertNotIn("<column>", text)


class ParseEvidenceFirstPredictionTests(unittest.TestCase):
    def test_parses_well_formed_bbox_first_prediction(self) -> None:
        text = "<bbox>403 271 492 309</bbox>\n<answer>125.4</answer>"
        parsed = parse_evidence_first_prediction(text)

        self.assertEqual(parsed.bbox, (403, 271, 492, 309))
        self.assertTrue(parsed.bbox_valid)
        self.assertEqual(parsed.answer, "125.4")
        self.assertTrue(parsed.bbox_before_answer)

    def test_detects_wrong_order_if_model_reverts_to_answer_first(self) -> None:
        text = "<answer>125.4</answer>\n<bbox>403 271 492 309</bbox>"
        parsed = parse_evidence_first_prediction(text)

        self.assertTrue(parsed.bbox_valid)
        self.assertEqual(parsed.answer, "125.4")
        self.assertFalse(parsed.bbox_before_answer)

    def test_missing_tags_become_none_without_raising(self) -> None:
        parsed = parse_evidence_first_prediction("no tags here")
        self.assertIsNone(parsed.bbox)
        self.assertIsNone(parsed.answer)
        self.assertIsNone(parsed.bbox_before_answer)

    def test_no_row_column_tags_expected_or_parsed(self) -> None:
        # sanity: this parser has no row/column concept at all
        parsed = parse_evidence_first_prediction("<bbox>1 1 2 2</bbox><answer>5</answer>")
        self.assertFalse(hasattr(parsed, "row"))
        self.assertFalse(hasattr(parsed, "column"))


class SummarizeEvidenceFirstGroundingMetricsTests(unittest.TestCase):
    def _row(self, variant: str, group_id: str, *, answer_correct: bool, cell_correct: bool, iou: float) -> dict:
        return {
            "variant": variant,
            "group_id": group_id,
            "answer_correct": answer_correct,
            "cell_correct": cell_correct,
            "bbox_iou": iou,
        }

    def test_confusion_counts_and_conditional_probabilities(self) -> None:
        rows = [
            self._row("clean", "g1", answer_correct=True, cell_correct=True, iou=0.9),
            self._row("clean", "g2", answer_correct=True, cell_correct=False, iou=0.1),
            self._row("clean", "g3", answer_correct=False, cell_correct=True, iou=0.8),
            self._row("clean", "g4", answer_correct=False, cell_correct=False, iou=0.0),
        ]

        summary = summarize_evidence_first_grounding_metrics(rows)
        clean = summary["clean"]

        self.assertEqual(clean["total"], 4)
        self.assertAlmostEqual(clean["answer_accuracy"], 0.5)
        self.assertAlmostEqual(clean["cell_acc_at_1"], 0.5)
        self.assertEqual(
            clean["confusion_counts"],
            {
                "cell_correct_answer_correct": 1,
                "cell_correct_answer_wrong": 1,
                "cell_wrong_answer_correct": 1,
                "cell_wrong_answer_wrong": 1,
            },
        )
        # P(cell_correct | answer_correct): of 2 answer-correct, 1 is cell-correct
        self.assertAlmostEqual(clean["p_cell_correct_given_answer_correct"], 0.5)
        # P(answer_correct | cell_correct): of 2 cell-correct, 1 is answer-correct
        self.assertAlmostEqual(clean["p_answer_correct_given_cell_correct"], 0.5)
        # P(answer_correct | cell_wrong): of 2 cell-wrong, 1 is answer-correct
        self.assertAlmostEqual(clean["p_answer_correct_given_cell_wrong"], 0.5)
        # P(cell_wrong | answer_wrong): of 2 answer-wrong, 1 is cell-wrong
        self.assertAlmostEqual(clean["p_cell_wrong_given_answer_wrong"], 0.5)

    def test_handles_all_zero_denominators_without_raising(self) -> None:
        rows = [self._row("clean", "g1", answer_correct=False, cell_correct=False, iou=0.0)]
        summary = summarize_evidence_first_grounding_metrics(rows)
        clean = summary["clean"]
        self.assertIsNone(clean["p_cell_correct_given_answer_correct"])
        self.assertIsNone(clean["p_answer_correct_given_cell_correct"])


class CategorizeHfrErrorsTests(unittest.TestCase):
    def test_buckets_by_cell_and_answer_correctness(self) -> None:
        rows = [
            {"variant": "header_address_swap", "group_id": "g1", "cell_correct": True, "answer_correct": True},
            {"variant": "header_address_swap", "group_id": "g2", "cell_correct": True, "answer_correct": False},
            {"variant": "header_address_swap", "group_id": "g3", "cell_correct": False, "answer_correct": True},
            {"variant": "header_address_swap", "group_id": "g4", "cell_correct": False, "answer_correct": False},
            {"variant": "clean", "group_id": "g5", "cell_correct": True, "answer_correct": True},  # excluded: wrong variant
        ]

        result = categorize_hfr_errors(rows)

        self.assertEqual(result["variant"], "header_address_swap")
        self.assertEqual(result["counts"]["A_correct_cell_correct_answer"], 1)
        self.assertEqual(result["counts"]["B_correct_cell_wrong_answer"], 1)
        self.assertEqual(result["counts"]["C_wrong_cell_correct_answer"], 1)
        self.assertEqual(result["counts"]["D_wrong_cell_wrong_answer"], 1)
        self.assertEqual(result["group_ids"]["A_correct_cell_correct_answer"], ["g1"])
        self.assertEqual(result["group_ids"]["B_correct_cell_wrong_answer"], ["g2"])
        self.assertEqual(result["group_ids"]["C_wrong_cell_correct_answer"], ["g3"])
        self.assertEqual(result["group_ids"]["D_wrong_cell_wrong_answer"], ["g4"])

    def test_supports_a_different_variant(self) -> None:
        rows = [{"variant": "target_value", "group_id": "g1", "cell_correct": True, "answer_correct": True}]
        result = categorize_hfr_errors(rows, variant="target_value")
        self.assertEqual(result["counts"]["A_correct_cell_correct_answer"], 1)


if __name__ == "__main__":
    unittest.main()
