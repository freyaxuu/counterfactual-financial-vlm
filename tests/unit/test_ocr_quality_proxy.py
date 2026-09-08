from __future__ import annotations

import unittest

from financial_vlm.evaluation.ocr_quality_proxy import (
    check_value_against_ocr,
    normalize_numeric,
    normalize_text,
    summarize_outcomes,
    tokens_near_bbox,
)
from financial_vlm.integrations.evolution_ai_datasets_adapter import OCRToken


def tok(text: str, top: int, left: int, bottom: int, right: int, confidence: int = 99) -> OCRToken:
    return OCRToken(top=top, left=left, bottom=bottom, right=right, text=text, confidence=confidence)


class NormalizeNumericTest(unittest.TestCase):
    def test_strips_currency_and_commas(self) -> None:
        self.assertEqual(normalize_numeric("$1,200.50"), 1200.5)

    def test_parens_mean_negative(self) -> None:
        self.assertEqual(normalize_numeric("(500)"), -500.0)

    def test_unparseable(self) -> None:
        self.assertIsNone(normalize_numeric("n/a"))

    def test_strips_magnitude_letter_suffix(self) -> None:
        # confirmed bug, 2026-08-12: EV values like "$1.2B" were falling
        # into "annotation_unparseable" even when correct -- OCR reads the
        # same suffixed form nearby, so this must strip (not convert) it to
        # stay comparable with `_NUMERIC_RE`'s bare-number extraction.
        self.assertEqual(normalize_numeric("$1.2B"), 1.2)
        self.assertEqual(normalize_numeric("3.4Mn"), 3.4)
        self.assertEqual(normalize_numeric("270.0"), 270.0)

    def test_strips_non_dollar_currency_symbols(self) -> None:
        # confirmed bug, 2026-08-12: EV/sales annotated in EUR/JPY ("€
        # 6,276,908", "¥ 4.3") fell into annotation_unparseable -- only "$"
        # was stripped.
        self.assertEqual(normalize_numeric("€ 6,276,908"), 6276908.0)
        self.assertEqual(normalize_numeric("¥ 4.3"), 4.3)
        self.assertEqual(normalize_numeric("£1,000"), 1000.0)

    def test_strips_multiple_notation_suffix(self) -> None:
        # confirmed bug, 2026-08-12: valuation_multiple values like "8.5x"
        # fell into annotation_unparseable.
        self.assertEqual(normalize_numeric("8.5x"), 8.5)
        self.assertEqual(normalize_numeric("8.5X"), 8.5)


class NormalizeTextTest(unittest.TestCase):
    def test_casefold_and_collapse_whitespace(self) -> None:
        self.assertEqual(normalize_text("  Acme   Corp "), "acme corp")


class TokensNearBboxTest(unittest.TestCase):
    def test_includes_tokens_with_center_inside_expanded_box(self) -> None:
        tokens = [tok("100", 10, 10, 20, 30), tok("far", 500, 500, 510, 530)]
        near = tokens_near_bbox(tokens, field_bbox=(5, 5, 25, 35), margin=2)
        self.assertEqual([t.text for t in near], ["100"])

    def test_sorts_by_left_coordinate(self) -> None:
        tokens = [tok("b", 10, 50, 20, 60), tok("a", 10, 10, 20, 20)]
        near = tokens_near_bbox(tokens, field_bbox=(0, 0, 100, 100))
        self.assertEqual([t.text for t in near], ["a", "b"])

    def test_excludes_tokens_outside_margin(self) -> None:
        tokens = [tok("far", 200, 200, 210, 230)]
        near = tokens_near_bbox(tokens, field_bbox=(0, 0, 10, 10), margin=2)
        self.assertEqual(near, [])


class CheckValueAgainstOcrTest(unittest.TestCase):
    def test_no_tokens_nearby(self) -> None:
        self.assertEqual(check_value_against_ocr("100", "monetary", []), "no_ocr_nearby")

    def test_monetary_exact_match(self) -> None:
        near = [tok("1,200", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("1200", "monetary", near), "match")

    def test_monetary_scale_mismatch(self) -> None:
        near = [tok("1.2", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("1200000", "monetary", near), "match_scale_or_sign")

    def test_monetary_genuine_mismatch(self) -> None:
        near = [tok("999", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("100", "monetary", near), "mismatch")

    def test_monetary_unparseable_annotation(self) -> None:
        near = [tok("100", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("n/a", "monetary", near), "annotation_unparseable")

    def test_monetary_magnitude_suffix_matches_bare_ocr_number(self) -> None:
        near = [tok("1.2", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("$1.2B", "monetary", near), "match")

    def test_monetary_dropped_decimal_point_is_scale_variant_not_mismatch(self) -> None:
        # confirmed bug, 2026-08-12: OCR misread "22.4" as "224" (missing the
        # decimal point) -- that's an OCR failure, not an annotation error.
        near = [tok("224", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("22.4", "monetary", near), "match_scale_or_sign")

    def test_monetary_comma_split_across_ocr_tokens_still_matches(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: annotation, bbox, and
        # OCR all agreed on "1,852", but OCR read it as two separate word
        # tokens ("1," and "852"), which broke naive number extraction.
        near = [tok("1,", 0, 0, 10, 15), tok("852", 0, 20, 10, 40)]
        self.assertEqual(check_value_against_ocr("1,852", "monetary", near), "match")

    def test_monetary_high_confidence_mismatch_stays_plain_mismatch(self) -> None:
        near = [tok("999", 0, 0, 10, 30, confidence=99)]
        self.assertEqual(check_value_against_ocr("100", "monetary", near), "mismatch")

    def test_monetary_low_confidence_mismatch_is_flagged_separately(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: "Q1"/"83.0" annotations
        # were correct on inspection -- OCR read low-confidence garbage
        # ("Qr723" cf=26, "BO" cf=46) nearby instead. A low-confidence
        # mismatch says more about OCR than the annotation.
        near = [tok("BO", 0, 0, 10, 30, confidence=46)]
        self.assertEqual(check_value_against_ocr("83.0", "monetary", near), "mismatch_low_ocr_confidence")

    def test_text_low_confidence_mismatch_is_flagged_separately(self) -> None:
        near = [tok("Qr723", 0, 0, 10, 30, confidence=26)]
        self.assertEqual(check_value_against_ocr("Q1", "text", near), "mismatch_low_ocr_confidence")

    def test_monetary_sign_flip_is_scale_variant(self) -> None:
        near = [tok("500", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("-500", "monetary", near), "match_scale_or_sign")

    def test_text_exact_match(self) -> None:
        near = [tok("Acme", 0, 0, 10, 30), tok("Corp", 0, 30, 10, 60)]
        self.assertEqual(check_value_against_ocr("Acme Corp", "text", near), "match")

    def test_text_word_set_match_different_order(self) -> None:
        near = [tok("Corp", 0, 0, 10, 30), tok("Acme", 0, 30, 10, 60)]
        self.assertIn(check_value_against_ocr("Acme Corp", "text", near), ("match", "match_word_set"))

    def test_text_mismatch(self) -> None:
        near = [tok("Globex", 0, 0, 10, 30)]
        self.assertEqual(check_value_against_ocr("Acme Corp", "text", near), "mismatch")


class SummarizeOutcomesTest(unittest.TestCase):
    def test_aggregates_overall_and_by_field(self) -> None:
        outcomes = [("sales", "match"), ("sales", "mismatch"), ("EBITDA", "match")]
        result = summarize_outcomes(outcomes)
        self.assertEqual(result["sample_n"], 3)
        self.assertEqual(result["outcome_counts"]["match"], 2)
        self.assertEqual(result["outcome_by_field"]["sales"], {"match": 1, "mismatch": 1})

    def test_empty(self) -> None:
        result = summarize_outcomes([])
        self.assertEqual(result["sample_n"], 0)
        self.assertEqual(result["outcome_rates"], {})


if __name__ == "__main__":
    unittest.main()
