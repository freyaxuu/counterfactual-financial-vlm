from __future__ import annotations

import unittest
from collections import Counter

from financial_vlm.evaluation.kpi_question_pairs import (
    QuestionPairRecord,
    build_question_text,
    metric_pair_automated_pass,
    period_label_text,
    stratified_cap_pairs,
)


def qp(
    pair_id: str,
    document_id: str = "doc1",
    page: str = "p1",
    metric: str = "sales",
    company_key: str = "company1",
) -> QuestionPairRecord:
    return QuestionPairRecord(
        pair_id=pair_id,
        pair_type="period",
        document_id=document_id,
        company_key=company_key,
        metric=metric,
        page_a=page,
        page_b=page,
        instance_a=0,
        instance_b=1,
        bbox_a=(0, 0, 1, 1),
        bbox_b=(0, 0, 1, 1),
        period_text_a="2021",
        period_text_b="2022",
        question_a="What was sales in 2021?",
        question_b="What was sales in 2022?",
        gold_answer_a="100",
        gold_answer_b="200",
        ocr_outcome_a="match",
        ocr_outcome_b="match",
        verification_status="automated_pass",
    )


class PeriodLabelTextTest(unittest.TestCase):
    def test_period_plus_normalized_month_year(self) -> None:
        # confirmed real issue, 2026-08-12: this used to produce the
        # unreadable "LTM 12 2022" -- month/year are now normalized to a
        # readable date before being appended to the period label.
        self.assertEqual(period_label_text("LTM", "12", "2022"), "LTM December 2022")

    def test_period_plus_year_only(self) -> None:
        self.assertEqual(period_label_text("LTM", None, "2022"), "LTM 2022")

    def test_year_only_two_digit_normalizes(self) -> None:
        self.assertEqual(period_label_text(None, None, "22"), "2022")

    def test_month_and_year_only_no_period(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: "12 20" (raw
        # concatenation) is now "December 2020".
        self.assertEqual(period_label_text(None, "12", "20"), "December 2020")

    def test_all_missing_falls_back_to_placeholder(self) -> None:
        self.assertEqual(period_label_text(None, None, None), "unspecified period")

    def test_period_only_no_date_parts(self) -> None:
        self.assertEqual(period_label_text("Prior Qtr", None, None), "Prior Qtr")

    def test_full_date_embedded_in_period_field_is_reformatted(self) -> None:
        # confirmed real raw-vocabulary pattern, 2026-08-12: some values
        # cram a full date into one field instead of splitting across
        # period/month/year.
        self.assertEqual(period_label_text("31.12.20", None, None), "December 2020")
        self.assertEqual(period_label_text("31/12/2020", None, None), "December 2020")

    def test_month_year_embedded_in_period_field_is_reformatted(self) -> None:
        self.assertEqual(period_label_text("12/20", None, None), "December 2020")

    def test_period_field_that_is_not_a_date_is_left_as_text(self) -> None:
        self.assertEqual(period_label_text("Prior Qtr", "12", "2022"), "Prior Qtr December 2022")

    def test_unparseable_date_like_text_falls_back_to_raw_text(self) -> None:
        # looks date-shaped but month "13" doesn't normalize -- must not
        # silently fabricate a date; falls back to treating it as a label.
        self.assertEqual(period_label_text("13/20", None, None), "13/20")


class BuildQuestionTextTest(unittest.TestCase):
    def test_formats_question(self) -> None:
        self.assertEqual(build_question_text("EBITDA", "LTM 2022"), "What was EBITDA in LTM 2022?")


class MetricPairAutomatedPassTest(unittest.TestCase):
    def test_both_match_passes(self) -> None:
        self.assertTrue(metric_pair_automated_pass("match", "match"))

    def test_word_set_and_scale_variants_pass(self) -> None:
        self.assertTrue(metric_pair_automated_pass("match_word_set", "match_scale_or_sign"))

    def test_one_side_mismatch_fails(self) -> None:
        self.assertFalse(metric_pair_automated_pass("match", "mismatch"))

    def test_low_confidence_mismatch_does_not_pass(self) -> None:
        # deliberately conservative -- see module docstring: absence of
        # confirmation (even likely-OCR-failure absence) isn't confirmation.
        self.assertFalse(metric_pair_automated_pass("match", "mismatch_low_ocr_confidence"))

    def test_unparseable_or_no_ocr_do_not_pass(self) -> None:
        self.assertFalse(metric_pair_automated_pass("match", "annotation_unparseable"))
        self.assertFalse(metric_pair_automated_pass("match", "no_ocr_nearby"))


class StratifiedCapPairsTest(unittest.TestCase):
    def test_caps_per_page(self) -> None:
        pairs = [qp(f"p{i}", page="page-A") for i in range(10)]
        result = stratified_cap_pairs(pairs, max_per_page=3, max_per_company=100)
        self.assertEqual(len(result), 3)

    def test_caps_per_company(self) -> None:
        pairs = [qp(f"p{i}", page=f"page-{i}", company_key="acme") for i in range(10)]
        result = stratified_cap_pairs(pairs, max_per_page=100, max_per_company=4)
        self.assertEqual(len(result), 4)

    def test_caps_per_metric_when_given(self) -> None:
        pairs = [qp(f"p{i}", page=f"page-{i}", company_key=f"co{i}", metric="sales") for i in range(10)]
        result = stratified_cap_pairs(pairs, max_per_page=100, max_per_company=100, max_per_metric=2)
        self.assertEqual(len(result), 2)

    def test_no_metric_cap_by_default(self) -> None:
        pairs = [qp(f"p{i}", page=f"page-{i}", company_key=f"co{i}", metric="sales") for i in range(10)]
        result = stratified_cap_pairs(pairs, max_per_page=100, max_per_company=100)
        self.assertEqual(len(result), 10)

    def test_under_cap_keeps_everything(self) -> None:
        pairs = [qp(f"p{i}", page=f"page-{i}", company_key=f"co{i}") for i in range(3)]
        result = stratified_cap_pairs(pairs, max_per_page=5, max_per_company=5)
        self.assertEqual(len(result), 3)

    def test_deterministic_for_a_given_seed(self) -> None:
        pairs = [qp(f"p{i}", page="page-A") for i in range(20)]
        first = stratified_cap_pairs(pairs, max_per_page=3, max_per_company=100, shuffle_seed=42)
        second = stratified_cap_pairs(pairs, max_per_page=3, max_per_company=100, shuffle_seed=42)
        self.assertEqual([p.pair_id for p in first], [p.pair_id for p in second])

    def test_no_single_dimension_exceeds_its_cap_under_combined_constraints(self) -> None:
        # a more realistic mixed population -- verify all three caps hold
        # simultaneously, not just whichever one is tested in isolation.
        pairs = []
        for doc_idx in range(3):
            for page_idx in range(5):
                for i in range(20):
                    pairs.append(
                        qp(
                            f"d{doc_idx}p{page_idx}i{i}",
                            document_id=f"doc{doc_idx}",
                            page=f"page{page_idx}",
                            company_key=f"co{doc_idx}",
                            metric=["sales", "EBITDA"][i % 2],
                        )
                    )
        result = stratified_cap_pairs(pairs, max_per_page=3, max_per_company=10, max_per_metric=15)
        page_counts = Counter((p.document_id, p.page_a) for p in result)
        company_counts = Counter(p.company_key for p in result)
        metric_counts = Counter(p.metric for p in result)
        self.assertTrue(all(c <= 3 for c in page_counts.values()))
        self.assertTrue(all(c <= 10 for c in company_counts.values()))
        self.assertTrue(all(c <= 15 for c in metric_counts.values()))


if __name__ == "__main__":
    unittest.main()
