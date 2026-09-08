from __future__ import annotations

import unittest

from financial_vlm.data.company_loader import (
    CompanyTrainCandidate,
    adapter_bbox_to_xyxy,
    build_company_clean_record,
    cap_per_document,
)


def make_candidate(document_id: str, metric: str = "sales") -> CompanyTrainCandidate:
    return CompanyTrainCandidate(
        document_id=document_id,
        page_id="page-0001",
        instance_index=0,
        metric=metric,
        period_text="December 2020",
        gold_answer="1,234",
        question=f"What was {metric} in December 2020?",
        bbox=(100, 50, 140, 400),  # (top, left, bottom, right), adapter convention
    )


class AdapterBboxTests(unittest.TestCase):
    def test_reorders_top_left_bottom_right_to_xyxy(self) -> None:
        top, left, bottom, right = 100, 50, 140, 400
        self.assertEqual(adapter_bbox_to_xyxy((top, left, bottom, right)), (left, top, right, bottom))


class BuildCompanyCleanRecordTests(unittest.TestCase):
    def test_produces_a_schema_valid_record(self) -> None:
        record = build_company_clean_record(
            document_id="doc-1",
            page_id="page-0001",
            instance_index=0,
            split="train",
            image_path="/path/to/private-dataset/files/doc-1/pages/page-0001/image.png",
            image_size=(800, 1000),
            metric="sales",
            period_text="December 2020",
            gold_answer="1,234",
            question="What was sales in December 2020?",
            bbox=(100, 50, 140, 400),  # (top, left, bottom, right)
            group_index=0,
        )

        self.assertEqual(record["source"], "company")
        self.assertEqual(record["template_family"], "kpi_table")
        self.assertEqual(record["document_id"], "doc-1")
        self.assertEqual(record["answer"]["raw"], "1,234")
        self.assertEqual(record["answer"]["normalised_value"], 1234.0)
        self.assertEqual(record["answer"]["metric"], "sales")
        self.assertEqual(record["answer"]["period"], "December 2020")
        self.assertEqual(record["evidence"]["type"], "region")
        self.assertEqual(record["evidence"]["target_id"], record["evidence_units"][0]["id"])
        self.assertEqual(len(record["evidence_units"]), 1)
        # bbox (top=100, left=50, bottom=140, right=400) on an 800x1000 image
        # -> xyxy (50, 100, 400, 140) -> normalised.
        self.assertEqual(
            record["evidence"]["bbox_normalised"],
            [round(50 / 800, 6), round(100 / 1000, 6), round(400 / 800, 6), round(140 / 1000, 6)],
        )

    def test_rejects_a_value_canonical_schema_cannot_parse_numerically(self) -> None:
        # "8.5x" passes the OCR-quality proxy's own smarter normalizer but
        # not canonical_schema.normalise_numeric_value's plainer regex --
        # build_company_train_v1.py filters these upstream, but the record
        # builder itself should also reject rather than silently accept
        # normalised_value=None (schema requires a real number).
        with self.assertRaises(ValueError):
            build_company_clean_record(
                document_id="doc-1",
                page_id="page-0001",
                instance_index=0,
                split="train",
                image_path="/path/to/private-dataset/files/doc-1/pages/page-0001/image.png",
                image_size=(800, 1000),
                metric="valuation_multiple",
                period_text="December 2020",
                gold_answer="8.5x",
                question="What was valuation_multiple in December 2020?",
                bbox=(100, 50, 140, 400),
                group_index=0,
            )


class CapPerDocumentTests(unittest.TestCase):
    def test_caps_each_document_independently(self) -> None:
        candidates = [make_candidate("doc-A") for _ in range(10)] + [make_candidate("doc-B") for _ in range(3)]

        capped = cap_per_document(candidates, max_per_document=4, shuffle_seed=1)

        counts: dict[str, int] = {}
        for candidate in capped:
            counts[candidate.document_id] = counts.get(candidate.document_id, 0) + 1
        self.assertEqual(counts["doc-A"], 4)
        self.assertEqual(counts["doc-B"], 3)

    def test_deterministic_for_a_fixed_seed(self) -> None:
        candidates = [make_candidate("doc-A", metric=f"m{i}") for i in range(20)]

        first = cap_per_document(candidates, max_per_document=5, shuffle_seed=42)
        second = cap_per_document(candidates, max_per_document=5, shuffle_seed=42)

        self.assertEqual([c.metric for c in first], [c.metric for c in second])

    def test_different_seeds_can_select_different_subsets(self) -> None:
        candidates = [make_candidate("doc-A", metric=f"m{i}") for i in range(20)]

        first = cap_per_document(candidates, max_per_document=5, shuffle_seed=1)
        second = cap_per_document(candidates, max_per_document=5, shuffle_seed=2)

        self.assertNotEqual([c.metric for c in first], [c.metric for c in second])


if __name__ == "__main__":
    unittest.main()
