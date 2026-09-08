from __future__ import annotations

import unittest

from financial_vlm.data.tatdqa_pilot import (
    build_doc_index,
    evidence_packet_to_json,
    extract_metric_tokens,
    extract_period_candidates,
    is_direct_extraction_question,
    iter_qa_records,
    locate_evidence,
    match_metric_period,
)


def synthetic_doc():
    return {
        "pages": [
            {
                "bbox": [0, 0, 600, 800],
                "blocks": [
                    {
                        "uuid": "block-1",
                        "bbox": [50, 100, 300, 130],
                        "text": "Revenue was 1,234 in 2020.",
                        "order": 1,
                        "words": {
                            "word_list": ["Revenue", "was", "1,234", "in", "2020."],
                            "bbox_list": [
                                [50, 100, 105, 120],
                                [112, 100, 140, 120],
                                [148, 100, 190, 120],
                                [198, 100, 210, 120],
                                [218, 100, 260, 120],
                            ],
                        },
                    }
                ],
            }
        ]
    }


class TatDqaPilotTests(unittest.TestCase):
    def test_accepts_single_fact_span_question(self) -> None:
        question = {
            "uid": "q1",
            "question": "What was revenue?",
            "answer": "1,234",
            "answer_type": "span",
            "facts": ["1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
            "req_comparison": False,
        }

        ok, reason = is_direct_extraction_question(question)

        self.assertTrue(ok)
        self.assertEqual(reason, "accepted")

    def test_accepts_zero_numeric_answer(self) -> None:
        question = {
            "answer": 0,
            "answer_type": "span",
            "facts": [0],
            "block_mapping": [{"block-1": [12, 12]}],
            "req_comparison": False,
        }

        ok, reason = is_direct_extraction_question(question)

        self.assertTrue(ok)
        self.assertEqual(reason, "accepted")

    def test_accepts_single_item_answer_list(self) -> None:
        question = {
            "answer": ["1,234"],
            "answer_type": "span",
            "facts": ["Revenue", "1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
            "req_comparison": False,
        }

        ok, reason = is_direct_extraction_question(question)

        self.assertTrue(ok)
        self.assertEqual(reason, "accepted")

    def test_rejects_multi_step_arithmetic(self) -> None:
        question = {
            "answer": "234",
            "answer_type": "arithmetic",
            "derivation": "1,234 - 1,000",
            "facts": ["1,234"],
            "req_comparison": False,
        }

        ok, reason = is_direct_extraction_question(question)

        self.assertFalse(ok)
        self.assertEqual(reason, "arithmetic_derivation")

    def test_locates_unique_block_mapping_as_evidence_packet(self) -> None:
        blocks, _ = build_doc_index(synthetic_doc())
        question = {
            "answer": ["1,234"],
            "facts": ["1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
        }

        evidence, reason = locate_evidence(question, blocks)

        self.assertEqual(reason, "accepted")
        self.assertIsNotNone(evidence)
        assert evidence is not None
        self.assertEqual(evidence.bbox, (148, 100, 190, 120))
        self.assertEqual(evidence.text, "1,234")
        packet = evidence_packet_to_json(evidence)
        self.assertEqual(packet["value_cell"]["label"], "word_span_region")
        self.assertEqual(packet["all_cells"][0]["block_uuid"], "block-1")

    def test_iter_qa_records_reads_document_records(self) -> None:
        raw = [{"doc": {"uid": "doc-1"}, "questions": [{"uid": "q1"}, {"uid": "q2"}]}]

        records = list(iter_qa_records(raw))

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], 0)
        self.assertEqual(records[0][1]["uid"], "doc-1")
        self.assertEqual(records[1][3]["uid"], "q2")

    def test_extracts_metric_and_period_candidates(self) -> None:
        question = "What was total revenue for the year ended December 31, 2020?"

        self.assertEqual(extract_metric_tokens(question), ("total", "revenue"))
        self.assertIn("December 31, 2020", extract_period_candidates(question))
        self.assertIn("2020", extract_period_candidates(question))

    def test_accepts_metric_period_match(self) -> None:
        blocks, _ = build_doc_index(synthetic_doc())
        question = {
            "question": "What was revenue in 2020?",
            "answer": "1,234",
            "facts": ["1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
        }
        evidence, _ = locate_evidence(question, blocks)
        assert evidence is not None

        match = match_metric_period(question, evidence, blocks)

        self.assertTrue(match.accepted)
        self.assertEqual(match.metric_candidate, "revenue")
        self.assertEqual(match.matched_metric_tokens, ("revenue",))
        self.assertEqual(match.matched_periods, ("2020",))

    def test_rejects_metric_period_mismatch(self) -> None:
        blocks, _ = build_doc_index(synthetic_doc())
        question = {
            "question": "What was operating income in 2020?",
            "answer": "1,234",
            "facts": ["1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
        }
        evidence, _ = locate_evidence(question, blocks)
        assert evidence is not None

        match = match_metric_period(question, evidence, blocks)

        self.assertFalse(match.accepted)
        self.assertEqual(match.reason, "metric_not_matched")


if __name__ == "__main__":
    unittest.main()
