from __future__ import annotations

import unittest

from financial_vlm.data.tatdqa_loader import build_tatdqa_clean_record
from financial_vlm.data.tatdqa_pilot import build_doc_index, locate_evidence, match_metric_period
from tests.unit.test_tatdqa_pilot import synthetic_doc


class TatDqaLoaderTests(unittest.TestCase):
    def test_builds_canonical_clean_record(self) -> None:
        blocks, pages = build_doc_index(synthetic_doc())
        question = {
            "uid": "q1",
            "question": "What was revenue in 2020?",
            "answer": "1,234",
            "scale": "million",
            "facts": ["1,234"],
            "block_mapping": [{"block-1": [12, 16]}],
        }
        evidence, _ = locate_evidence(
            {
                "answer": "1,234",
                "facts": ["1,234"],
                "block_mapping": [{"block-1": [12, 16]}],
            },
            blocks,
        )
        assert evidence is not None
        metric_period = match_metric_period(question, evidence, blocks)

        record = build_tatdqa_clean_record(
            doc_uid="doc-1",
            split="test_gold",
            image_path="images/doc-1_p00.png",
            image_size=(600, 800),
            question=question,
            question_index=0,
            accepted_index=0,
            evidence=evidence,
            metric_period_match=metric_period,
        )

        self.assertEqual(record["source"], "tatdqa")
        self.assertEqual(record["template_family"], "public_page")
        self.assertEqual(record["answer"]["normalised_value"], 1234.0)
        self.assertEqual(record["answer"]["scale"], "million")
        self.assertEqual(record["answer"]["metric"], "revenue")
        self.assertEqual(record["answer"]["period"], "2020")
        self.assertEqual(record["evidence"]["target_id"], record["evidence_units"][0]["id"])
        self.assertEqual(record["evidence_units"][0]["bbox_normalised"], [0.246667, 0.125, 0.316667, 0.15])
        self.assertTrue(record["evidence_units"][0]["metadata"]["metric_period_match"]["accepted"])
        self.assertEqual(pages[0]["bbox"], [0, 0, 600, 800])


if __name__ == "__main__":
    unittest.main()
