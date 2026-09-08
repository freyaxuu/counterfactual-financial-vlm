from __future__ import annotations

import unittest

from financial_vlm.training.standard_qa import (
    answer_target,
    filter_records_by_variant,
    normalize_interval_strategy,
    standard_qa_prompt,
)


class StandardQATrainingTests(unittest.TestCase):
    def test_standard_prompt_uses_question_without_evidence_metadata(self) -> None:
        prompt = standard_qa_prompt("What was revenue in 2022?")

        self.assertIn("What was revenue in 2022?", prompt)
        self.assertIn("Return only the answer value", prompt)
        self.assertNotIn("cell", prompt.casefold())
        self.assertNotIn("bbox", prompt.casefold())
        self.assertNotIn("evidence", prompt.casefold())

    def test_answer_target_reads_canonical_answer_raw(self) -> None:
        self.assertEqual(answer_target({"answer": {"raw": "1,234"}}), "1,234")

    def test_filter_records_treats_missing_variant_as_clean(self) -> None:
        records = [
            {"group_id": "g1", "answer": "100"},
            {"group_id": "g2", "variant": "target_value", "answer": "111"},
        ]

        filtered = filter_records_by_variant(records, ["clean"])

        self.assertEqual([record["group_id"] for record in filtered], ["g1"])

    def test_normalizes_yaml_boolean_interval_strategy(self) -> None:
        self.assertEqual(normalize_interval_strategy(False), "no")
        self.assertEqual(normalize_interval_strategy(True), "steps")
        self.assertEqual(normalize_interval_strategy("epoch"), "epoch")


if __name__ == "__main__":
    unittest.main()
