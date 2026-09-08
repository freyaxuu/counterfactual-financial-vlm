from __future__ import annotations

import unittest

from financial_vlm.evaluation.conditional_probabilities import summarize_conditional_probabilities


def pred(group_id: str, variant: str, correct: bool, setting: str = "full_page") -> dict[str, object]:
    return {
        "group_id": group_id,
        "setting": setting,
        "variant": variant,
        "numeric_correct": correct,
        "exact_correct": correct,
    }


class ConditionalProbabilityTests(unittest.TestCase):
    def test_summarizes_failures_conditioned_on_clean_correct_groups(self) -> None:
        records = [
            pred("g1", "clean", True),
            pred("g1", "target_value_replacement", False),
            pred("g1", "header_address_swap", True),
            pred("g1", "irrelevant_value_replacement", True),
            pred("g2", "clean", True),
            pred("g2", "target_value_replacement", True),
            pred("g2", "header_address_swap", False),
            pred("g2", "irrelevant_value_replacement", False),
            pred("g3", "clean", False),
            pred("g3", "target_value_replacement", False),
            pred("g3", "header_address_swap", False),
            pred("g3", "irrelevant_value_replacement", False),
        ]

        summary = summarize_conditional_probabilities(records)
        full_page = summary["by_setting"]["full_page"]

        self.assertEqual(full_page["groups_complete"], 3)
        self.assertEqual(full_page["clean_correct_groups"], 2)
        self.assertAlmostEqual(full_page["probabilities"]["P(T=0|C=1)"], 0.5)
        self.assertAlmostEqual(full_page["probabilities"]["P(H=0|C=1)"], 0.5)
        self.assertAlmostEqual(full_page["probabilities"]["P(T=0_or_H=0|C=1)"], 1.0)
        self.assertAlmostEqual(full_page["probabilities"]["P(I=0|C=1)"], 0.5)
        self.assertAlmostEqual(full_page["differences"]["P(H=0|C=1)-P(I=0|C=1)"], 0.0)
        self.assertAlmostEqual(full_page["differences"]["P(T=0|C=1)-P(I=0|C=1)"], 0.0)
        self.assertTrue(full_page["targeted_failure_exceeds_nuisance"]["any_targeted_gt_irrelevant"])

    def test_legacy_variant_names_are_canonicalized(self) -> None:
        records = [
            pred("g1", "clean", True),
            pred("g1", "target_value", False),
            pred("g1", "header_swap", False),
            pred("g1", "irrelevant_cell", True),
        ]

        full_page = summarize_conditional_probabilities(records)["by_setting"]["full_page"]

        self.assertEqual(full_page["groups_complete"], 1)
        self.assertAlmostEqual(full_page["probabilities"]["P(T=0|C=1)"], 1.0)
        self.assertAlmostEqual(full_page["probabilities"]["P(H=0|C=1)"], 1.0)
        self.assertAlmostEqual(full_page["probabilities"]["P(I=0|C=1)"], 0.0)


if __name__ == "__main__":
    unittest.main()
