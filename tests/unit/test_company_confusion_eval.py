from __future__ import annotations

import unittest

from financial_vlm.evaluation.company_confusion_eval import (
    PairOutcome,
    QuestionPrediction,
    ScoredQuestion,
    collapsed_pair_rate,
    competing_fact_capture_rate,
    conditional_capture_rate,
    evaluate_pair,
    is_captured,
    is_collapsed,
    micro_pair_accuracy,
    question_accuracy,
)


def pair(pair_id: str, document_id: str = "doc1", page_a: str = "p1", gold_a: str = "100", gold_b: str = "200") -> dict:
    return {
        "pair_id": pair_id,
        "pair_type": "period",
        "document_id": document_id,
        "page_a": page_a,
        "gold_answer_a": gold_a,
        "gold_answer_b": gold_b,
    }


class IsCapturedTest(unittest.TestCase):
    def test_matches_other_side_numerically(self) -> None:
        self.assertTrue(is_captured("200", "200"))
        self.assertTrue(is_captured("$200.0", "200"))

    def test_none_prediction_not_captured(self) -> None:
        self.assertFalse(is_captured(None, "200"))

    def test_unrelated_value_not_captured(self) -> None:
        self.assertFalse(is_captured("999", "200"))


class IsCollapsedTest(unittest.TestCase):
    def test_same_normalized_value_is_collapsed(self) -> None:
        self.assertTrue(is_collapsed("100", "$100.00"))

    def test_different_values_not_collapsed(self) -> None:
        self.assertFalse(is_collapsed("100", "200"))

    def test_missing_prediction_not_collapsed(self) -> None:
        self.assertFalse(is_collapsed(None, "100"))


class EvaluatePairTest(unittest.TestCase):
    def test_both_correct(self) -> None:
        p = pair("pair1")
        preds = {
            "pair1_A": QuestionPrediction(prediction="100", numeric_correct=True),
            "pair1_B": QuestionPrediction(prediction="200", numeric_correct=True),
        }
        outcome = evaluate_pair(p, preds)
        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.pair_correct)
        self.assertFalse(outcome.a_captured_by_b)
        self.assertFalse(outcome.b_captured_by_a)

    def test_confusion_swap_detected(self) -> None:
        # model answers side A's question with side B's gold value, and vice versa.
        p = pair("pair1")
        preds = {
            "pair1_A": QuestionPrediction(prediction="200", numeric_correct=False),
            "pair1_B": QuestionPrediction(prediction="100", numeric_correct=False),
        }
        outcome = evaluate_pair(p, preds)
        self.assertFalse(outcome.pair_correct)
        self.assertTrue(outcome.a_captured_by_b)
        self.assertTrue(outcome.b_captured_by_a)
        self.assertFalse(outcome.collapsed)  # different values from each other, just swapped

    def test_collapsed_both_same_wrong_value(self) -> None:
        p = pair("pair1")
        preds = {
            "pair1_A": QuestionPrediction(prediction="150", numeric_correct=False),
            "pair1_B": QuestionPrediction(prediction="150", numeric_correct=False),
        }
        outcome = evaluate_pair(p, preds)
        self.assertTrue(outcome.collapsed)

    def test_missing_prediction_returns_none(self) -> None:
        p = pair("pair1")
        preds = {"pair1_A": QuestionPrediction(prediction="100", numeric_correct=True)}
        self.assertIsNone(evaluate_pair(p, preds))


class MicroPairAccuracyTest(unittest.TestCase):
    def test_fraction_of_fully_correct_pairs(self) -> None:
        outcomes = [
            PairOutcome("p1", "period", "d1", True, True, True, False, False, False),
            PairOutcome("p2", "period", "d1", True, False, False, False, False, False),
        ]
        self.assertEqual(micro_pair_accuracy(outcomes), 0.5)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(micro_pair_accuracy([]))


class CompetingFactCaptureRateTest(unittest.TestCase):
    def test_counts_both_sides(self) -> None:
        outcomes = [
            PairOutcome("p1", "period", "d1", False, False, False, True, True, False),
            PairOutcome("p2", "period", "d1", True, True, True, False, False, False),
        ]
        # 2 captured out of 2*2=4 total answers
        self.assertEqual(competing_fact_capture_rate(outcomes), 0.5)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(competing_fact_capture_rate([]))


class ConditionalCaptureRateTest(unittest.TestCase):
    def test_fraction_of_wrong_answers_that_are_the_swap(self) -> None:
        # 2 wrong answers total (a and b of p1), both captured
        outcomes = [PairOutcome("p1", "period", "d1", False, False, False, True, True, False)]
        self.assertEqual(conditional_capture_rate(outcomes), 1.0)

    def test_no_wrong_answers_is_none(self) -> None:
        outcomes = [PairOutcome("p1", "period", "d1", True, True, True, False, False, False)]
        self.assertIsNone(conditional_capture_rate(outcomes))


class CollapsedPairRateTest(unittest.TestCase):
    def test_fraction_collapsed(self) -> None:
        outcomes = [
            PairOutcome("p1", "period", "d1", False, False, False, False, False, True),
            PairOutcome("p2", "period", "d1", True, True, True, False, False, False),
        ]
        self.assertEqual(collapsed_pair_rate(outcomes), 0.5)


class QuestionAccuracyTest(unittest.TestCase):
    def test_fraction_correct(self) -> None:
        items = [ScoredQuestion("d1", True), ScoredQuestion("d1", False), ScoredQuestion("d2", True)]
        self.assertAlmostEqual(question_accuracy(items), 2 / 3)

    def test_empty_is_none(self) -> None:
        self.assertIsNone(question_accuracy([]))


class TatqaEvalFunctionReuseTest(unittest.TestCase):
    """The frozen Company evaluation protocol reuses
    `tatqa_natural_pairs_eval.semantic_switch_success_rate` and
    `.clustered_bootstrap_delta` directly against this module's
    `PairOutcome` (see `scripts/aggregate_company_confusion_results.py`),
    not a reimplementation. Confirms that duck-typing actually works at
    runtime -- both dataclasses expose `correct_a`/`correct_b`/
    `document_id`, but they are different classes, so this isn't guaranteed
    by the type checker alone."""

    def test_semantic_switch_success_rate_accepts_company_pair_outcomes(self) -> None:
        from financial_vlm.evaluation.tatqa_natural_pairs_eval import semantic_switch_success_rate

        outcomes = [
            PairOutcome("p1", "period", "d1", True, True, True, False, False, False),
            PairOutcome("p2", "period", "d1", True, False, False, False, False, False),
        ]
        # eligible transitions: p1 has 2 (both correct), p2 has 1 (A correct, source)
        # successful: p1's 2 transitions both succeed; p2's 1 transition fails (B wrong)
        self.assertAlmostEqual(semantic_switch_success_rate(outcomes), 2 / 3)

    def test_clustered_bootstrap_delta_accepts_company_pair_outcomes(self) -> None:
        from financial_vlm.evaluation.tatqa_natural_pairs_eval import clustered_bootstrap_delta

        baseline = [
            PairOutcome("p1", "period", "d1", True, True, True, False, False, False),
            PairOutcome("p2", "period", "d2", False, False, False, False, False, False),
        ]
        other = [
            PairOutcome("p1", "period", "d1", True, True, True, False, False, False),
            PairOutcome("p2", "period", "d2", True, True, True, False, False, False),
        ]
        result = clustered_bootstrap_delta(baseline, other, micro_pair_accuracy, n_resamples=100, seed=0)
        self.assertAlmostEqual(result["point_estimate"], 0.5)


if __name__ == "__main__":
    unittest.main()
