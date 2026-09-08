from __future__ import annotations

import unittest

from financial_vlm.evaluation.tatqa_natural_pairs_eval import (
    ERROR_MODES,
    PairOutcome,
    QuestionPrediction,
    ScoredQuestion,
    UnexplainedWrongItem,
    build_table_value_index,
    classify_wrong_prediction,
    clustered_bootstrap_delta,
    clustered_bootstrap_delta_generic,
    collapsed_pair_rate,
    competing_fact_capture_rate,
    conditional_capture_rate,
    error_taxonomy,
    evaluate_pair,
    individual_accuracy,
    is_captured,
    is_collapsed,
    is_scale_shift_error,
    micro_pair_accuracy,
    other_cell_match_rate,
    pair_sides,
    permutation_null_other_cell_match,
    question_accuracy,
    question_key,
    rescue_regression,
    semantic_switch_success_rate,
    table_macro_metric,
    unexplained_wrong_items,
)


def make_record(pair_id="p1", document_id="doc-1", table_id="doc-1", answer_a="100", answer_b="200", cell_a="r1c1", cell_b="r2c1"):
    return {
        "pair_id": pair_id,
        "document_id": document_id,
        "table_id": table_id,
        "question_a": "What was Revenue in 2019?",
        "answer_a": answer_a,
        "evidence_a": {"cell_id": cell_a},
        "question_b": "What was Revenue in 2018?",
        "answer_b": answer_b,
        "evidence_b": {"cell_id": cell_b},
    }


class QuestionKeyTests(unittest.TestCase):
    def test_deterministic_and_sensitive_to_all_fields(self) -> None:
        k1 = question_key("doc-1", "What was Revenue in 2019?", "r1c1")
        k2 = question_key("doc-1", "What was Revenue in 2019?", "r1c1")
        self.assertEqual(k1, k2)
        self.assertNotEqual(k1, question_key("doc-2", "What was Revenue in 2019?", "r1c1"))
        self.assertNotEqual(k1, question_key("doc-1", "What was Revenue in 2020?", "r1c1"))
        self.assertNotEqual(k1, question_key("doc-1", "What was Revenue in 2019?", "r1c2"))


class PairSidesTests(unittest.TestCase):
    def test_extracts_both_sides(self) -> None:
        record = make_record()
        side_a, side_b = pair_sides(record)
        self.assertEqual(side_a.answer_raw, "100")
        self.assertEqual(side_b.answer_raw, "200")
        self.assertEqual(side_a.target_cell_id, "r1c1")
        self.assertNotEqual(side_a.question_key, side_b.question_key)


class CaptureAndCollapseTests(unittest.TestCase):
    def test_is_captured_numeric_match(self) -> None:
        self.assertTrue(is_captured("$200", "200"))
        self.assertTrue(is_captured("(200)", "-200"))
        self.assertFalse(is_captured("300", "200"))
        self.assertFalse(is_captured(None, "200"))

    def test_is_collapsed(self) -> None:
        self.assertTrue(is_collapsed("$150", "150.0"))
        self.assertFalse(is_collapsed("150", "160"))
        self.assertFalse(is_collapsed(None, "150"))


class EvaluatePairTests(unittest.TestCase):
    def test_pair_correct_when_both_sides_match_gold(self) -> None:
        record = make_record()
        side_a, side_b = pair_sides(record)
        preds = {
            side_a.question_key: QuestionPrediction(prediction="100", numeric_correct=True),
            side_b.question_key: QuestionPrediction(prediction="200", numeric_correct=True),
        }
        outcome = evaluate_pair(record, preds)
        self.assertIsNotNone(outcome)
        self.assertTrue(outcome.pair_correct)
        self.assertFalse(outcome.collapsed)
        self.assertFalse(outcome.a_captured_by_b)
        self.assertFalse(outcome.b_captured_by_a)

    def test_competing_fact_capture_detected(self) -> None:
        record = make_record(answer_a="100", answer_b="200")
        side_a, side_b = pair_sides(record)
        # Model answers A with B's gold value -- classic period/metric confusion.
        preds = {
            side_a.question_key: QuestionPrediction(prediction="200", numeric_correct=False),
            side_b.question_key: QuestionPrediction(prediction="200", numeric_correct=True),
        }
        outcome = evaluate_pair(record, preds)
        self.assertFalse(outcome.pair_correct)
        self.assertTrue(outcome.a_captured_by_b)
        self.assertFalse(outcome.b_captured_by_a)
        self.assertTrue(outcome.collapsed)  # both predictions normalize to "200"

    def test_missing_prediction_returns_none(self) -> None:
        record = make_record()
        outcome = evaluate_pair(record, {})
        self.assertIsNone(outcome)


class AggregateMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.outcomes = [
            PairOutcome("p1", "t1", "t1", True, True, True, False, False, False),
            PairOutcome("p2", "t1", "t1", True, False, False, False, True, False),
            PairOutcome("p3", "t2", "t2", False, False, False, True, True, True),
        ]

    def test_individual_accuracy(self) -> None:
        preds = [
            QuestionPrediction("100", True),
            QuestionPrediction("200", True),
            QuestionPrediction("300", False),
        ]
        self.assertAlmostEqual(individual_accuracy(preds), 2 / 3)
        self.assertIsNone(individual_accuracy([]))

    def test_micro_pair_accuracy(self) -> None:
        self.assertAlmostEqual(micro_pair_accuracy(self.outcomes), 1 / 3)

    def test_table_macro_averages_per_table_first(self) -> None:
        # t1: [True, False] -> 0.5 ;  t2: [False] -> 0.0 ; macro mean = 0.25
        macro = table_macro_metric(self.outcomes, micro_pair_accuracy)
        self.assertAlmostEqual(macro, 0.25)

    def test_competing_fact_capture_rate(self) -> None:
        # captures: p1 (0,0) p2 (0,1) p3 (1,1) -> sum=3, denom=2*3=6
        self.assertAlmostEqual(competing_fact_capture_rate(self.outcomes), 0.5)

    def test_collapsed_pair_rate(self) -> None:
        self.assertAlmostEqual(collapsed_pair_rate(self.outcomes), 1 / 3)

    def test_semantic_switch_success_rate(self) -> None:
        # p1 (T,T): 2 eligible, 2 successful. p2 (T,F): 1 eligible (A source), 0 successful.
        # p3 (F,F): 0 eligible. Total: 2/3 eligible-successful.
        self.assertAlmostEqual(semantic_switch_success_rate(self.outcomes), 2 / 3)

    def test_semantic_switch_success_rate_edge_cases(self) -> None:
        all_wrong = [PairOutcome("p1", "t1", "t1", False, False, False, False, False, False)]
        self.assertIsNone(semantic_switch_success_rate(all_wrong))  # 0 eligible transitions
        all_right = [PairOutcome("p1", "t1", "t1", True, True, True, False, False, False)]
        self.assertAlmostEqual(semantic_switch_success_rate(all_right), 1.0)
        one_right_one_wrong = [PairOutcome("p1", "t1", "t1", True, False, False, False, False, False)]
        self.assertAlmostEqual(semantic_switch_success_rate(one_right_one_wrong), 0.0)

    def test_conditional_capture_rate(self) -> None:
        # wrong sides: p2.B (captured=True), p3.A (captured=True), p3.B (captured=True) -> 3 wrong, 3 captured
        self.assertAlmostEqual(conditional_capture_rate(self.outcomes), 1.0)

    def test_conditional_capture_rate_na_when_no_errors(self) -> None:
        all_correct = [PairOutcome("p1", "t1", "t1", True, True, True, False, False, False)]
        self.assertIsNone(conditional_capture_rate(all_correct))


class RescueRegressionTests(unittest.TestCase):
    def test_rates(self) -> None:
        baseline = {
            "p1": PairOutcome("p1", "t1", "t1", True, True, True, False, False, False),
            "p2": PairOutcome("p2", "t1", "t1", False, True, False, False, False, False),
            "p3": PairOutcome("p3", "t2", "t2", False, False, False, False, False, False),
            "p4": PairOutcome("p4", "t2", "t2", True, True, True, False, False, False),
        }
        other = {
            "p1": PairOutcome("p1", "t1", "t1", True, True, True, False, False, False),  # both correct
            "p2": PairOutcome("p2", "t1", "t1", False, True, False, False, False, False),  # baseline correct? no baseline wrong here -> both wrong actually baseline p2 pair_correct=False
            "p3": PairOutcome("p3", "t2", "t2", True, True, True, False, False, False),  # rescued
            "p4": PairOutcome("p4", "t2", "t2", False, True, False, False, False, False),  # regressed
        }
        counts = rescue_regression(baseline, other)
        self.assertEqual(counts.both_correct, 1)  # p1
        self.assertEqual(counts.baseline_correct_other_wrong, 1)  # p4
        self.assertEqual(counts.baseline_wrong_other_correct, 1)  # p3
        self.assertEqual(counts.both_wrong, 1)  # p2
        self.assertAlmostEqual(counts.rescue_rate, 1 / 2)
        self.assertAlmostEqual(counts.regression_rate, 1 / 2)


class BootstrapTests(unittest.TestCase):
    def test_improvement_detected_with_tight_ci(self) -> None:
        # 20 documents; "other" always correct, "baseline" always wrong -> delta should be ~1.0 with a tight CI.
        outcomes_a = [PairOutcome(f"p{i}", f"doc{i}", f"doc{i}", False, False, False, False, False, False) for i in range(20)]
        outcomes_b = [PairOutcome(f"p{i}", f"doc{i}", f"doc{i}", True, True, True, False, False, False) for i in range(20)]
        result = clustered_bootstrap_delta(outcomes_a, outcomes_b, micro_pair_accuracy, n_resamples=500, seed=1)
        self.assertAlmostEqual(result["point_estimate"], 1.0)
        self.assertGreater(result["ci_low"], 0.5)

    def test_no_shared_documents_returns_empty(self) -> None:
        outcomes_a = [PairOutcome("p1", "docA", "docA", True, True, True, False, False, False)]
        outcomes_b = [PairOutcome("p2", "docB", "docB", True, True, True, False, False, False)]
        result = clustered_bootstrap_delta(outcomes_a, outcomes_b, micro_pair_accuracy, n_resamples=100, seed=1)
        self.assertEqual(result["deltas"], [])

    def test_generic_bootstrap_on_scored_questions(self) -> None:
        items_a = [ScoredQuestion(f"doc{i}", False) for i in range(15)]
        items_b = [ScoredQuestion(f"doc{i}", True) for i in range(15)]
        result = clustered_bootstrap_delta_generic(
            items_a, items_b, lambda q: q.document_id, question_accuracy, n_resamples=300, seed=2
        )
        self.assertAlmostEqual(result["point_estimate"], 1.0)
        self.assertGreater(result["ci_low"], 0.5)


class ErrorTaxonomyTests(unittest.TestCase):
    def test_is_scale_shift_error_detects_powers_of_ten(self) -> None:
        self.assertTrue(is_scale_shift_error("246", "24.6"))
        self.assertTrue(is_scale_shift_error("147.40", "14,740"))
        self.assertTrue(is_scale_shift_error("2.55", "$255"))
        self.assertFalse(is_scale_shift_error("246", "24.7"))  # not a real 10x match
        self.assertFalse(is_scale_shift_error(None, "24.6"))
        self.assertFalse(is_scale_shift_error("246", "0"))  # no div-by-zero

    def test_is_scale_shift_error_false_when_actually_correct(self) -> None:
        self.assertFalse(is_scale_shift_error("24.6", "24.6"))

    def test_classify_wrong_prediction_priority(self) -> None:
        # prediction=1000 is BOTH the paired gold AND a 10x scale shift of
        # target=100 -- captured_by_pair must win the priority order.
        self.assertEqual(classify_wrong_prediction("1000", "100", "1000"), "captured_by_pair")
        self.assertEqual(classify_wrong_prediction("246", "24.6", "999"), "scale_shift")
        self.assertEqual(classify_wrong_prediction("999", "24.6", "111"), "other")

    def test_error_taxonomy_counts_across_both_sides(self) -> None:
        record = make_record(pair_id="p1", answer_a="100", answer_b="200")
        side_a, side_b = pair_sides(record)
        preds = {
            side_a.question_key: QuestionPrediction(prediction="200", numeric_correct=False),  # captured by B
            side_b.question_key: QuestionPrediction(prediction="2000", numeric_correct=False),  # scale shift of 200
        }
        counts = error_taxonomy([record], preds)
        self.assertEqual(counts["captured_by_pair"], 1)
        self.assertEqual(counts["scale_shift"], 1)
        self.assertEqual(counts["other"], 0)
        self.assertEqual(set(counts), set(ERROR_MODES))

    def test_error_taxonomy_skips_missing_predictions(self) -> None:
        record = make_record(pair_id="p1")
        self.assertEqual(
            error_taxonomy([record], {}),
            {"captured_by_pair": 0, "scale_shift": 0, "other_cell_in_table": 0, "other": 0},
        )

    def test_classify_wrong_prediction_other_cell_in_table(self) -> None:
        # Without table_values, an unexplained match falls into "other" (backward compatible).
        self.assertEqual(classify_wrong_prediction("777", "24.6", "999"), "other")
        # With table_values supplied, the same prediction is recognized as
        # matching some other real cell in the table.
        self.assertEqual(classify_wrong_prediction("777", "24.6", "999", table_values={"777", "111"}), "other_cell_in_table")
        # captured_by_pair still takes priority even when table_values would also match.
        self.assertEqual(classify_wrong_prediction("999", "24.6", "999", table_values={"999"}), "captured_by_pair")
        # scale_shift still takes priority over other_cell_in_table.
        self.assertEqual(classify_wrong_prediction("246", "24.6", "999", table_values={"246"}), "scale_shift")

    def test_error_taxonomy_with_table_values_populates_fourth_bucket(self) -> None:
        record = make_record(pair_id="p1", table_id="t1", answer_a="100", answer_b="200")
        side_a, side_b = pair_sides(record)
        preds = {
            side_a.question_key: QuestionPrediction(prediction="777", numeric_correct=False),  # matches other cell
            side_b.question_key: QuestionPrediction(prediction="999", numeric_correct=False),  # unexplained
        }
        table_values_by_table = {"t1": {"100", "200", "777"}}
        counts = error_taxonomy([record], preds, table_values_by_table)
        self.assertEqual(counts["other_cell_in_table"], 1)
        self.assertEqual(counts["other"], 1)

    def test_error_taxonomy_ignores_correct_sides(self) -> None:
        record = make_record(pair_id="p1", answer_a="100", answer_b="200")
        side_a, side_b = pair_sides(record)
        preds = {
            side_a.question_key: QuestionPrediction(prediction="100", numeric_correct=True),
            side_b.question_key: QuestionPrediction(prediction="999", numeric_correct=False),
        }
        counts = error_taxonomy([record], preds)
        self.assertEqual(sum(counts.values()), 1)
        self.assertEqual(counts["other"], 1)


class BroaderConfusionTests(unittest.TestCase):
    def test_build_table_value_index(self) -> None:
        records = [
            make_record(pair_id="p1", document_id="t1", table_id="t1", answer_a="100", answer_b="200"),
            make_record(pair_id="p2", document_id="t1", table_id="t1", answer_a="300", answer_b="200"),
            make_record(pair_id="p3", document_id="t2", table_id="t2", answer_a="999", answer_b="888"),
        ]
        index = build_table_value_index(records)
        self.assertEqual(index["t1"], {"100", "200", "300"})
        self.assertEqual(index["t2"], {"999", "888"})

    def test_unexplained_wrong_items_excludes_captured_and_scale_shift(self) -> None:
        record = make_record(pair_id="p1", table_id="t1", answer_a="100", answer_b="200")
        side_a, side_b = pair_sides(record)
        preds = {
            side_a.question_key: QuestionPrediction(prediction="200", numeric_correct=False),  # captured by B
            side_b.question_key: QuestionPrediction(prediction="1000", numeric_correct=False),  # scale shift of 100... wait target is 200
        }
        items = unexplained_wrong_items([record], preds)
        # side_a is captured_by_pair -> excluded. side_b: prediction "1000" vs target "200",
        # ratio 1000/200 = 5, not a recognized scale-shift ratio -> unexplained.
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].table_id, "t1")
        self.assertEqual(items[0].prediction_norm, "1000")

    def test_other_cell_match_rate(self) -> None:
        items = [
            UnexplainedWrongItem("t1", "300"),  # matches
            UnexplainedWrongItem("t1", "999"),  # no match
            UnexplainedWrongItem("t2", "888"),  # matches
        ]
        table_values = {"t1": {"100", "200", "300"}, "t2": {"999", "888"}}
        self.assertAlmostEqual(other_cell_match_rate(items, table_values), 2 / 3)
        self.assertIsNone(other_cell_match_rate([], table_values))

    def test_permutation_null_detects_genuine_signal(self) -> None:
        # Every item's prediction genuinely matches its OWN table's values --
        # a real signal that should sit far above the shuffled-null rate.
        table_values = {f"t{i}": {f"v{i}"} for i in range(30)}
        items = [UnexplainedWrongItem(f"t{i}", f"v{i}") for i in range(30)]
        result = permutation_null_other_cell_match(items, table_values, n_permutations=500, seed=3)
        self.assertAlmostEqual(result["observed_rate"], 1.0)
        self.assertLess(result["null_mean"], 0.5)
        self.assertLess(result["p_value_one_sided"], 0.05)

    def test_permutation_null_no_signal_when_matches_are_incidental(self) -> None:
        # Predictions never match any table's values at all -- both observed
        # and null rates should be ~0, p-value should NOT indicate significance.
        table_values = {f"t{i}": {f"v{i}"} for i in range(20)}
        items = [UnexplainedWrongItem(f"t{i}", "unrelated_value") for i in range(20)]
        result = permutation_null_other_cell_match(items, table_values, n_permutations=200, seed=4)
        self.assertEqual(result["observed_rate"], 0.0)
        self.assertEqual(result["null_mean"], 0.0)

    def test_permutation_null_empty_items(self) -> None:
        result = permutation_null_other_cell_match([], {}, n_permutations=50, seed=0)
        self.assertIsNone(result["observed_rate"])
        self.assertEqual(result["n_permutations"], 0)


if __name__ == "__main__":
    unittest.main()
