from __future__ import annotations

from collections import Counter
import unittest

from financial_vlm.data.tatqa_natural_pairs import (
    ADJACENT_RULE,
    METRIC_RELATEDNESS_JACCARD_THRESHOLD,
    NONADJACENT_RULE,
    PROTOCOL_VERSION,
    build_derived_pair_record,
    build_metric_candidates_for_table,
    build_pair_record,
    build_period_pairs_for_table,
    classify_pair,
    confusable_metric_flag,
    extract_eligible_question,
    extract_table_facts,
    is_bare_total_row,
    is_degenerate_metric_label,
    is_numeric_like_metric_label,
    lexical_jaccard,
    normalize_metric_text,
    period_sort_key,
    select_primary_period_pairs,
    strip_basis_qualifiers,
    temporal_remainder,
    unit_class,
    validate_natural_pairs,
)
from financial_vlm.data.tatqa_source_rerender import flatten_grid


def synthetic_doc():
    grid = [
        ["", "Years Ended December 31,", ""],
        ["", "2019", "2018"],
        ["Revenue", "1,234", "1,100"],
        ["EBITDA", "500", "480"],
        ["Adjusted EBITDA", "600", "550"],
    ]
    return {"table": {"uid": "table-1", "table": grid}}, grid


def question(uid, text, answer, scale=""):
    return {
        "uid": uid,
        "question": text,
        "answer": [answer],
        "derivation": "",
        "answer_type": "span",
        "answer_from": "table",
        "req_comparison": False,
        "scale": scale,
    }


def extract(doc, grid, q, index=0, split="test_gold"):
    cells = flatten_grid(grid)
    result, reason = extract_eligible_question(doc, q, index, cells, split)
    assert reason == "accepted", reason
    assert result is not None
    return result


class TatqaNaturalPairsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.doc, self.grid = synthetic_doc()
        self.revenue_2019 = extract(self.doc, self.grid, question("q1", "What was Revenue in 2019?", "1,234"), 0)
        self.revenue_2018 = extract(self.doc, self.grid, question("q2", "What was Revenue in 2018?", "1,100"), 1)
        self.ebitda_2019 = extract(self.doc, self.grid, question("q3", "What was EBITDA in 2019?", "500"), 2)
        self.adj_ebitda_2019 = extract(
            self.doc, self.grid, question("q4", "What was Adjusted EBITDA in 2019?", "600"), 3
        )

    def test_extract_eligible_question_reads_metric_and_period(self) -> None:
        self.assertEqual(self.revenue_2019.metric, "Revenue")
        self.assertEqual(self.revenue_2019.period, "2019")
        self.assertEqual(self.revenue_2019.normalized_answer, "1234")

    def test_extract_eligible_question_rejects_comparison(self) -> None:
        q = question("q5", "Compare revenue in 2019 and 2018", "1,234")
        q["req_comparison"] = True
        cells = flatten_grid(self.grid)
        result, reason = extract_eligible_question(self.doc, q, 4, cells, "test_gold")
        self.assertIsNone(result)
        self.assertEqual(reason, "rejected_comparison_required")

    def test_pair_across_different_splits_rejected(self) -> None:
        other_split = extract(
            self.doc, self.grid, question("q6", "What was Revenue in 2018?", "1,100"), 5, split="dev"
        )
        pair_type, reason = classify_pair(self.revenue_2019, other_split)
        self.assertIsNone(pair_type)
        self.assertEqual(reason, "rejected_different_split")

    def test_period_pair(self) -> None:
        pair_type, rule = classify_pair(self.revenue_2019, self.revenue_2018)
        self.assertEqual(pair_type, "period")
        self.assertEqual(rule, "same_metric_same_table_different_period")

    def test_metric_pair(self) -> None:
        pair_type, rule = classify_pair(self.revenue_2019, self.ebitda_2019)
        self.assertEqual(pair_type, "metric")
        self.assertEqual(rule, "same_period_same_table_different_metric")

    def test_basis_pair_takes_priority_over_metric(self) -> None:
        pair_type, rule = classify_pair(self.ebitda_2019, self.adj_ebitda_2019)
        self.assertEqual(pair_type, "basis")
        self.assertEqual(rule, "same_base_metric_qualifier_differs")

    def test_two_dimensions_changed_is_rejected(self) -> None:
        pair_type, reason = classify_pair(self.revenue_2018, self.ebitda_2019)
        self.assertIsNone(pair_type)
        self.assertEqual(reason, "rejected_more_than_one_dimension_changed")

    def test_self_pair_rejected(self) -> None:
        pair_type, reason = classify_pair(self.revenue_2019, self.revenue_2019)
        self.assertIsNone(pair_type)
        self.assertEqual(reason, "rejected_self_pair")

    def test_unit_pair_same_fact_different_scale(self) -> None:
        grid = [
            ["", "2019", "2018"],
            ["Revenue", "1,234", "1,100"],
            ["Revenue", "999", "888"],
        ]
        doc = {"table": {"uid": "table-2", "table": grid}}
        qa = extract(doc, grid, question("u1", "What was Revenue in 2019 (thousands)?", "1,234", scale="thousand"), 0)
        qb = extract(doc, grid, question("u2", "What was Revenue in 2019 (millions)?", "999", scale="million"), 1)
        pair_type, rule = classify_pair(qa, qb)
        self.assertEqual(pair_type, "unit")
        self.assertEqual(rule, "same_fact_different_scale_annotation")

    def test_strip_basis_qualifiers(self) -> None:
        base, had = strip_basis_qualifiers(normalize_metric_text("Adjusted EBITDA"))
        self.assertEqual(base, "ebitda")
        self.assertTrue(had)
        base2, had2 = strip_basis_qualifiers(normalize_metric_text("EBITDA"))
        self.assertEqual(base2, "ebitda")
        self.assertFalse(had2)

    def test_confusable_metric_flag(self) -> None:
        self.assertTrue(confusable_metric_flag("Revenue", "Sales"))
        self.assertFalse(confusable_metric_flag("Revenue", "EBITDA"))

    def test_build_pair_record_shape(self) -> None:
        record = build_pair_record(self.revenue_2019, self.revenue_2018, "period", "same_metric_same_table_different_period", 1)
        self.assertEqual(record["pair_id"], "tatqa_period_000001")
        self.assertEqual(record["split"], "test_gold")
        self.assertEqual(record["document_id"], "table-1")
        self.assertIsNone(record["page_id"])
        self.assertEqual(record["metric_a"], "Revenue")
        self.assertEqual(record["period_a"], "2019")
        self.assertEqual(record["period_b"], "2018")
        self.assertEqual(record["changed_dimension"], "period")
        self.assertEqual(record["construction_rule"], "same_metric_same_table_different_period")
        self.assertIn("cell_id", record["evidence_a"])

    def test_validate_natural_pairs_passes_on_clean_set(self) -> None:
        record = build_pair_record(self.revenue_2019, self.revenue_2018, "period", "same_metric_same_table_different_period", 1)
        report = validate_natural_pairs([record])
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["total_pairs"], 1)

    def test_validate_natural_pairs_raises_on_duplicate_pair_id(self) -> None:
        record = build_pair_record(self.revenue_2019, self.revenue_2018, "period", "same_metric_same_table_different_period", 1)
        record_b = build_pair_record(self.ebitda_2019, self.adj_ebitda_2019, "basis", "same_base_metric_qualifier_differs", 1)
        record_b = dict(record_b)
        record_b["pair_id"] = record["pair_id"]
        with self.assertRaises(ValueError):
            validate_natural_pairs([record, record_b])

    def test_validate_natural_pairs_raises_on_self_pair(self) -> None:
        record = build_pair_record(self.revenue_2019, self.revenue_2018, "period", "same_metric_same_table_different_period", 1)
        record = dict(record)
        record["question_b_id"] = record["question_a_id"]
        with self.assertRaises(ValueError):
            validate_natural_pairs([record])


class TemporalAndUnitRuleTests(unittest.TestCase):
    def test_temporal_remainder_same_year_different_scenario(self) -> None:
        self.assertEqual(temporal_remainder("2019 actual"), "actual")
        self.assertEqual(temporal_remainder("2019 threshold"), "threshold")
        self.assertNotEqual(temporal_remainder("2019 actual"), temporal_remainder("2019 threshold"))

    def test_temporal_remainder_full_dates_reduce_to_empty(self) -> None:
        self.assertEqual(temporal_remainder("September 27, 2019"), "")
        self.assertEqual(temporal_remainder("September 28, 2018"), "")
        self.assertEqual(temporal_remainder("May 31, 2019"), temporal_remainder("November 30, 2018"))

    def test_temporal_remainder_bare_years(self) -> None:
        self.assertEqual(temporal_remainder("2019"), "")
        self.assertEqual(temporal_remainder("2018"), "")

    def test_period_sort_key_orders_by_year(self) -> None:
        periods = ["2019", "2016", "2018", "2017"]
        ordered = sorted(periods, key=period_sort_key)
        self.assertEqual(ordered, ["2016", "2017", "2018", "2019"])

    def test_unit_class_distinguishes_per_share_and_percent(self) -> None:
        self.assertEqual(unit_class("0.55", "Diluted net income (loss) per share", None), "per_share")
        self.assertEqual(unit_class("12.7%", "Percentage of revenue", None), "percent")
        self.assertEqual(unit_class("6,577", "Research and development", None), "unscaled")

    def test_is_bare_total_row_vs_contextualized_total(self) -> None:
        self.assertTrue(is_bare_total_row("Total"))
        self.assertTrue(is_bare_total_row("Totals"))
        self.assertFalse(is_bare_total_row("Total current assets"))
        self.assertFalse(is_bare_total_row(None))

    def test_is_degenerate_metric_label(self) -> None:
        self.assertTrue(is_degenerate_metric_label("—"))
        self.assertTrue(is_degenerate_metric_label(""))
        self.assertTrue(is_degenerate_metric_label(None))
        self.assertFalse(is_degenerate_metric_label("Revenue"))

    def test_is_numeric_like_metric_label(self) -> None:
        # Real malformed-row-header examples found in the v2 candidate pool.
        self.assertTrue(is_numeric_like_metric_label("$ (50.5)"))
        self.assertTrue(is_numeric_like_metric_label("(6 )"))
        self.assertTrue(is_numeric_like_metric_label("(3 )"))
        self.assertTrue(is_numeric_like_metric_label("$(39,460)"))
        self.assertTrue(is_numeric_like_metric_label("(2)%"))
        self.assertTrue(is_numeric_like_metric_label("1,234.56"))
        self.assertTrue(is_numeric_like_metric_label("-"))
        self.assertFalse(is_numeric_like_metric_label(None))
        self.assertFalse(is_numeric_like_metric_label(""))
        self.assertFalse(is_numeric_like_metric_label("Revenue"))
        self.assertFalse(is_numeric_like_metric_label("Total current assets"))
        self.assertFalse(is_numeric_like_metric_label("Level 1"))
        self.assertFalse(is_numeric_like_metric_label("Note 2(a)"))

    def test_lexical_jaccard(self) -> None:
        self.assertGreaterEqual(
            lexical_jaccard("weighted average number of shares", "weighted average number of shares outstanding"),
            0.34,
        )
        self.assertEqual(lexical_jaccard("revenue", "ebitda"), 0.0)


def multi_period_multi_metric_grid():
    return [
        ["", "Years Ended December 31,", "", "", ""],
        ["", "2019", "2018", "2017", "2016"],
        ["Revenue", "1,000", "900", "800", "700"],
        ["EBITDA", "300", "250", "200", "150"],
        ["Adjusted EBITDA", "350", "300", "250", "200"],
        ["Net income per share", "2.5", "2.1", "1.8", "1.5"],
        ["Net income", "500", "420", "380", "300"],
        ["Total", "1,500", "1,300", "1,100", "900"],
        ["Total current assets", "2,000", "1,800", "1,600", "1,400"],
    ]


class TableDerivedFactsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = multi_period_multi_metric_grid()
        self.doc = {"table": {"uid": "table-derived-1", "table": self.grid}}
        self.cells = flatten_grid(self.grid)
        self.facts = extract_table_facts(self.doc, self.cells, "test_gold")

    def test_bare_total_excluded_contextualized_total_kept(self) -> None:
        metrics = {f.metric_raw for f in self.facts}
        self.assertNotIn("Total", metrics)
        self.assertIn("Total current assets", metrics)

    def test_numeric_like_row_header_excluded(self) -> None:
        # A malformed table section where the nearest row header above a
        # value cell is itself another value (real pattern found in the v2
        # candidate pool, e.g. metric_raw == "$ (50.5)").
        grid = [
            ["", "2017", "2018"],
            ["$ (50.5)", "65.3", "15.2"],
            ["Revenue", "1,000", "1,100"],
        ]
        doc = {"table": {"uid": "table-malformed", "table": grid}}
        cells = flatten_grid(grid)
        stats = Counter()
        facts = extract_table_facts(doc, cells, "test_gold", stats=stats)
        metrics = {f.metric_raw for f in facts}
        self.assertNotIn("$ (50.5)", metrics)
        self.assertIn("Revenue", metrics)
        self.assertEqual(stats["fact_rejected_numeric_metric_label"], 2)

    def test_period_pairs_prefer_adjacent_then_fill_with_nonadjacent_up_to_cap(self) -> None:
        pairs = build_period_pairs_for_table(self.facts)
        revenue_pairs = [(a, b, rule) for a, b, rule in pairs if a.metric_raw == "Revenue"]
        # 4 periods (2016-2019) -> C(4,2)=6 candidates, capped at
        # METRIC_GROUP_PERIOD_PAIR_CAP=4: all 3 adjacent (gap=1) pairs kept
        # first, then the smallest-gap remaining candidate (gap=2) fills the
        # last slot -- deterministic, not every possible pair.
        self.assertEqual(len(revenue_pairs), 4)
        seen_period_pairs = {(a.period_raw, b.period_raw) for a, b, _ in revenue_pairs}
        self.assertEqual(seen_period_pairs, {("2016", "2017"), ("2017", "2018"), ("2018", "2019"), ("2016", "2018")})
        adjacent_rules = {rule for a, b, rule in revenue_pairs if (a.period_raw, b.period_raw) in {("2016", "2017"), ("2017", "2018"), ("2018", "2019")}}
        self.assertEqual(adjacent_rules, {"same_metric_adjacent_period_different_value"})
        nonadjacent_rule = next(rule for a, b, rule in revenue_pairs if (a.period_raw, b.period_raw) == ("2016", "2018"))
        self.assertEqual(nonadjacent_rule, "same_metric_nonadjacent_period_different_value")

    def test_period_pairs_deterministic_across_repeated_calls(self) -> None:
        first = build_period_pairs_for_table(self.facts)
        second = build_period_pairs_for_table(self.facts)
        key = lambda items: [(a.period_raw, b.period_raw, a.metric_raw, rule) for a, b, rule in items]
        self.assertEqual(key(first), key(second))

    def test_metric_candidates_exclude_per_share_vs_aggregate(self) -> None:
        candidates = build_metric_candidates_for_table(self.facts)
        pairs_metrics = {frozenset({a.metric_raw, b.metric_raw}) for a, b, _, _ in candidates}
        self.assertNotIn(frozenset({"Net income per share", "Net income"}), pairs_metrics)

    def test_metric_candidates_include_basis_qualifier_match(self) -> None:
        candidates = build_metric_candidates_for_table(self.facts)
        matches = [(a, b, via) for a, b, via, _ in candidates if {a.metric_raw, b.metric_raw} == {"EBITDA", "Adjusted EBITDA"}]
        # One match per period (4 periods in the fixture grid).
        self.assertEqual(len(matches), 4)
        self.assertTrue(all(via == "basis_qualifier" for _, _, via in matches))

    def test_metric_candidate_record_carries_inclusion_basis(self) -> None:
        candidates = build_metric_candidates_for_table(self.facts)
        fact_a, fact_b, via, score = candidates[0]
        record = build_derived_pair_record(
            fact_a, fact_b, "metric", via, 1, "metric_candidate", "tatqa_metric_candidate",
            extra={"jaccard_score": score, "matched_via": via, "inclusion_basis": "auto_threshold_v2"},
        )
        self.assertEqual(record["inclusion_basis"], "auto_threshold_v2")
        self.assertEqual(record["subset"], "metric_candidate")
        self.assertEqual(record["protocol_version"], PROTOCOL_VERSION)
        self.assertEqual(record["question_generation_method"], "deterministic_template")

    def test_metric_candidates_exclude_footnote_suffix_duplicates(self) -> None:
        grid = [
            ["", "2019"],
            ["Corporate items and intercompany eliminations", "(177)"],
            ["Corporate items and intercompany eliminations 2", "(204)"],
        ]
        doc = {"table": {"uid": "table-footnote", "table": grid}}
        cells = flatten_grid(grid)
        facts = extract_table_facts(doc, cells, "test_gold")
        candidates = build_metric_candidates_for_table(facts)
        pairs_metrics = {frozenset({a.metric_raw, b.metric_raw}) for a, b, _, _ in candidates}
        self.assertNotIn(
            frozenset(
                {
                    "Corporate items and intercompany eliminations",
                    "Corporate items and intercompany eliminations 2",
                }
            ),
            pairs_metrics,
        )

    def test_metric_relatedness_threshold_is_v2_value(self) -> None:
        self.assertEqual(METRIC_RELATEDNESS_JACCARD_THRESHOLD, 0.50)


class PeriodPairCapTests(unittest.TestCase):
    def test_metric_group_cap_binds_even_before_any_nonadjacent_pair(self) -> None:
        # 6 periods -> 5 adjacent (gap=1) pairs alone already exceed
        # METRIC_GROUP_PERIOD_PAIR_CAP=4: only the smallest-gap 4 survive,
        # none of them are skip-period pairs, and 2019 (the newest period)
        # is the one dropped since it can only form gap>=1 pairs after the
        # first 4 adjacent ones are already taken by 2014-2018.
        years = ["2014", "2015", "2016", "2017", "2018", "2019"]
        header = ["", *years]
        row = ["Revenue", *[str(1000 + i) for i in range(len(years))]]
        grid = [header, row]
        doc = {"table": {"uid": "table-cap-metric", "table": grid}}
        cells = flatten_grid(grid)
        facts = extract_table_facts(doc, cells, "test_gold")
        pairs = build_period_pairs_for_table(facts)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all(rule == "same_metric_adjacent_period_different_value" for _, _, rule in pairs))
        seen = {(a.period_raw, b.period_raw) for a, b, _ in pairs}
        self.assertEqual(seen, {("2014", "2015"), ("2015", "2016"), ("2016", "2017"), ("2017", "2018")})

    def test_table_cap_binds_across_many_single_pair_metrics(self) -> None:
        # 30 distinct metrics, each with exactly 2 periods (1 candidate pair
        # apiece, well under the per-metric cap) -> 30 table-wide candidates,
        # trimmed to TABLE_PERIOD_PAIR_CAP=24.
        header = ["", "2018", "2019"]
        rows = [header]
        for i in range(30):
            rows.append([f"Metric {i:02d}", str(1000 + i), str(2000 + i)])
        doc = {"table": {"uid": "table-cap-table", "table": rows}}
        cells = flatten_grid(rows)
        facts = extract_table_facts(doc, cells, "test_gold")
        pairs = build_period_pairs_for_table(facts)
        self.assertEqual(len(pairs), 24)

    def test_table_cap_is_deterministic(self) -> None:
        header = ["", "2018", "2019"]
        rows = [header] + [[f"Metric {i:02d}", str(1000 + i), str(2000 + i)] for i in range(30)]
        doc = {"table": {"uid": "table-cap-table", "table": rows}}
        cells = flatten_grid(rows)
        facts = extract_table_facts(doc, cells, "test_gold")
        first = {(a.metric_raw, a.period_raw, b.period_raw) for a, b, _ in build_period_pairs_for_table(facts)}
        second = {(a.metric_raw, a.period_raw, b.period_raw) for a, b, _ in build_period_pairs_for_table(facts)}
        self.assertEqual(first, second)


class SameYearScenarioColumnTests(unittest.TestCase):
    def test_period_pair_rejected_when_only_scenario_differs(self) -> None:
        grid = [["", "2019 actual", "2019 threshold"], ["Group ROCE (%)", "54.5%", "50.1%"]]
        doc = {"table": {"uid": "table-scenario", "table": grid}}
        cells = flatten_grid(grid)
        facts = extract_table_facts(doc, cells, "test_gold")
        pairs = build_period_pairs_for_table(facts)
        self.assertEqual(pairs, [])

    def test_period_pair_accepted_for_genuinely_different_years(self) -> None:
        grid = [["", "September 27, 2019", "September 28, 2018"], ["Sales", "100", "90"]]
        doc = {"table": {"uid": "table-dates", "table": grid}}
        cells = flatten_grid(grid)
        facts = extract_table_facts(doc, cells, "test_gold")
        pairs = build_period_pairs_for_table(facts)
        self.assertEqual(len(pairs), 1)


def _synthetic_period_pair_record(pair_id, table_id, metric_a, rule):
    return {
        "pair_id": pair_id,
        "table_id": table_id,
        "document_id": table_id,
        "metric_a": metric_a,
        "metric_b": metric_a,
        "construction_rule": rule,
    }


class SelectPrimaryPeriodPairsTests(unittest.TestCase):
    def setUp(self) -> None:
        # table-A: metric "Revenue" has 3 adjacent + 2 skip candidates (over
        # the per-group cap of 1+1); metric "EBITDA" has 1 adjacent only.
        # table-B: 6 metrics, 1 adjacent pair each (6 total, to exercise the
        # per-table cap). table-C: 1 metric, 1 skip pair only (no adjacent).
        self.records = [
            _synthetic_period_pair_record("p_a_rev_1", "table-A", "Revenue", ADJACENT_RULE),
            _synthetic_period_pair_record("p_a_rev_2", "table-A", "Revenue", ADJACENT_RULE),
            _synthetic_period_pair_record("p_a_rev_3", "table-A", "Revenue", ADJACENT_RULE),
            _synthetic_period_pair_record("p_a_rev_skip_1", "table-A", "Revenue", NONADJACENT_RULE),
            _synthetic_period_pair_record("p_a_rev_skip_2", "table-A", "Revenue", NONADJACENT_RULE),
            _synthetic_period_pair_record("p_a_ebitda_1", "table-A", "EBITDA", ADJACENT_RULE),
        ]
        for i in range(6):
            self.records.append(_synthetic_period_pair_record(f"p_b_m{i}_1", "table-B", f"Metric{i}", ADJACENT_RULE))
        self.records.append(_synthetic_period_pair_record("p_c_skip_1", "table-C", "OnlySkip", NONADJACENT_RULE))

    def test_per_group_cap_enforced(self) -> None:
        result = select_primary_period_pairs(self.records, seed=1, table_cap=100, target_adjacent=100, target_skip=100)
        selected = set(result["pair_ids"])
        # At most 1 of the 3 Revenue-adjacent candidates survives.
        self.assertLessEqual(len(selected & {"p_a_rev_1", "p_a_rev_2", "p_a_rev_3"}), 1)
        # At most 1 of the 2 Revenue-skip candidates survives.
        self.assertLessEqual(len(selected & {"p_a_rev_skip_1", "p_a_rev_skip_2"}), 1)

    def test_per_table_cap_enforced(self) -> None:
        # table-B contributes 6 group-capped candidates (1 per metric); cap at 3.
        result = select_primary_period_pairs(self.records, seed=2, table_cap=3, target_adjacent=100, target_skip=100)
        table_b_selected = [pid for pid in result["pair_ids"] if pid.startswith("p_b_")]
        self.assertLessEqual(len(table_b_selected), 3)

    def test_target_counts_respected_when_pool_smaller(self) -> None:
        # Only 1 skip candidate survives per-group capping for table-C/table-A
        # combined (2 groups with skip candidates -> at most 2 skip total),
        # target_skip=100 should just keep everything available, not error.
        result = select_primary_period_pairs(self.records, seed=3, table_cap=100, target_adjacent=100, target_skip=100)
        self.assertEqual(result["stats"]["final_skip"], result["stats"]["stage2_skip"])
        self.assertLessEqual(result["stats"]["final_skip"], 2)

    def test_target_counts_capped_when_pool_larger(self) -> None:
        result = select_primary_period_pairs(self.records, seed=4, table_cap=100, target_adjacent=2, target_skip=1)
        self.assertEqual(result["stats"]["final_adjacent"], 2)
        self.assertEqual(result["stats"]["final_skip"], 1)

    def test_deterministic_given_same_seed(self) -> None:
        first = select_primary_period_pairs(self.records, seed=42, table_cap=3, target_adjacent=5, target_skip=2)
        second = select_primary_period_pairs(self.records, seed=42, table_cap=3, target_adjacent=5, target_skip=2)
        self.assertEqual(first["pair_ids"], second["pair_ids"])

    def test_deterministic_regardless_of_input_order(self) -> None:
        shuffled = list(reversed(self.records))
        result_a = select_primary_period_pairs(self.records, seed=7, table_cap=3, target_adjacent=5, target_skip=2)
        result_b = select_primary_period_pairs(shuffled, seed=7, table_cap=3, target_adjacent=5, target_skip=2)
        self.assertEqual(result_a["pair_ids"], result_b["pair_ids"])

    def test_no_model_prediction_fields_read(self) -> None:
        # Records here carry only construction-time fields (pair_id, table_id,
        # document_id, metric_a/b, construction_rule) -- no answer/prediction
        # field exists at all, and the function still runs correctly,
        # confirming it never touches model output.
        result = select_primary_period_pairs(self.records, seed=1, table_cap=100, target_adjacent=100, target_skip=100)
        self.assertGreater(result["stats"]["final_total"], 0)


if __name__ == "__main__":
    unittest.main()
