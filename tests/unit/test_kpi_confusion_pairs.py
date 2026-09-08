from __future__ import annotations

import unittest

from financial_vlm.evaluation.kpi_confusion_pairs import (
    KPIRecord,
    check_duplicate_period_key_value_agreement,
    classify_period_label,
    column_x_from_bboxes,
    count_metric_pair_co_occurrence,
    find_disagreeing_duplicate_instances,
    find_period_and_basis_pairs,
    group_by_company,
    normalize_month,
    normalize_year,
)


def rec(
    doc: str,
    idx: int,
    company: str,
    period: str | None = None,
    year: str | None = None,
    month: str | None = None,
    metrics: dict[str, str] | None = None,
    page: str = "p1",
    column_x: float | None = None,
) -> KPIRecord:
    return KPIRecord(
        document_id=doc,
        page_id=page,
        instance_index=idx,
        column_x=column_x,
        company_key=company,
        period=period,
        year=year,
        month=month,
        metric_values=metrics or {},
    )


class ClassifyPeriodLabelTest(unittest.TestCase):
    def test_scenario_basis(self) -> None:
        self.assertEqual(classify_period_label("A"), "scenario_basis")
        self.assertEqual(classify_period_label("Forecast"), "scenario_basis")

    def test_ambiguous_relative(self) -> None:
        self.assertEqual(classify_period_label("At Close"), "ambiguous_relative")

    def test_actual_temporal(self) -> None:
        self.assertEqual(classify_period_label("Q1"), "actual_temporal")

    def test_scenario_basis_is_case_insensitive(self) -> None:
        # confirmed real the private dataset bug, 2026-08-12: raw labels appear in
        # multiple cases ("BUDGET"/"Budget"/"budget"); the original
        # exact-string check let "BUDGET" fall through to "unknown_other",
        # allowing an Actual-vs-Budget pair through as a fake period pair.
        self.assertEqual(classify_period_label("BUDGET"), "scenario_basis")
        self.assertEqual(classify_period_label("budget"), "scenario_basis")
        self.assertEqual(classify_period_label("actual"), "scenario_basis")
        self.assertEqual(classify_period_label("a"), "scenario_basis")

    def test_ambiguous_relative_is_case_insensitive(self) -> None:
        self.assertEqual(classify_period_label("AT CLOSE"), "ambiguous_relative")

    def test_actual_temporal_is_case_insensitive(self) -> None:
        self.assertEqual(classify_period_label("q1"), "actual_temporal")

    def test_bare_b_and_compound_labels_containing_it_are_budget_scenario_basis(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: raw period field was
        # literally "Year-to-Date B" -- "B" follows the same single-letter
        # convention as A/F/E (Actual/Forecast/Estimate) already handled.
        self.assertEqual(classify_period_label("B"), "scenario_basis")
        self.assertEqual(classify_period_label("Year-to-Date B"), "scenario_basis")

    def test_gaap_and_related_basis_qualifiers_are_scenario_basis(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: "1Q GAAP"-style labels
        # slipped through as ordinary periods until GAAP/Reported/ProForma/
        # Landing/Target/Reforecast were added -- see report section 5C,
        # which already lists GAAP vs. non-GAAP as its own basis category.
        self.assertEqual(classify_period_label("1Q GAAP"), "scenario_basis")
        self.assertEqual(classify_period_label("12M Reported"), "scenario_basis")
        self.assertEqual(classify_period_label("5M ProForma"), "scenario_basis")
        self.assertEqual(classify_period_label("12M Reforecast"), "scenario_basis")

    def test_compound_labels_are_classified_by_contained_keyword(self) -> None:
        # confirmed real the private dataset bug, 2026-08-12: compound labels like
        # "12M Budget" and "PQ Actual" never exactly matched any set entry
        # and fell through to "unknown_other", treated as an ordinary,
        # comparable period even though they encode a basis/scenario signal.
        self.assertEqual(classify_period_label("12M Budget"), "scenario_basis")
        self.assertEqual(classify_period_label("PQ Actual"), "scenario_basis")
        self.assertEqual(classify_period_label("5M Actual"), "scenario_basis")

    def test_keyword_must_be_a_whole_word_not_a_substring_of_another_word(self) -> None:
        # "F" (Forecast/scenario_basis) must not match inside "FQ1" (Fiscal
        # Quarter 1, an ordinary temporal label with no space after "F").
        self.assertEqual(classify_period_label("FQ1"), "unknown_other")
        # "A" must not match inside "Adj." (Adjusted Closing).
        self.assertEqual(classify_period_label("Adj. Closing"), "unknown_other")

    def test_missing(self) -> None:
        self.assertEqual(classify_period_label(None), "missing")

    def test_unknown(self) -> None:
        self.assertEqual(classify_period_label("some novel label"), "unknown_other")


class NormalizeYearMonthTest(unittest.TestCase):
    def test_two_digit_year_assumed_2000s(self) -> None:
        self.assertEqual(normalize_year("23"), "2023")

    def test_four_digit_year_passthrough(self) -> None:
        self.assertEqual(normalize_year("2019"), "2019")

    def test_unparseable_year(self) -> None:
        self.assertIsNone(normalize_year("FY"))

    def test_month_name_variants_normalize_together(self) -> None:
        self.assertEqual(normalize_month("Sep"), normalize_month("9"))
        self.assertEqual(normalize_month("December"), normalize_month("12"))

    def test_unknown_month(self) -> None:
        self.assertIsNone(normalize_month("Q1"))


class GroupByCompanyTest(unittest.TestCase):
    def test_groups_by_document_and_company(self) -> None:
        records = [rec("doc1", 0, "acme"), rec("doc1", 1, "acme"), rec("doc1", 2, "globex")]
        groups = group_by_company(records)
        self.assertEqual(len(groups[("doc1", "acme")]), 2)
        self.assertEqual(len(groups[("doc1", "globex")]), 1)


class FindPeriodAndBasisPairsTest(unittest.TestCase):
    def test_different_years_produce_a_valid_period_pair(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="LTM", year="2021"),
            rec("doc1", 1, "acme", period="LTM", year="2022"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 1)
        self.assertEqual(result.groups_with_valid_period_pair, 1)
        self.assertEqual(len(result.basis_pairs), 0)

    def test_actual_vs_forecast_same_year_is_a_basis_pair_not_a_period_pair(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="A", year="2023"),
            rec("doc1", 1, "acme", period="F", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.basis_pairs), 1)
        self.assertEqual(len(result.valid_period_pairs), 0)

    def test_actual_vs_budget_different_case_is_still_a_basis_pair(self) -> None:
        # confirmed real the private dataset bug, 2026-08-12: "BUDGET" (caps) vs.
        # "Actual" -- case mismatch previously let this through as a fake
        # period pair instead of being caught as an actual-vs-forecast/
        # budget basis pair.
        records = [
            rec("doc1", 0, "acme", period="Actual", year="2023"),
            rec("doc1", 1, "acme", period="BUDGET", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        # "BUDGET" isn't in FORECAST_LABELS (only F/E/Forecast/Estimate), so
        # this isn't a basis pair either -- it must be excluded entirely,
        # not silently treated as a valid period pair.
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(len(result.basis_pairs), 0)
        self.assertEqual(result.excluded_ambiguous_count, 1)

    def test_actual_vs_forecast_different_case_is_a_basis_pair(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="actual", year="2023"),
            rec("doc1", 1, "acme", period="FORECAST", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.basis_pairs), 1)
        self.assertEqual(len(result.valid_period_pairs), 0)

    def test_ambiguous_relative_label_is_excluded_not_paired(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Prior Qtr", year="2023"),
            rec("doc1", 1, "acme", period="Curr Qtr", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(len(result.basis_pairs), 0)
        self.assertEqual(result.excluded_ambiguous_count, 1)

    def test_identical_normalized_period_counted_separately(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="23"),
            rec("doc1", 1, "acme", period="Q1", year="2023"),  # same after normalization
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(result.identical_period_key_count, 1)
        self.assertEqual(len(result.valid_period_pairs), 0)

    def test_single_instance_group_yields_no_pairs(self) -> None:
        records = [rec("doc1", 0, "acme", period="Q1", year="2023")]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(len(result.basis_pairs), 0)

    def test_unparseable_year_is_excluded_as_ambiguous(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="FY23"),
            rec("doc1", 1, "acme", period="Q2", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(result.excluded_ambiguous_count, 1)

    def test_identical_label_but_different_columns_is_a_valid_pair_not_a_duplicate(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: two columns on the same
        # page share one nominal period label -- that's two distinct facts,
        # not the same fact reported twice.
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=400.0),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 1)
        self.assertEqual(result.identical_period_key_count, 0)

    def test_identical_label_same_column_is_still_a_suspect_duplicate(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=105.0),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(result.identical_period_key_count, 1)

    def test_identical_label_without_column_position_stays_conservative(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023"),
            rec("doc1", 1, "acme", period="Q1", year="2023"),
        ]
        result = find_period_and_basis_pairs(group_by_company(records))
        self.assertEqual(len(result.valid_period_pairs), 0)
        self.assertEqual(result.identical_period_key_count, 1)


class CheckDuplicatePeriodKeyValueAgreementTest(unittest.TestCase):
    def test_agreeing_duplicates(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", metrics={"sales": "100"}),
        ]
        result = check_duplicate_period_key_value_agreement(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(result["fully_agree"], 1)
        self.assertEqual(result["disagree_on_at_least_one_metric"], 0)

    def test_disagreeing_duplicates(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", metrics={"sales": "200"}),
        ]
        result = check_duplicate_period_key_value_agreement(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(result["fully_agree"], 0)
        self.assertEqual(result["disagree_on_at_least_one_metric"], 1)
        self.assertEqual(result["disagreement_rate"], 1.0)

    def test_no_comparable_metrics_is_not_counted(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", metrics={}),
            rec("doc1", 1, "acme", period="Q1", year="2023", metrics={}),
        ]
        result = check_duplicate_period_key_value_agreement(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(result["duplicate_groups_checked"], 0)
        self.assertIsNone(result["disagreement_rate"])

    def test_same_label_different_columns_is_not_a_duplicate_at_all(self) -> None:
        # confirmed real the private dataset case, 2026-08-12 -- two distinct columns
        # sharing a label must not be compared as if they were one fact.
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0, metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=400.0, metrics={"sales": "999"}),
        ]
        result = check_duplicate_period_key_value_agreement(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(result["duplicate_groups_checked"], 0)

    def test_same_label_same_column_still_compared(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0, metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=105.0, metrics={"sales": "200"}),
        ]
        result = check_duplicate_period_key_value_agreement(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(result["duplicate_groups_checked"], 1)
        self.assertEqual(result["disagree_on_at_least_one_metric"], 1)


class FindDisagreeingDuplicateInstancesTest(unittest.TestCase):
    def test_flags_both_instances_of_a_disagreeing_same_column_duplicate(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0, metrics={"sales": "100"}, page="p1"),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=105.0, metrics={"sales": "200"}, page="p1"),
        ]
        flagged = find_disagreeing_duplicate_instances(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(flagged, {("doc1", "p1", 0), ("doc1", "p1", 1)})

    def test_does_not_flag_different_columns_sharing_a_label(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0, metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=400.0, metrics={"sales": "999"}),
        ]
        flagged = find_disagreeing_duplicate_instances(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(flagged, set())

    def test_does_not_flag_agreeing_duplicates(self) -> None:
        records = [
            rec("doc1", 0, "acme", period="Q1", year="2023", column_x=100.0, metrics={"sales": "100"}),
            rec("doc1", 1, "acme", period="Q1", year="2023", column_x=105.0, metrics={"sales": "100"}),
        ]
        flagged = find_disagreeing_duplicate_instances(group_by_company(records), metric_fields=("sales",))
        self.assertEqual(flagged, set())


class CountMetricPairCoOccurrenceTest(unittest.TestCase):
    def test_counts_instances_with_both_fields_populated(self) -> None:
        maps = [
            {"total_debt": "10", "net_debt": "8"},
            {"total_debt": "10"},
            {"net_debt": "8"},
            {},
        ]
        result = count_metric_pair_co_occurrence(maps, [("total_debt", "net_debt")])
        self.assertEqual(result[("total_debt", "net_debt")], 1)

    def test_missing_pair_counts_zero(self) -> None:
        result = count_metric_pair_co_occurrence([{"a": "1"}], [("x", "y")])
        self.assertEqual(result[("x", "y")], 0)


class ColumnXFromBboxesTest(unittest.TestCase):
    def test_averages_left_coordinate_of_metric_fields_only(self) -> None:
        bboxes = {
            "sales": (688, 100, 702, 146),
            "EBITDA": (720, 110, 734, 156),
            "company_name": (78, 0, 95, 900),  # row header, spans whole row -- excluded
        }
        result = column_x_from_bboxes(bboxes, metric_fields=("sales", "EBITDA"))
        self.assertEqual(result, 105.0)

    def test_no_metric_fields_present_returns_none(self) -> None:
        bboxes = {"company_name": (78, 0, 95, 900)}
        result = column_x_from_bboxes(bboxes, metric_fields=("sales", "EBITDA"))
        self.assertIsNone(result)

    def test_empty_bboxes_returns_none(self) -> None:
        self.assertIsNone(column_x_from_bboxes({}, metric_fields=("sales",)))


if __name__ == "__main__":
    unittest.main()
