from __future__ import annotations

import unittest

from financial_vlm.evaluation.company_profile_audit import (
    compare_matched_company,
    compute_field_completeness,
    detect_table_conflation_candidates,
    match_instances_to_table_rows,
    normalize_company_name,
    parse_date_loose,
    summarize_company_profile_audit,
    summarize_table_conflation,
    values_match,
)
from financial_vlm.integrations.evolution_ai_datasets_adapter import (
    CompanyProfileDocument,
    CompanyProfileInstance,
    CompanyProfileTableRow,
)


def instance(
    document_id: str,
    instance_index: int,
    field_values: dict[str, str],
    page_id: str = "page-0",
) -> CompanyProfileInstance:
    return CompanyProfileInstance(
        document_id=document_id,
        page_id=page_id,
        instance_index=instance_index,
        field_values=field_values,
    )


def table_row(
    document_id: str,
    row_index: int,
    cell_values: dict[str, str],
    page_id: str = "page-1",
    cell_bboxes: dict[str, tuple[int, int, int, int]] | None = None,
) -> CompanyProfileTableRow:
    return CompanyProfileTableRow(
        document_id=document_id,
        page_id=page_id,
        row_index=row_index,
        cell_values=cell_values,
        cell_bboxes=cell_bboxes or {},
    )


class NormalizeCompanyNameTest(unittest.TestCase):
    def test_casefold_and_whitespace(self) -> None:
        self.assertEqual(normalize_company_name(" Acme  Corp "), normalize_company_name("acme corp"))


class ParseDateLooseTest(unittest.TestCase):
    def test_parses_common_formats(self) -> None:
        self.assertEqual(parse_date_loose("2021-03-15").isoformat(), "2021-03-15")
        self.assertEqual(parse_date_loose("15/03/2021").isoformat(), "2021-03-15")

    def test_unparseable_returns_none(self) -> None:
        self.assertIsNone(parse_date_loose("Q1 FY21"))

    def test_parses_ordinal_day_and_two_digit_year(self) -> None:
        self.assertEqual(parse_date_loose("15th March 2021").isoformat(), "2021-03-15")
        self.assertEqual(parse_date_loose("03/21").isoformat(), "2021-03-01")

    def test_parses_abbreviated_month_dash_two_digit_year(self) -> None:
        self.assertEqual(parse_date_loose("Mar-21").isoformat(), "2021-03-01")

    def test_parses_day_month_two_digit_year_slash_format(self) -> None:
        self.assertEqual(parse_date_loose("15/03/21").isoformat(), "2021-03-15")


class ValuesMatchTest(unittest.TestCase):
    def test_monetary_match_despite_formatting(self) -> None:
        self.assertEqual(values_match("total_value", "$1,200,000", "1200000"), "match")

    def test_monetary_mismatch(self) -> None:
        self.assertEqual(values_match("total_value", "1200000", "1300000"), "mismatch")

    def test_text_match_case_insensitive(self) -> None:
        self.assertEqual(values_match("industry", "Healthcare", "healthcare"), "match")

    def test_date_match_across_formats(self) -> None:
        self.assertEqual(values_match("entry_date", "2021-03-15", "15/03/2021"), "match")

    def test_date_unparseable(self) -> None:
        self.assertEqual(values_match("entry_date", "Q1 FY21", "2021-03-15"), "unparseable")

    def test_monetary_scale_mismatch_detected_separately_from_mismatch(self) -> None:
        # grouped in raw currency, table in millions -- same figure, different unit convention.
        self.assertEqual(values_match("total_value", "45000000", "45"), "scale_mismatch")

    def test_numerical_field_does_not_get_scale_correction(self) -> None:
        # gross_IRR is "numerical", not "monetary" -- a 1000x ratio there is a real mismatch.
        self.assertEqual(values_match("gross_IRR", "4500", "4.5"), "mismatch")

    def test_text_granularity_mismatch_city_country_vs_country_only(self) -> None:
        self.assertEqual(values_match("country/City", "London, United Kingdom", "United Kingdom"), "granularity_mismatch")

    def test_date_granularity_mismatch_day_precision_vs_month_precision(self) -> None:
        # confirmed real the private dataset case, 2026-08-12: grouped side records the
        # exact day, table side only month-year -- not a genuine disagreement.
        self.assertEqual(values_match("entry_date", "October 31, 2011", "Oct-11"), "granularity_mismatch")

    def test_date_granularity_mismatch_month_precision_vs_year_precision(self) -> None:
        self.assertEqual(values_match("entry_date", "Mar-21", "2021"), "granularity_mismatch")

    def test_date_same_precision_genuinely_different_is_still_mismatch(self) -> None:
        self.assertEqual(values_match("entry_date", "Oct-11", "Mar-11"), "mismatch")

    def test_date_different_year_at_month_precision_is_still_mismatch(self) -> None:
        self.assertEqual(values_match("entry_date", "October 31, 2011", "Oct-12"), "mismatch")

    def test_text_field_genuinely_different_values_still_mismatch(self) -> None:
        self.assertEqual(values_match("industry", "Healthcare", "Consumer Goods"), "mismatch")


class CompletenessTest(unittest.TestCase):
    def test_per_field_and_overall_rates(self) -> None:
        documents = [
            CompanyProfileDocument(
                document_id="doc-1",
                instances=(
                    instance("doc-1", 0, {"company_name": "Acme", "industry": "Healthcare"}),
                    instance("doc-1", 1, {"company_name": "Beta"}),
                ),
                table_rows=(),
            )
        ]
        result = compute_field_completeness(documents, fields=("company_name", "industry"))
        self.assertEqual(result["total_instances"], 2)
        self.assertEqual(result["per_field"]["company_name"]["present"], 2)
        self.assertEqual(result["per_field"]["company_name"]["completeness_rate"], 1.0)
        self.assertEqual(result["per_field"]["industry"]["present"], 1)
        self.assertEqual(result["per_field"]["industry"]["completeness_rate"], 0.5)
        self.assertEqual(result["overall_completeness_rate"], 0.75)

    def test_empty_documents(self) -> None:
        result = compute_field_completeness([], fields=("company_name",))
        self.assertEqual(result["total_instances"], 0)
        self.assertEqual(result["overall_completeness_rate"], 0.0)
        self.assertEqual(result["per_field"]["company_name"]["completeness_rate"], 0.0)


class MatchInstancesToTableRowsTest(unittest.TestCase):
    def test_matches_by_normalized_company_name(self) -> None:
        document = CompanyProfileDocument(
            document_id="doc-1",
            instances=(instance("doc-1", 0, {"company_name": "  Acme Corp "}),),
            table_rows=(table_row("doc-1", 5, {"company_name": "acme corp"}),),
        )
        matched, unmatched_instances, unmatched_rows = match_instances_to_table_rows(document)
        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched_instances, 0)
        self.assertEqual(unmatched_rows, 0)
        self.assertEqual(matched[0].company_name_normalized, normalize_company_name("Acme Corp"))

    def test_unmatched_when_names_differ(self) -> None:
        document = CompanyProfileDocument(
            document_id="doc-1",
            instances=(instance("doc-1", 0, {"company_name": "Acme Corp"}),),
            table_rows=(table_row("doc-1", 5, {"company_name": "Globex Inc"}),),
        )
        matched, unmatched_instances, unmatched_rows = match_instances_to_table_rows(document)
        self.assertEqual(len(matched), 0)
        self.assertEqual(unmatched_instances, 1)
        self.assertEqual(unmatched_rows, 1)

    def test_extra_duplicate_rows_count_as_unmatched(self) -> None:
        document = CompanyProfileDocument(
            document_id="doc-1",
            instances=(instance("doc-1", 0, {"company_name": "Acme Corp"}),),
            table_rows=(
                table_row("doc-1", 5, {"company_name": "Acme Corp"}),
                table_row("doc-1", 6, {"company_name": "Acme Corp"}),
            ),
        )
        matched, unmatched_instances, unmatched_rows = match_instances_to_table_rows(document)
        self.assertEqual(len(matched), 1)
        self.assertEqual(unmatched_instances, 0)
        self.assertEqual(unmatched_rows, 1)


class CompareMatchedCompanyTest(unittest.TestCase):
    def test_only_compares_fields_present_on_both_sides(self) -> None:
        document = CompanyProfileDocument(
            document_id="doc-1",
            instances=(
                instance(
                    "doc-1",
                    0,
                    {
                        "company_name": "Acme Corp",
                        "total_value": "1200000",
                        "description": "not annotated on the table side",
                    },
                ),
            ),
            table_rows=(
                table_row(
                    "doc-1",
                    5,
                    {"company_name": "Acme Corp", "total_value_table": "1300000"},
                ),
            ),
        )
        matched, _, _ = match_instances_to_table_rows(document)
        comparisons = compare_matched_company(matched[0])
        fields_compared = {c.grouped_field for c in comparisons}
        self.assertIn("total_value", fields_compared)
        self.assertNotIn("description", fields_compared)
        total_value_comparison = next(c for c in comparisons if c.grouped_field == "total_value")
        self.assertEqual(total_value_comparison.status, "mismatch")


class SummarizeCompanyProfileAuditTest(unittest.TestCase):
    def test_aggregates_completeness_and_consistency(self) -> None:
        documents = [
            CompanyProfileDocument(
                document_id="doc-1",
                instances=(
                    instance(
                        "doc-1",
                        0,
                        {"company_name": "Acme Corp", "industry": "Healthcare", "total_value": "1000"},
                    ),
                ),
                table_rows=(
                    table_row(
                        "doc-1",
                        5,
                        {
                            "company_name": "Acme Corp",
                            "industry": "Healthcare",
                            "total_value_table": "1000",
                            "LTM_EBITDA": "500",
                        },
                    ),
                ),
            )
        ]
        summary = summarize_company_profile_audit(documents)
        self.assertEqual(summary["documents"], 1)
        self.assertEqual(summary["consistency"]["matched_companies"], 1)
        self.assertEqual(summary["consistency"]["unmatched_instances"], 0)
        self.assertEqual(summary["consistency"]["unmatched_table_rows"], 0)
        self.assertEqual(summary["consistency"]["overall_mismatch_rate"], 0.0)
        self.assertEqual(summary["consistency"]["overall_scale_mismatch_count"], 0)
        self.assertEqual(summary["table_only_field_counts"], {"LTM_EBITDA": 1})
        self.assertGreater(summary["completeness"]["total_instances"], 0)

    def test_scale_mismatch_is_not_counted_as_a_plain_mismatch(self) -> None:
        documents = [
            CompanyProfileDocument(
                document_id="doc-1",
                instances=(
                    instance("doc-1", 0, {"company_name": "Acme Corp", "total_value": "45000000"}),
                ),
                table_rows=(
                    table_row("doc-1", 5, {"company_name": "Acme Corp", "total_value_table": "45"}),
                ),
            )
        ]
        summary = summarize_company_profile_audit(documents)
        # company_name itself matches cleanly, so it's the only field counted
        # in overall_mismatch_rate -- total_value's scale difference must not
        # inflate that rate.
        self.assertEqual(summary["consistency"]["overall_mismatch_rate"], 0.0)
        self.assertEqual(summary["consistency"]["overall_scale_mismatch_count"], 1)
        self.assertEqual(summary["consistency"]["per_field"]["total_value"]["scale_mismatch"], 1)
        self.assertEqual(summary["consistency"]["per_field"]["total_value"]["mismatch"], 0)
        self.assertIsNone(summary["consistency"]["per_field"]["total_value"]["mismatch_rate"])

    def test_no_documents_produces_null_mismatch_rate(self) -> None:
        summary = summarize_company_profile_audit([])
        self.assertEqual(summary["documents"], 0)
        self.assertIsNone(summary["consistency"]["overall_mismatch_rate"])


def bbox_row(document_id: str, row_index: int, y: int, page_id: str = "page-1") -> CompanyProfileTableRow:
    return table_row(
        document_id,
        row_index,
        {"company_name": f"row {row_index}"},
        page_id=page_id,
        cell_bboxes={"company_name": (y, 0, y + 10, 100)},
    )


class DetectTableConflationCandidatesTest(unittest.TestCase):
    def test_evenly_spaced_rows_are_not_flagged(self) -> None:
        rows = [bbox_row("doc-1", i, y) for i, y in enumerate([100, 200, 300, 400, 500])]
        candidates = detect_table_conflation_candidates(rows)
        self.assertEqual(candidates, [])

    def test_one_large_gap_among_even_rows_is_flagged(self) -> None:
        rows = [bbox_row("doc-1", i, y) for i, y in enumerate([100, 200, 300, 800, 900, 1000])]
        candidates = detect_table_conflation_candidates(rows)
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.document_id, "doc-1")
        self.assertEqual(candidate.page_id, "page-1")
        self.assertEqual(candidate.row_count, 6)
        self.assertEqual(candidate.rows_above_split, 3)
        self.assertEqual(candidate.rows_below_split, 3)
        self.assertGreaterEqual(candidate.gap_ratio, 2.5)

    def test_fewer_than_three_geolocated_rows_is_skipped(self) -> None:
        rows = [bbox_row("doc-1", i, y) for i, y in enumerate([100, 900])]
        candidates = detect_table_conflation_candidates(rows)
        self.assertEqual(candidates, [])

    def test_rows_without_bboxes_are_ignored(self) -> None:
        rows = [
            table_row("doc-1", 0, {"company_name": "a"}),
            table_row("doc-1", 1, {"company_name": "b"}),
            table_row("doc-1", 2, {"company_name": "c"}),
        ]
        candidates = detect_table_conflation_candidates(rows)
        self.assertEqual(candidates, [])

    def test_pages_are_evaluated_independently(self) -> None:
        rows = [bbox_row("doc-1", i, y, page_id="page-A") for i, y in enumerate([100, 200, 300])]
        rows += [bbox_row("doc-1", i, y, page_id="page-B") for i, y in enumerate([100, 900])]
        candidates = detect_table_conflation_candidates(rows)
        self.assertEqual(candidates, [])


class SummarizeTableConflationTest(unittest.TestCase):
    def test_aggregates_counts_and_sorts_candidates_by_gap_ratio(self) -> None:
        flagged_rows = [bbox_row("doc-1", i, y) for i, y in enumerate([100, 200, 300, 800, 900, 1000])]
        clean_rows = [bbox_row("doc-2", i, y, page_id="page-2") for i, y in enumerate([100, 200, 300])]
        no_bbox_row = table_row("doc-3", 0, {"company_name": "x"})
        documents = [
            CompanyProfileDocument(document_id="doc-1", instances=(), table_rows=tuple(flagged_rows)),
            CompanyProfileDocument(document_id="doc-2", instances=(), table_rows=tuple(clean_rows)),
            CompanyProfileDocument(document_id="doc-3", instances=(), table_rows=(no_bbox_row,)),
        ]
        summary = summarize_table_conflation(documents)
        self.assertEqual(summary["total_table_rows"], 6 + 3 + 1)
        self.assertEqual(summary["table_rows_with_bbox_data"], 9)
        self.assertEqual(summary["pages_evaluable"], 2)
        self.assertEqual(summary["pages_flagged"], 1)
        self.assertEqual(len(summary["candidates"]), 1)
        self.assertEqual(summary["candidates"][0]["document_id"], "doc-1")


if __name__ == "__main__":
    unittest.main()
