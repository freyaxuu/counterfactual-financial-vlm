"""Reliability audit for `company_profile` annotations.

Pure functions over the project-owned dataclasses in
`financial_vlm.integrations.evolution_ai_datasets_adapter` — no private-package
dependency, so this module is fully unit-testable with synthetic fixtures.

Three checks, matching the audit's scope:

- completeness: how often each of the 26 `company_profile` grouped fields is
  actually populated;
- cross-representation consistency: for the fields that exist in both the
  grouped-field instance and the summary-table row for the same company
  (matched by normalized `company_name` within a document), do the two
  annotated values agree?
- table conflation: geometric evidence (row bounding-box gaps) that a
  `tables["company_profile"]` group actually contains rows from more than one
  visually distinct table on the page. Relies on `CompanyProfileTableRow.cell_bboxes`,
  which is itself built from an undocumented private-package attribute -- see
  the WARNING in `evolution_ai_datasets_adapter`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
import statistics
from typing import Any, Mapping, Sequence

from financial_vlm.evaluation.pilot_accuracy import (
    is_exact_correct,
    is_numeric_correct,
    normalize_numeric_answer,
    normalize_text_answer,
)
from financial_vlm.integrations.evolution_ai_datasets_adapter import (
    COMPANY_PROFILE_FIELD_DATA_TYPES,
    COMPANY_PROFILE_GROUPED_FIELDS,
    TABLE_TO_GROUPED_FIELD_MAP,
    CompanyProfileDocument,
    CompanyProfileInstance,
    CompanyProfileTableRow,
)

DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d/%m/%y",
    "%m/%d/%y",
    "%d-%m-%Y",
    "%d %B %Y",
    "%d %b %Y",
    "%B %d, %Y",
    "%b %d, %Y",
    "%b-%y",
    "%B %Y",
    "%b %Y",
    "%m/%Y",
    "%m/%y",
    "%Y",
)

ORDINAL_SUFFIX_RE = re.compile(r"(?<=\d)(st|nd|rd|th)\b", re.IGNORECASE)

# Formats in DATE_FORMATS that don't carry a day (datetime.strptime defaults
# it to 1) or don't carry a month either -- needed to tell "the two sides
# disagree" apart from "one side just recorded less precision than the
# other" (see `date_granularity_mismatch`).
_MONTH_PRECISION_FORMATS: frozenset[str] = frozenset({"%b-%y", "%B %Y", "%b %Y", "%m/%Y", "%m/%y"})
_YEAR_PRECISION_FORMATS: frozenset[str] = frozenset({"%Y"})
_PRECISION_RANK: dict[str, int] = {"year": 0, "month": 1, "day": 2}


def _date_precision(fmt: str) -> str:
    if fmt in _YEAR_PRECISION_FORMATS:
        return "year"
    if fmt in _MONTH_PRECISION_FORMATS:
        return "month"
    return "day"

# Monetary fields are sometimes recorded in different units across the two
# annotation surfaces (e.g. raw currency on the company-profile detail page
# vs. millions on the summary table). A ratio close to one of these factors
# is a unit-convention mismatch, not necessarily a disagreement about the
# underlying figure -- reported separately from "mismatch".
MONETARY_SCALE_FACTORS: tuple[Decimal, ...] = (Decimal(1_000), Decimal(1_000_000), Decimal(1_000_000_000))
MONETARY_SCALE_RELATIVE_TOLERANCE = Decimal("0.02")


def normalize_company_name(value: str) -> str:
    return normalize_text_answer(value).casefold()


def parse_date_loose_with_precision(value: str) -> tuple[date, str] | None:
    """Like `parse_date_loose`, but also reports whether the matched format
    carries day precision, only month precision (day defaults to 1), or
    only year precision (month and day default to 1). Needed to distinguish
    a genuine date disagreement from one side simply recording less
    precision than the other -- see `date_granularity_mismatch`.
    """

    text = ORDINAL_SUFFIX_RE.sub("", value.strip())
    text = re.sub(r"\s+", " ", text).strip()
    for fmt in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, fmt).date()
        except ValueError:
            continue
        return parsed, _date_precision(fmt)
    return None


def parse_date_loose(value: str) -> date | None:
    result = parse_date_loose_with_precision(value)
    return result[0] if result else None


def date_granularity_mismatch(grouped_value: str, table_value: str) -> bool:
    """True if the two dates disagree only because one side was recorded at
    coarser precision than the other (e.g. "October 31, 2011" vs "Oct-11" --
    confirmed against a real the private dataset case, 2026-08-12) -- not a genuine
    disagreement about the underlying date.
    """

    grouped = parse_date_loose_with_precision(grouped_value)
    table = parse_date_loose_with_precision(table_value)
    if grouped is None or table is None:
        return False
    grouped_date, grouped_precision = grouped
    table_date, table_precision = table
    if grouped_date == table_date:
        return False
    if grouped_precision == table_precision:
        return False  # same precision, genuinely different dates
    if _PRECISION_RANK[grouped_precision] > _PRECISION_RANK[table_precision]:
        finer_date, coarser_date, coarser_precision = grouped_date, table_date, table_precision
    else:
        finer_date, coarser_date, coarser_precision = table_date, grouped_date, grouped_precision
    if coarser_precision == "year":
        return finer_date.year == coarser_date.year
    return finer_date.year == coarser_date.year and finer_date.month == coarser_date.month


def _is_close(a: Decimal, b: Decimal, relative_tolerance: Decimal = MONETARY_SCALE_RELATIVE_TOLERANCE) -> bool:
    if b == 0:
        return a == 0
    return abs(a - b) <= abs(b) * relative_tolerance


def monetary_scale_mismatch(grouped_value: str, table_value: str) -> bool:
    """True if the two values agree once one is scaled by 1e3/1e6/1e9."""

    grouped_normalized = normalize_numeric_answer(grouped_value)
    table_normalized = normalize_numeric_answer(table_value)
    if grouped_normalized is None or table_normalized is None:
        return False
    try:
        grouped_number = Decimal(grouped_normalized)
        table_number = Decimal(table_normalized)
    except InvalidOperation:
        return False
    if table_number == 0:
        return False
    ratio = grouped_number / table_number
    return any(
        _is_close(ratio, factor) or _is_close(ratio, Decimal(1) / factor) for factor in MONETARY_SCALE_FACTORS
    )


def text_granularity_mismatch(grouped_value: str, table_value: str) -> bool:
    """True if the shorter value is a comma-separated component of the longer one.

    Confirmed against real `country/City` mismatches in the private dataset: the
    grouped-field side records "City, Country" while the table side records
    only "Country" -- a format/granularity difference between the two
    annotation surfaces, not a disagreement about the underlying fact.
    """

    shorter, longer = sorted((grouped_value.strip(), table_value.strip()), key=len)
    shorter_cf = shorter.casefold()
    if shorter_cf == longer.strip().casefold():
        return False
    parts = [part.strip().casefold() for part in longer.split(",")]
    return shorter_cf in parts


def values_match(field_name: str, grouped_value: str, table_value: str) -> str:
    """Return "match", "mismatch", "scale_mismatch", "granularity_mismatch", or
    "unparseable" (dates only).

    "scale_mismatch" applies only to monetary fields where the figures agree
    up to a 1e3/1e6/1e9 unit-convention difference. "granularity_mismatch"
    covers two distinct cases depending on `data_type`: text fields where
    one side is a comma-separated component of the other (see
    `text_granularity_mismatch`), and date fields where one side was simply
    recorded at coarser precision -- e.g. day vs. month-only (see
    `date_granularity_mismatch`). All of these are real data-quality issues
    (inconsistent conventions across annotation surfaces), but distinct from
    the two sides recording genuinely different values.
    """

    data_type = COMPANY_PROFILE_FIELD_DATA_TYPES.get(field_name, "text")
    if data_type in ("monetary", "numerical"):
        if is_numeric_correct(grouped_value, table_value):
            return "match"
        if data_type == "monetary" and monetary_scale_mismatch(grouped_value, table_value):
            return "scale_mismatch"
        return "mismatch"
    if data_type == "date":
        grouped_date = parse_date_loose(grouped_value)
        table_date = parse_date_loose(table_value)
        if grouped_date is None or table_date is None:
            return "unparseable"
        if grouped_date == table_date:
            return "match"
        if date_granularity_mismatch(grouped_value, table_value):
            return "granularity_mismatch"
        return "mismatch"
    if is_exact_correct(grouped_value, table_value):
        return "match"
    if text_granularity_mismatch(grouped_value, table_value):
        return "granularity_mismatch"
    return "mismatch"


@dataclass(frozen=True)
class MatchedCompany:
    document_id: str
    company_name_normalized: str
    instance: CompanyProfileInstance
    table_row: CompanyProfileTableRow


@dataclass(frozen=True)
class FieldComparison:
    document_id: str
    company_name_normalized: str
    grouped_field: str
    table_field: str
    grouped_value: str
    table_value: str
    status: str  # "match" | "mismatch" | "unparseable"


def match_instances_to_table_rows(document: CompanyProfileDocument) -> tuple[list[MatchedCompany], int, int]:
    """Pair grouped-field instances to table rows by normalized company_name.

    Returns (matched pairs, unmatched instance count, unmatched table row count).
    Ambiguous ties (more than one instance or row sharing a normalized name)
    are paired in encounter order; only the first of each is matched, the rest
    count as unmatched.
    """

    instances_by_name: dict[str, list[CompanyProfileInstance]] = defaultdict(list)
    for instance in document.instances:
        name = instance.field_values.get("company_name")
        if name:
            instances_by_name[normalize_company_name(name)].append(instance)

    rows_by_name: dict[str, list[CompanyProfileTableRow]] = defaultdict(list)
    for row in document.table_rows:
        name = row.cell_values.get("company_name")
        if name:
            rows_by_name[normalize_company_name(name)].append(row)

    matched: list[MatchedCompany] = []
    unmatched_instances = 0
    unmatched_rows = 0

    all_names = set(instances_by_name) | set(rows_by_name)
    for name in all_names:
        instances = instances_by_name.get(name, [])
        rows = rows_by_name.get(name, [])
        if instances and rows:
            matched.append(
                MatchedCompany(
                    document_id=document.document_id,
                    company_name_normalized=name,
                    instance=instances[0],
                    table_row=rows[0],
                )
            )
            unmatched_instances += len(instances) - 1
            unmatched_rows += len(rows) - 1
        else:
            unmatched_instances += len(instances)
            unmatched_rows += len(rows)

    return matched, unmatched_instances, unmatched_rows


def compare_matched_company(pair: MatchedCompany) -> list[FieldComparison]:
    comparisons: list[FieldComparison] = []
    for table_field, grouped_field in TABLE_TO_GROUPED_FIELD_MAP.items():
        grouped_value = pair.instance.field_values.get(grouped_field)
        table_value = pair.table_row.cell_values.get(table_field)
        if grouped_value is None or table_value is None:
            continue
        status = values_match(grouped_field, grouped_value, table_value)
        comparisons.append(
            FieldComparison(
                document_id=pair.document_id,
                company_name_normalized=pair.company_name_normalized,
                grouped_field=grouped_field,
                table_field=table_field,
                grouped_value=grouped_value,
                table_value=table_value,
                status=status,
            )
        )
    return comparisons


def compute_field_completeness(
    documents: Sequence[CompanyProfileDocument],
    fields: Sequence[str] = COMPANY_PROFILE_GROUPED_FIELDS,
) -> dict[str, Any]:
    total_instances = sum(len(doc.instances) for doc in documents)
    present_counts: Counter[str] = Counter()
    for doc in documents:
        for instance in doc.instances:
            for field in fields:
                if field in instance.field_values:
                    present_counts[field] += 1

    per_field = {
        field: {
            "present": present_counts.get(field, 0),
            "total": total_instances,
            "completeness_rate": (present_counts.get(field, 0) / total_instances) if total_instances else 0.0,
        }
        for field in fields
    }

    overall_present = sum(present_counts.get(field, 0) for field in fields)
    overall_total = total_instances * len(fields)
    return {
        "total_instances": total_instances,
        "per_field": per_field,
        "overall_completeness_rate": (overall_present / overall_total) if overall_total else 0.0,
    }


@dataclass(frozen=True)
class TableConflationCandidate:
    """A `company_profile` table (one page) whose rows show a large vertical
    gap relative to the other row-to-row gaps on that page -- consistent with
    two visually separate tables having been annotated as one table group."""

    document_id: str
    page_id: str
    row_count: int
    split_after_row_rank: int  # 1-indexed position (top to bottom) of the gap
    rows_above_split: int
    rows_below_split: int
    max_gap: float
    median_other_gap: float
    gap_ratio: float


def _row_vertical_center(bboxes: Mapping[str, tuple[int, int, int, int]]) -> float | None:
    centers = [(top + bottom) / 2 for top, _left, bottom, _right in bboxes.values()]
    if not centers:
        return None
    return sum(centers) / len(centers)


def detect_table_conflation_candidates(
    table_rows: Sequence[CompanyProfileTableRow],
    min_gap_ratio: float = 2.5,
) -> list[TableConflationCandidate]:
    """Flag (document, page) tables with an outlier vertical gap between rows.

    Rows without any bounding-box data are skipped entirely (the private
    package doesn't document `textblock`, so coverage may be partial -- see
    the adapter module's WARNING). Requires at least 3 geo-located rows on a
    page to compute a meaningful "other gaps" median; pages with fewer are
    skipped rather than flagged, since there's no baseline to compare against.
    """

    rows_by_page: dict[tuple[str, str], list[CompanyProfileTableRow]] = defaultdict(list)
    for row in table_rows:
        rows_by_page[(row.document_id, row.page_id)].append(row)

    candidates: list[TableConflationCandidate] = []
    for (document_id, page_id), rows in rows_by_page.items():
        centered = [(row, _row_vertical_center(row.cell_bboxes)) for row in rows]
        centered = [(row, center) for row, center in centered if center is not None]
        if len(centered) < 3:
            continue
        centered.sort(key=lambda item: item[1])
        centers = [center for _row, center in centered]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]

        max_gap = max(gaps)
        max_gap_index = gaps.index(max_gap)
        other_gaps = gaps[:max_gap_index] + gaps[max_gap_index + 1 :]
        if not other_gaps:
            continue
        median_other = statistics.median(other_gaps)
        if median_other <= 0:
            continue
        gap_ratio = max_gap / median_other
        if gap_ratio < min_gap_ratio:
            continue

        candidates.append(
            TableConflationCandidate(
                document_id=document_id,
                page_id=page_id,
                row_count=len(centered),
                split_after_row_rank=max_gap_index + 1,
                rows_above_split=max_gap_index + 1,
                rows_below_split=len(centered) - (max_gap_index + 1),
                max_gap=max_gap,
                median_other_gap=median_other,
                gap_ratio=gap_ratio,
            )
        )
    return candidates


def summarize_table_conflation(
    documents: Sequence[CompanyProfileDocument],
    min_gap_ratio: float = 2.5,
) -> dict[str, Any]:
    all_rows = [row for doc in documents for row in doc.table_rows]
    geo_rows = [row for row in all_rows if row.cell_bboxes]
    rows_per_page: Counter[tuple[str, str]] = Counter((row.document_id, row.page_id) for row in geo_rows)
    pages_evaluable = sum(1 for count in rows_per_page.values() if count >= 3)

    candidates = detect_table_conflation_candidates(all_rows, min_gap_ratio=min_gap_ratio)

    return {
        "total_table_rows": len(all_rows),
        "table_rows_with_bbox_data": len(geo_rows),
        "pages_evaluable": pages_evaluable,
        "pages_flagged": len(candidates),
        "min_gap_ratio_threshold": min_gap_ratio,
        "candidates": [
            {
                "document_id": c.document_id,
                "page_id": c.page_id,
                "row_count": c.row_count,
                "split_after_row_rank": c.split_after_row_rank,
                "rows_above_split": c.rows_above_split,
                "rows_below_split": c.rows_below_split,
                "gap_ratio": c.gap_ratio,
            }
            for c in sorted(candidates, key=lambda c: c.gap_ratio, reverse=True)
        ],
    }


def summarize_company_profile_audit(documents: Sequence[CompanyProfileDocument]) -> dict[str, Any]:
    completeness = compute_field_completeness(documents)

    all_comparisons: list[FieldComparison] = []
    total_unmatched_instances = 0
    total_unmatched_rows = 0
    total_matched = 0
    table_only_field_counts: Counter[str] = Counter()

    for doc in documents:
        matched, unmatched_instances, unmatched_rows = match_instances_to_table_rows(doc)
        total_matched += len(matched)
        total_unmatched_instances += unmatched_instances
        total_unmatched_rows += unmatched_rows
        for pair in matched:
            all_comparisons.extend(compare_matched_company(pair))
        for row in doc.table_rows:
            for field_name in row.cell_values:
                if field_name not in TABLE_TO_GROUPED_FIELD_MAP:
                    table_only_field_counts[field_name] += 1

    per_field_consistency: dict[str, dict[str, Any]] = {}
    status_counts_by_field: dict[str, Counter[str]] = defaultdict(Counter)
    for comparison in all_comparisons:
        status_counts_by_field[comparison.grouped_field][comparison.status] += 1

    for grouped_field in sorted({c.grouped_field for c in all_comparisons}):
        counts = status_counts_by_field[grouped_field]
        match = counts.get("match", 0)
        mismatch = counts.get("mismatch", 0)
        scale_mismatch = counts.get("scale_mismatch", 0)
        granularity_mismatch = counts.get("granularity_mismatch", 0)
        unparseable = counts.get("unparseable", 0)
        comparable = match + mismatch
        per_field_consistency[grouped_field] = {
            "match": match,
            "mismatch": mismatch,
            "scale_mismatch": scale_mismatch,
            "granularity_mismatch": granularity_mismatch,
            "unparseable": unparseable,
            "compared_pairs": comparable,
            "mismatch_rate": (mismatch / comparable) if comparable else None,
            "scale_mismatch_rate": (
                (scale_mismatch / (comparable + scale_mismatch)) if (comparable + scale_mismatch) else None
            ),
            "granularity_mismatch_rate": (
                (granularity_mismatch / (comparable + granularity_mismatch))
                if (comparable + granularity_mismatch)
                else None
            ),
        }

    total_match = sum(v["match"] for v in per_field_consistency.values())
    total_mismatch = sum(v["mismatch"] for v in per_field_consistency.values())
    total_scale_mismatch = sum(v["scale_mismatch"] for v in per_field_consistency.values())
    total_granularity_mismatch = sum(v["granularity_mismatch"] for v in per_field_consistency.values())
    total_comparable = total_match + total_mismatch

    return {
        "documents": len(documents),
        "completeness": completeness,
        "consistency": {
            "matched_companies": total_matched,
            "unmatched_instances": total_unmatched_instances,
            "unmatched_table_rows": total_unmatched_rows,
            "per_field": per_field_consistency,
            "overall_mismatch_rate": (total_mismatch / total_comparable) if total_comparable else None,
            "overall_scale_mismatch_count": total_scale_mismatch,
            "overall_granularity_mismatch_count": total_granularity_mismatch,
        },
        "table_only_field_counts": dict(sorted(table_only_field_counts.items())),
        "table_conflation": summarize_table_conflation(documents),
    }
