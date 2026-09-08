"""Model-independent construction of period/basis confusion-pair candidates
from `grouped_fields.KPIs` instances, plus co-occurrence counting for
metric-pair candidates.

Pure functions over a small project-owned `KPIRecord`, not the adapter's
`GroupedFieldInstance` directly -- callers build one `KPIRecord` per
instance (see `scripts/diagnose_kpi_confusion_pairs.py`), which keeps this
module fully unit-testable with synthetic fixtures and independent of the
private package.

See `docs/company-benchmark-diagnosis-report.md` section 5 for the finding
this code reproduces: the `period` field is a compound of genuine temporal
window, reporting basis (actual/forecast/budget), and entry-relative anchor
("At Close", "Entry") rather than a clean period label, and a naive
same-label match systematically overcounts period pairs. `SCENARIO_BASIS`
and `AMBIGUOUS_RELATIVE` encode the exclusion rules that made that finding
reproducible -- extend those sets (not the pairing logic) as more raw label
variants are observed in the data.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SCENARIO_BASIS: frozenset[str] = frozenset(
    {
        "A",
        "F",
        "E",
        # Added 2026-08-12, confirmed via a spot-checked example: the raw
        # `period` field for one instance was literally "Year-to-Date B",
        # and a bare "B" (11 occurrences in this dataset's raw label
        # vocabulary) follows the exact same single-letter-code convention
        # as "A"/"F"/"E" (Actual/Forecast/Estimate) already handled above --
        # "B" = Budget. Word-boundary matching (see `_contains_any_keyword`)
        # means this also now correctly catches compound labels like
        # "Year-to-Date B", not just a bare "B" on its own.
        "B",
        "Budget",
        "Forecast",
        "Actual",
        "Estimate",
        "LTM A",
        # Added 2026-08-12 after a spot-check surfaced "1Q GAAP"-style
        # labels slipping through as ordinary periods: GAAP vs. non-GAAP is
        # explicitly listed as its own basis category in this project's own
        # confusion-pair taxonomy (report section 5C), the same kind of
        # non-period semantic dimension as Budget/Forecast. "Reported",
        # "Pro-Forma"/"ProForma", "Landing", "Target", and "Reforecast" are
        # the same class of basis/scenario qualifier observed in this
        # dataset's raw period-label vocabulary (e.g. "12M Reported", "5M
        # ProForma", "12M Reforecast") and are excluded for consistency,
        # even though (like "Actual") this conservatively also excludes
        # same-qualifier pairs (two "Reported" periods from different
        # years) that would arguably be fine -- see the module's open
        # question about that tradeoff.
        "GAAP",
        "Reported",
        "Pro-Forma",
        "ProForma",
        "Pro Forma",
        "Landing",
        "Target",
        "Reforecast",
    }
)
AMBIGUOUS_RELATIVE: frozenset[str] = frozenset(
    {"Prior Qtr", "Curr Qtr", "At Close", "Investment", "Acquisition", "Entry", "Current", "Close"}
)
ACTUAL_TEMPORAL: frozenset[str] = frozenset(
    {
        "LTM",
        "Q1",
        "Q2",
        "Q3",
        "Q4",
        "YTD",
        "Quarter Ended",
        "1Q",
        "2Q",
        "3Q",
        "4Q",
        "Year-to-Date",
        "12M Audited",
        "H1",
        "H2",
        "FY",
    }
)

ACTUAL_LABELS: frozenset[str] = frozenset({"A", "Actual"})
FORECAST_LABELS: frozenset[str] = frozenset({"F", "E", "Forecast", "Estimate"})

# Casefolded lookup versions of the four label sets above -- confirmed real
# bug, 2026-08-12: raw labels appear in multiple cases in this dataset
# ("BUDGET"/"Budget"/"budget", "Actual"/"actual"), and the original exact-
# string membership checks missed every variant but the one literally typed
# into the set above. A label like "BUDGET" then fell through to
# "unknown_other" instead of "scenario_basis", letting an Actual-vs-Budget
# pair through as if it were a genuine period pair -- exactly the pattern
# this classification exists to exclude. All matching below is casefolded;
# the display-case sets above are kept only as the canonical/documented
# vocabulary.
_SCENARIO_BASIS_CF: frozenset[str] = frozenset(s.casefold() for s in SCENARIO_BASIS)
_AMBIGUOUS_RELATIVE_CF: frozenset[str] = frozenset(s.casefold() for s in AMBIGUOUS_RELATIVE)
_ACTUAL_TEMPORAL_CF: frozenset[str] = frozenset(s.casefold() for s in ACTUAL_TEMPORAL)
_ACTUAL_LABELS_CF: frozenset[str] = frozenset(s.casefold() for s in ACTUAL_LABELS)
_FORECAST_LABELS_CF: frozenset[str] = frozenset(s.casefold() for s in FORECAST_LABELS)

_MONTH_TABLE: Mapping[str, str] = {
    "1": "01", "01": "01", "jan": "01", "january": "01",
    "2": "02", "02": "02", "feb": "02", "february": "02",
    "3": "03", "03": "03", "mar": "03", "march": "03",
    "4": "04", "04": "04", "apr": "04", "april": "04",
    "5": "05", "05": "05", "may": "05",
    "6": "06", "06": "06", "jun": "06", "june": "06",
    "7": "07", "07": "07", "jul": "07", "july": "07",
    "8": "08", "08": "08", "aug": "08", "august": "08",
    "9": "09", "09": "09", "sep": "09", "sept": "09", "september": "09",
    "10": "10", "oct": "10", "october": "10",
    "11": "11", "nov": "11", "november": "11",
    "12": "12", "dec": "12", "december": "12",
}  # fmt: skip


def _contains_any_keyword(text: str, keywords: frozenset[str]) -> bool:
    return any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords)


def classify_period_label(period: str | None) -> str:
    """"missing" | "scenario_basis" | "ambiguous_relative" | "actual_temporal" | "unknown_other".

    Matching is casefolded and by whole-word/phrase substring, not exact
    full-string equality -- two confirmed real bugs, 2026-08-12:

    1. Raw labels appear in multiple cases ("BUDGET"/"Budget"/"budget");
       exact-case matching missed every variant but the one literally typed
       into the reference sets.
    2. Many raw labels are *compound* -- "12M Budget", "PQ Actual", "5M
       Actual" -- and exact full-string matching against e.g. "Budget"
       never matches "12M Budget" at all, so these fell through to
       "unknown_other" and were treated as ordinary, comparable periods
       even though they clearly encode a basis/scenario or entry-relative
       signal. A spot-check surfaced this as a real Actual-vs-Budget pair
       let through as a fake "period" pair.

    Word-boundary substring matching (not bare `in`) avoids matching a
    keyword inside an unrelated longer word (e.g. "a" must not match inside
    "actual" via the wrong keyword). Category precedence -- scenario_basis,
    then ambiguous_relative, then actual_temporal -- is deliberate: if a
    label contains signals from more than one category, err toward
    excluding it from period-pairing rather than toward classifying it as a
    clean period.
    """

    if period is None:
        return "missing"
    normalized = period.strip().casefold()
    if _contains_any_keyword(normalized, _SCENARIO_BASIS_CF):
        return "scenario_basis"
    if _contains_any_keyword(normalized, _AMBIGUOUS_RELATIVE_CF):
        return "ambiguous_relative"
    if _contains_any_keyword(normalized, _ACTUAL_TEMPORAL_CF):
        return "actual_temporal"
    return "unknown_other"


def normalize_year(year: str | None) -> str | None:
    """2-digit years are assumed 2000s (all observed values -- 23, 22, ...,
    13 -- are plausible only as 2000s given this dataset's report dates)."""

    if year is None:
        return None
    year = year.strip()
    if len(year) == 2 and year.isdigit():
        return "20" + year
    if len(year) == 4 and year.isdigit():
        return year
    return None


def normalize_month(month: str | None) -> str | None:
    if month is None:
        return None
    return _MONTH_TABLE.get(month.strip().lower())


@dataclass(frozen=True)
class KPIRecord:
    """Minimal, package-independent view of one `KPIs` grouped-field
    instance -- just what the pairing logic needs."""

    document_id: str
    page_id: str
    instance_index: int
    company_key: str  # normalized or hashed company name -- caller decides
    period: str | None
    year: str | None
    month: str | None
    metric_values: Mapping[str, str]
    # Representative x-coordinate of this instance's metric-field bounding
    # boxes (see `KPI_METRIC_FIELDS`) -- i.e. which physical table column
    # this instance is. Optional and defaults to None so existing callers
    # that don't have bbox data keep working; but without it, two instances
    # that happen to share the same free-text period label can't be told
    # apart from two genuinely different (mislabeled) columns -- see
    # `_different_columns` and `_cluster_by_column` for why this matters.
    column_x: float | None = None


def group_by_company(records: Sequence[KPIRecord]) -> dict[tuple[str, str], list[KPIRecord]]:
    groups: dict[tuple[str, str], list[KPIRecord]] = defaultdict(list)
    for record in records:
        groups[(record.document_id, record.company_key)].append(record)
    return groups


def column_x_from_bboxes(
    field_bboxes: Mapping[str, tuple[int, int, int, int]], metric_fields: Sequence[str]
) -> float | None:
    """Mean left-coordinate of an instance's populated metric-field bounding
    boxes -- a proxy for "which table column is this", used to build
    `KPIRecord.column_x`. Takes `metric_fields` explicitly (rather than
    importing `KPI_METRIC_FIELDS` from the adapter) so this module stays
    independent of the private-package-backed adapter -- callers pass
    `KPI_METRIC_FIELDS` in. Excludes `company_name`/`period`/`month`/`year`,
    which are often row-level labels spanning the whole row width rather
    than column-specific.
    """

    lefts = [bbox[1] for name, bbox in field_bboxes.items() if name in metric_fields]
    return sum(lefts) / len(lefts) if lefts else None


_DEFAULT_COLUMN_POSITION_THRESHOLD = 20.0


def _different_columns(a: KPIRecord, b: KPIRecord, threshold: float = _DEFAULT_COLUMN_POSITION_THRESHOLD) -> bool:
    """True only if both instances have a known column position AND those
    positions are clearly apart -- confirmed necessary, 2026-08-12: a real
    the private dataset page has two parallel columns sharing one nominal period
    label (688,709-765) and (688,828-884); treating them as "the same
    period reported twice" was wrong. When position is unknown for either
    side, this conservatively returns False (can't rule out that it's a
    genuine duplicate) rather than guessing."""

    if a.column_x is None or b.column_x is None:
        return False
    return abs(a.column_x - b.column_x) > threshold


def _cluster_by_column(
    records: Sequence[KPIRecord], threshold: float = _DEFAULT_COLUMN_POSITION_THRESHOLD
) -> list[list[KPIRecord]]:
    """Group records whose `column_x` values are close together (same
    physical table column). Records without a `column_x` are kept together
    in one cluster -- same conservative default as `_different_columns`
    ("can't confirm they're in different columns, so don't split them
    apart"), which also keeps this backward-compatible with callers that
    don't have bbox data at all."""

    positioned = sorted((r for r in records if r.column_x is not None), key=lambda r: r.column_x)  # type: ignore[arg-type,return-value]
    unpositioned = [r for r in records if r.column_x is None]

    clusters: list[list[KPIRecord]] = []
    current: list[KPIRecord] = []
    for record in positioned:
        if current and abs(record.column_x - current[-1].column_x) > threshold:  # type: ignore[operator]
            clusters.append(current)
            current = []
        current.append(record)
    if current:
        clusters.append(current)

    if unpositioned:
        clusters.append(unpositioned)
    return clusters


@dataclass(frozen=True)
class PairCandidate:
    document_id: str
    company_key: str
    pair_type: str  # "period" | "basis"
    instance_a: int
    instance_b: int
    page_a: str
    page_b: str


@dataclass(frozen=True)
class PeriodPairSummary:
    valid_period_pairs: tuple[PairCandidate, ...]
    basis_pairs: tuple[PairCandidate, ...]
    excluded_ambiguous_count: int
    identical_period_key_count: int
    groups_with_valid_period_pair: int
    groups_with_basis_pair: int


def find_period_and_basis_pairs(
    groups: Mapping[tuple[str, str], Sequence[KPIRecord]],
) -> PeriodPairSummary:
    """Replicates the exclusion logic behind the report's section 5A/5C
    numbers: scenario/basis and entry-relative labels are excluded from
    period pairs (they're not genuinely different reporting periods), a
    clean Actual-vs-Forecast/Estimate same-year pair is split out separately
    as a basis pair, and identical normalized (year, month, period) pairs
    are treated as suspect duplicates *unless* their bounding boxes place
    them in clearly different table columns (`_different_columns`) -- in
    which case they're evidently two distinct, if identically mislabeled,
    periods and become a valid period pair instead. When column position
    isn't available, identical-key pairs are counted separately rather than
    silently dropped or silently treated as confirming duplicates (see
    `check_duplicate_period_key_value_agreement`)."""

    valid_period_pairs: list[PairCandidate] = []
    basis_pairs: list[PairCandidate] = []
    excluded_ambiguous = 0
    identical_period_key = 0
    groups_with_valid = 0
    groups_with_basis = 0

    for (document_id, company_key), records in groups.items():
        if len(records) < 2:
            continue
        enriched = [
            (record, classify_period_label(record.period), normalize_year(record.year), normalize_month(record.month))
            for record in records
        ]
        group_has_valid = False
        group_has_basis = False
        for i in range(len(enriched)):
            for j in range(i + 1, len(enriched)):
                a, a_class, a_year, a_month = enriched[i]
                b, b_class, b_year, b_month = enriched[j]

                labels_cf = {p.strip().casefold() for p in (a.period, b.period) if p is not None}
                if (
                    a_year is not None
                    and a_year == b_year
                    and (labels_cf & _ACTUAL_LABELS_CF)
                    and (labels_cf & _FORECAST_LABELS_CF)
                ):
                    basis_pairs.append(
                        PairCandidate(
                            document_id, company_key, "basis", a.instance_index, b.instance_index, a.page_id, b.page_id
                        )
                    )
                    group_has_basis = True
                    continue

                if a_class == "ambiguous_relative" or b_class == "ambiguous_relative":
                    excluded_ambiguous += 1
                    continue
                if a_class == "scenario_basis" or b_class == "scenario_basis":
                    excluded_ambiguous += 1
                    continue
                if a_year is None or b_year is None:
                    excluded_ambiguous += 1
                    continue

                key_a = (a_year, a_month, a.period)
                key_b = (b_year, b_month, b.period)
                if key_a == key_b and not _different_columns(a, b):
                    identical_period_key += 1
                    continue

                valid_period_pairs.append(
                    PairCandidate(
                        document_id, company_key, "period", a.instance_index, b.instance_index, a.page_id, b.page_id
                    )
                )
                group_has_valid = True

        if group_has_valid:
            groups_with_valid += 1
        if group_has_basis:
            groups_with_basis += 1

    return PeriodPairSummary(
        valid_period_pairs=tuple(valid_period_pairs),
        basis_pairs=tuple(basis_pairs),
        excluded_ambiguous_count=excluded_ambiguous,
        identical_period_key_count=identical_period_key,
        groups_with_valid_period_pair=groups_with_valid,
        groups_with_basis_pair=groups_with_basis,
    )


def _iter_duplicate_period_key_column_clusters(
    groups: Mapping[tuple[str, str], Sequence[KPIRecord]],
    position_threshold: float = _DEFAULT_COLUMN_POSITION_THRESHOLD,
):
    """Yield each same-(company, normalized year/month/period)-label cluster
    of >=2 instances that ALSO sit in the same table column (or have no
    known column position at all -- see `_cluster_by_column`).

    This is the shared "candidate genuine duplicate" definition behind both
    `check_duplicate_period_key_value_agreement` and
    `find_disagreeing_duplicate_instances`. Confirmed necessary, 2026-08-12:
    grouping by label alone (without this column split) wrongly treats two
    parallel same-labeled columns in the source table as one fact reported
    twice.
    """

    for records in groups.values():
        by_key: dict[tuple[str, str | None, str | None], list[KPIRecord]] = defaultdict(list)
        for record in records:
            year = normalize_year(record.year)
            if year is None:
                continue
            by_key[(year, record.month, record.period)].append(record)
        for key_records in by_key.values():
            if len(key_records) < 2:
                continue
            for cluster in _cluster_by_column(key_records, position_threshold):
                if len(cluster) >= 2:
                    yield cluster


def check_duplicate_period_key_value_agreement(
    groups: Mapping[tuple[str, str], Sequence[KPIRecord]],
    metric_fields: Sequence[str] = ("sales", "EBITDA", "net_debt", "EV"),
    position_threshold: float = _DEFAULT_COLUMN_POSITION_THRESHOLD,
) -> dict[str, Any]:
    """For instances sharing the same (company, normalized year/month/period)
    label AND the same table column (see
    `_iter_duplicate_period_key_column_clusters`), do their metric values
    actually agree?

    See `docs/company-benchmark-diagnosis-report.md` section 2.3 -- a low
    agreement rate here means the free-text period label, even after
    position-disambiguation, is not a reliable uniqueness key for that
    specific cluster, and those instances should be treated as suspect
    rather than assumed identical.
    """

    agree = 0
    disagree = 0
    for cluster in _iter_duplicate_period_key_column_clusters(groups, position_threshold):
        any_compared = False
        all_agree = True
        for metric in metric_fields:
            present = [r.metric_values[metric] for r in cluster if metric in r.metric_values]
            if len(present) >= 2:
                any_compared = True
                if len(set(present)) > 1:
                    all_agree = False
        if not any_compared:
            continue
        if all_agree:
            agree += 1
        else:
            disagree += 1

    total = agree + disagree
    return {
        "duplicate_groups_checked": total,
        "fully_agree": agree,
        "disagree_on_at_least_one_metric": disagree,
        "disagreement_rate": (disagree / total) if total else None,
    }


def find_disagreeing_duplicate_instances(
    groups: Mapping[tuple[str, str], Sequence[KPIRecord]],
    metric_fields: Sequence[str] = ("sales", "EBITDA", "net_debt", "EV"),
    position_threshold: float = _DEFAULT_COLUMN_POSITION_THRESHOLD,
) -> set[tuple[str, str, int]]:
    """(document_id, page_id, instance_index) triples for instances that
    belong to a same-label, same-column duplicate cluster
    (`_iter_duplicate_period_key_column_clusters`) with an internal value
    disagreement on at least one metric.

    Callers building confusion pairs should drop any pair touching a
    flagged instance rather than trusting its value -- see
    `docs/company-benchmark-diagnosis-report.md` section 2.3. Deliberately
    instance-level, not label-level: flagging a whole (year, month, period)
    label would also wrongly exclude other, genuinely distinct columns that
    happen to share that label (see `_different_columns`).
    """

    flagged: set[tuple[str, str, int]] = set()
    for cluster in _iter_duplicate_period_key_column_clusters(groups, position_threshold):
        for metric in metric_fields:
            present = [(record, record.metric_values[metric]) for record in cluster if metric in record.metric_values]
            if len(present) >= 2 and len({value for _record, value in present}) > 1:
                flagged.update((record.document_id, record.page_id, record.instance_index) for record, _value in present)
    return flagged


def count_metric_pair_co_occurrence(
    field_value_maps: Sequence[Mapping[str, str]],
    candidate_pairs: Sequence[tuple[str, str]],
) -> dict[tuple[str, str], int]:
    """How many instances have BOTH fields of a candidate metric-pair
    populated.

    Co-occurrence alone does not make a pair valid -- see
    `docs/company-benchmark-diagnosis-report.md` section 5B: only pairs with
    a defensible shared-root/gross-net/component relation should be checked
    here, and every final pair still needs human adjudication.
    """

    counts: dict[tuple[str, str], int] = {pair: 0 for pair in candidate_pairs}
    for values in field_value_maps:
        for a, b in candidate_pairs:
            if a in values and b in values:
                counts[(a, b)] += 1
    return counts
