"""Natural Financial Confusion Pair construction for TAT-QA.

Unlike the source-level-rerender counterfactual pipeline
(``tatqa_source_rerender.py``), this module never edits or re-renders a
table: it pairs two *naturally occurring* questions whose gold answers both
live in the same untouched TAT-QA table but differ in exactly one
financially meaningful dimension (period, metric, accounting basis, or
unit/scale). The resulting benchmark measures whether a model confuses two
real, competing facts on an unedited document image -- a different question
from "does the model behave correctly under a synthetic intervention"
(TVFR/HFR/ISR/CGS).

This module is pure logic (no PIL, no image work) and reuses the existing
eligibility/extraction building blocks from ``tatqa_source_rerender`` and
``tatdqa_pilot`` rather than re-deriving them:

- ``tatdqa_pilot.is_direct_extraction_question`` -- the high-confidence
  single-scalar-lookup filter (rejects comparison questions, non-scalar or
  multi-span answers, non-numeric answers, ``count``-type answers, and
  arithmetic questions whose derivation isn't a direct restatement of the
  answer).
- ``tatqa_source_rerender.locate_answer_cell`` -- requires the answer to
  resolve to exactly one grid cell.
- ``tatqa_source_rerender.build_evidence_packet`` -- classifies the header
  context (row/column headers) around that cell.

The eligibility pool used here is deliberately *looser* than
``tatqa_source_rerender.build_variant_plans``'s CF-accepted set: natural
pairs need each side's answer to be uniquely locatable, but do not need a
header-swap partner or a safe irrelevant cell to exist (those are CF-plan
concerns only).

Per ``AGENTS.md`` dataset roles, TAT-QA is public real-world evaluation
data: this benchmark is eval-only and must never be used for training.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import random
import re
from typing import Any, Callable, Mapping, Sequence

from financial_vlm.data.tatdqa_pilot import (
    is_direct_extraction_question,
    normalize_number_text,
    scalar_answer_text,
)
from financial_vlm.data.tatqa_loader import metric_and_period, safe_slug
from financial_vlm.data.tatqa_source_rerender import (
    EvidencePacket,
    GridCell,
    YEAR_RE,
    build_evidence_packet,
    is_numeric_text,
    locate_answer_cell,
)


# Bumped only when construction rules themselves change. Per the frozen
# Strategy C protocol, rules must never change based on later model
# results -- a rule change means a new version, not a silent rewrite of v1.
#
# v1 -> v2 (2026-08-11): the user reviewed a spot-check sample of the v1
# metric_candidate pool (882 candidates, jaccard >= 0.34) and decided
# against a per-candidate manual adjudication pass -- the spot check found
# the false-positive rate concentrated below jaccard 0.50 (e.g. "Weighted
# average number of shares outstanding incl. dilutive effect" vs "Weighted
# average number of treasury shares", jaccard 0.36 -- lexically overlapping
# but not a genuine same-concept confusion), while jaccard >= 0.50 and
# basis_qualifier matches looked clean on inspection. v2 raises
# METRIC_RELATEDNESS_JACCARD_THRESHOLD to 0.50 and excludes footnote-suffix
# duplicate rows (see ``is_footnote_suffix_duplicate``) so the threshold
# gate itself does the accept/reject work that per-candidate human labels
# were going to do -- this is a stricter automatic construction rule, not a
# model or human judgment call on any individual candidate.
#
# v2 -> v3 (2026-08-11): root-cause diagnosis of the model evaluation results
# (docs/tatqa-natural-pairs-evaluation-report.md) included a data-quality
# scan, independent of any model prediction, that found 6/1302 period_derived
# pairs had a numeric-looking metric_raw (e.g. "$ (50.5)", "(2)%") -- a
# malformed-table row where ``build_evidence_packet``'s nearest row header
# above a value cell was itself another value cell, not a text label. v3
# rejects these at fact-extraction time (see ``is_numeric_like_metric_label``)
# so no fact is ever built on a non-text metric label.
#
# v3 -> v4 (2026-08-12): the evaluation report's broader-confusion finding
# (docs/tatqa-natural-pairs-evaluation-report.md) showed non-adjacent-period
# confusion (e.g. 2017 vs 2019) is a real, measurable phenomenon that
# adjacent-only period pairing could never test directly. The user asked to
# allow any two periods within a metric group to pair (not just
# chronologically adjacent ones), capped per (table, metric) group and per
# table -- mirroring the structural-sampling approach used for the private
# company benchmark -- instead of relying on adjacency alone to bound pool
# size. See METRIC_GROUP_PERIOD_PAIR_CAP / TABLE_PERIOD_PAIR_CAP and
# build_period_pairs_for_table for the real-data numbers behind the cap
# values and the deterministic (gap-ascending) selection rule.
PROTOCOL_VERSION = "v4"

PAIR_TYPES = ("period", "metric", "basis", "unit")

# Fixed, documented vocabulary for the "basis" dimension (Type C). A pair is
# only ever classified as "basis" if one metric label contains exactly one
# of these qualifier words/phrases and the other doesn't, with the same
# base label otherwise -- never inferred from anything but the label text.
BASIS_QUALIFIER_WORDS = (
    "adjusted",
    "reported",
    "actual",
    "budget",
    "gaap",
    "non-gaap",
    "underlying",
    "core",
    "normalized",
    "normalised",
)

# Small, fixed related-term clusters used only for the secondary
# `confusable_metric_flag` on metric pairs -- never for inclusion/exclusion.
CONFUSABLE_METRIC_GROUPS = (
    frozenset({"revenue", "sales", "turnover"}),
    frozenset({"profit", "income", "earnings"}),
    frozenset({"debt", "borrowings", "liabilities"}),
    frozenset({"ebitda"}),
    frozenset({"margin"}),
    frozenset({"cash", "liquidity"}),
    frozenset({"expense", "expenses", "cost", "costs"}),
)

# --- Table-derived (Strategy B/C) construction rules -----------------------
#
# Per docs/tatqa-natural-pairs-construction-diagnosis.md: applying
# build_evidence_packet to *every* numeric cell (not just cells an existing
# TAT-QA question happens to reference) recovers a (metric, period) address
# for 65% of numeric cells, vs. the ~11% of *questions* that survive the
# existing-question filter -- this is what unlocks a benchmark substantially
# larger than the human-authored subset.

DEGENERATE_METRIC_RE = re.compile(r"^[\W_]*$")
FULL_DATE_RE = re.compile(
    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s*(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
PER_SHARE_RE = re.compile(r"\bper[\s-]+(?:share|unit)\b", re.IGNORECASE)
PERCENT_VALUE_RE = re.compile(r"%\s*\)?$")

# v2: raised from 0.34 (see PROTOCOL_VERSION changelog above) -- the v1
# spot-check found real false positives below 0.50 (lexical overlap without
# genuine same-concept confusion, e.g. "shares outstanding incl. dilutive
# effect" vs "number of treasury shares") while >= 0.50 and basis_qualifier
# matches looked clean. This is now the sole acceptance gate for the
# metric_candidate subset -- there is no separate human/model adjudication
# step downstream of it.
METRIC_RELATEDNESS_JACCARD_THRESHOLD = 0.50
_JACCARD_STOPWORDS = frozenset({"of", "the", "and", "in", "for", "to", "a", "an"})

# A single trailing footnote-style marker -- an optional space, then one
# digit optionally parenthesized -- with no other change to the label
# indicates the *same* underlying line item annotated with two different
# footnote references (e.g. "Corporate items and intercompany eliminations"
# vs "Corporate items and intercompany eliminations 2"), not two distinct,
# naturally confusable metrics. Found via v1 spot-check (8/882 candidates).
_FOOTNOTE_SUFFIX_RE = re.compile(r"\s*\(?\d\)?$")


def is_footnote_suffix_duplicate(metric_a_norm: str, metric_b_norm: str) -> bool:
    base_a = _FOOTNOTE_SUFFIX_RE.sub("", metric_a_norm).strip()
    base_b = _FOOTNOTE_SUFFIX_RE.sub("", metric_b_norm).strip()
    return bool(base_a) and base_a == base_b and metric_a_norm != metric_b_norm


def is_degenerate_metric_label(text: str | None) -> bool:
    return text is None or DEGENERATE_METRIC_RE.match(text.strip()) is not None


# Deliberately more permissive than tatqa_source_rerender.NUMERIC_RE (a
# strict single-value-cell format check): a malformed row header can carry
# looser formatting than a value cell would (e.g. a space before the
# closing paren in "(6 )", or "$" before "(" in "$ (50.5)") and should
# still be rejected as non-text. Any string built entirely from digits,
# whitespace, and common numeric punctuation is treated as numeric-like.
_NUMERIC_LIKE_METRIC_RE = re.compile(r"^[\s$£€()%.,0-9-]+$")


def is_numeric_like_metric_label(text: str | None) -> bool:
    """True if a row-header label is actually a numeric value, not real
    text -- found via the v2 root-cause diagnosis (see PROTOCOL_VERSION
    changelog): ``build_evidence_packet``'s nearest-row-header lookup can
    land on another value cell in a malformed/hierarchical table section,
    producing a "metric" like ``"$ (50.5)"`` or ``"(2)%"``."""

    if text is None:
        return False
    stripped = text.strip()
    return bool(stripped) and bool(_NUMERIC_LIKE_METRIC_RE.match(stripped))


def is_bare_total_row(metric_text: str | None) -> bool:
    """True only for a bare ``"Total"``/``"Totals"`` label.

    A fully contextualized label like ``"Total current assets"`` is *not*
    bare -- its semantic content survives beyond the word "Total", so it
    stays eligible. Only the content-free case is excluded.
    """

    if metric_text is None:
        return False
    return metric_text.strip().casefold() in ("total", "totals")


def temporal_remainder(period_text: str) -> str:
    """Strip full-date and bare-year tokens, returning whatever text (if
    any) remains.

    Two periods with equal remainders differ *only* in their temporal
    component (``"2019"`` vs ``"2018"``, ``"May 31, 2019"`` vs
    ``"November 30, 2018"``) and are a legitimate period pair. Two periods
    with *different* remainders differ in some non-temporal way (e.g.
    ``"2019 actual"`` vs ``"2019 threshold"``, remainders ``"actual"`` vs
    ``"threshold"``) and must not be treated as a period pair -- see
    docs/tatqa-natural-pairs-construction-diagnosis.md Step 4/8.
    """

    text = FULL_DATE_RE.sub("", period_text)
    text = YEAR_RE.sub("", text)
    text = re.sub(r"[,:;]+", " ", text)
    return normalize_metric_text(text)


def period_sort_key(period_text: str) -> tuple[int, str]:
    match = YEAR_RE.search(period_text)
    year = int(match.group()) if match else 0
    return (year, period_text)


def unit_class(value_text: str, metric_text: str | None, unit_tag: str | None) -> str:
    """Coarse compatibility class for a fact's value.

    Rejects pairs that look lexically related but sit on incompatible
    scales -- most importantly a per-share figure next to an aggregate
    figure (found via manual audit, Step 8: ``"Diluted net income (loss)
    per share"`` = 0.55 vs ``"Net income (loss)"`` = 40,913, same table,
    same period, high lexical overlap, completely different scale).
    """

    if PER_SHARE_RE.search(metric_text or ""):
        return "per_share"
    if PERCENT_VALUE_RE.search(value_text.strip()):
        return "percent"
    return unit_tag or "unscaled"


def lexical_jaccard(metric_a_norm: str, metric_b_norm: str) -> float:
    tokens_a = {t for t in metric_a_norm.split() if t not in _JACCARD_STOPWORDS and len(t) > 1}
    tokens_b = {t for t in metric_b_norm.split() if t not in _JACCARD_STOPWORDS and len(t) > 1}
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _table_header_row_count(cells: Sequence[GridCell]) -> int:
    """Number of leading rows whose column-0 cell is blank -- TAT-QA's
    dominant hierarchical-header pattern (a spanning title row above the
    year row; 50.9%/54.7% of test_gold/dev tables have this). Computed once
    per table, shared by every column, rather than
    ``EvidencePacket.spanning_headers`` which is contaminated by whatever
    data row happens to sit above a *specific* value cell (see
    ``_column_scope_text`` docstring).
    """

    by_row_col = {(cell.row_idx, cell.col_idx): cell for cell in cells}
    nrows = max((cell.row_idx for cell in cells), default=-1) + 1
    count = 0
    for row_idx in range(nrows):
        col0 = by_row_col.get((row_idx, 0))
        if col0 is not None and col0.text.strip():
            break
        count += 1
    return count


def _column_scope_text(
    by_row_col: Mapping[tuple[int, int], GridCell],
    col_idx: int,
    header_row_count: int,
) -> str | None:
    """Stable per-column "which section of the table" signal, used to
    reject metric-candidate pairs with an unresolved scope difference.

    Deliberately *not* ``EvidencePacket.spanning_headers`` -- that field is
    computed relative to one specific value cell's row (`row_idx <
    value_cell.row_idx`), so two facts in the same column but different
    rows can see a different set of "cells above" (an intervening metric's
    own value row leaks in for whichever fact sits lower). Measured impact:
    using ``spanning_headers[-1]`` as the scope gate dropped 91% of
    relatedness-passing candidates as a false-positive scope mismatch. This
    version only looks at the table's fixed leading header-row band (same
    boundary for every column), so it returns the same text for every fact
    in a given column regardless of which data row it's in.
    """

    parts = []
    for row_idx in range(header_row_count):
        cell = by_row_col.get((row_idx, col_idx))
        if cell is not None and cell.text and not YEAR_RE.search(cell.text):
            parts.append(cell.text.strip())
    text = " ".join(parts).strip()
    return text or None


@dataclass(frozen=True)
class TableFact:
    """A single reliably-addressable (metric, period, value) fact, derived
    directly from one numeric table cell -- independent of whether any
    TAT-QA question happens to ask about that cell."""

    table_uid: str
    split: str
    cell: GridCell
    value_raw: str
    value_norm: str
    metric_raw: str
    metric_norm: str
    period_raw: str
    unit_class: str
    scope: str | None
    row_header_ids: tuple[str, ...]
    column_header_ids: tuple[str, ...]

    @property
    def fact_id(self) -> str:
        return f"{self.table_uid}:{self.cell.cell_id}"


def extract_table_facts(
    doc: Mapping[str, Any],
    cells: Sequence[GridCell],
    split: str,
    stats: Any = None,
) -> list[TableFact]:
    """Derive a fact for every reliably-addressable numeric cell in a table.

    This is the Strategy-B/table-derived extraction path (as opposed to
    ``extract_eligible_question``, which only looks at cells an existing
    TAT-QA question references). See
    docs/tatqa-natural-pairs-construction-diagnosis.md Step 2 for the
    reliability numbers behind these filters.

    ``stats``, if given a ``collections.Counter``-like object, is
    incremented per rejection reason (``fact_rejected_<reason>``) and per
    accepted fact (``fact_accepted``) for the construction audit.
    """

    table_uid = str((doc.get("table") or {}).get("uid") or "")
    facts: list[TableFact] = []
    by_row_col = {(cell.row_idx, cell.col_idx): cell for cell in cells}
    header_row_count = _table_header_row_count(cells)

    for cell in cells:
        if not cell.text or not is_numeric_text(cell.text):
            continue
        if stats is not None:
            stats["fact_candidate_numeric_cells"] += 1
        packet = build_evidence_packet(cells, cell)
        metric = packet.row_headers[-1].text if packet.row_headers else None
        if is_degenerate_metric_label(metric):
            if stats is not None:
                stats["fact_rejected_no_metric_or_degenerate"] += 1
            continue
        if is_bare_total_row(metric):
            if stats is not None:
                stats["fact_rejected_bare_total_row"] += 1
            continue
        if is_numeric_like_metric_label(metric):
            if stats is not None:
                stats["fact_rejected_numeric_metric_label"] += 1
            continue

        period = None
        for header_cell in reversed(packet.column_headers):
            if YEAR_RE.search(header_cell.text):
                period = header_cell.text
                break
        if period is None:
            if stats is not None:
                stats["fact_rejected_no_period"] += 1
            continue

        # EvidencePacket.unit_cells is prone to a known false-positive (a
        # numeric value cell from an adjacent row gets misclassified as a
        # "unit cell" because is_unit_or_scale_text matches a bare '$'
        # without checking is_numeric_text first -- see
        # docs/tatqa-source-rerender-cf-feasibility.md and Step 2 of
        # docs/tatqa-natural-pairs-construction-diagnosis.md). Scanning
        # every numeric cell in a table (not just question-anchored ones)
        # surfaces this far more often than the CF pipeline ever saw it
        # (measured: 42% of non-trivial unit tags were a leaked numeric
        # value). Filtered locally here rather than patching the shared
        # tatqa_source_rerender.is_unit_or_scale_text, which is out of
        # scope for this module and used elsewhere in the CF pipeline.
        unit_tag = (
            " ".join(
                sorted(
                    {
                        c.text.strip().casefold()
                        for c in packet.unit_cells
                        if not is_numeric_text(c.text)
                    }
                )
            )
            or None
        )
        scope = _column_scope_text(by_row_col, cell.col_idx, header_row_count)

        facts.append(
            TableFact(
                table_uid=table_uid,
                split=split,
                cell=cell,
                value_raw=cell.text,
                value_norm=normalize_number_text(cell.text),
                metric_raw=metric,
                metric_norm=normalize_metric_text(metric),
                period_raw=period,
                unit_class=unit_class(cell.text, metric, unit_tag),
                scope=scope,
                row_header_ids=tuple(c.cell_id for c in packet.row_headers),
                column_header_ids=tuple(c.cell_id for c in packet.column_headers),
            )
        )
        if stats is not None:
            stats["fact_accepted"] += 1

    return facts


def _dedupe_by_key(facts: Sequence[TableFact], key_fn: Callable[[TableFact], Any]) -> dict[Any, TableFact]:
    """Group facts by ``key_fn``; keep a key only if every fact sharing it
    agrees on ``value_norm``. Ambiguous/conflicting-value groups (e.g. a
    generic sub-row label like "Percentage of revenue" repeated under
    several parent metrics) are dropped outright rather than guessed at.
    """

    groups: dict[Any, list[TableFact]] = defaultdict(list)
    for fact in facts:
        groups[key_fn(fact)].append(fact)
    chosen: dict[Any, TableFact] = {}
    for key, group in groups.items():
        if len({f.value_norm for f in group}) > 1:
            continue
        chosen[key] = group[0]
    return chosen


# v4 (see PROTOCOL_VERSION changelog): any two periods within a metric
# group may pair (not just chronologically adjacent ones), gated by two
# caps mirroring the structural-sampling approach used for the private
# company benchmark (per-page / per-metric-family / per-company limits).
# Real-data check (test_gold) before picking these numbers: metric-group
# period counts are almost always small (k=2: 754 groups, k=3: 317, k=4:
# 33, k=5: 91, max k=5 -> full C(k,2) never exceeds 10) but a table can
# stack many metric groups, and *uncapped* per-table period-pair totals
# range up to 180 (median 6, mean 11.8, p90 21, p99 106) -- the real
# table-dominance risk is at the table level, not the metric-group level.
METRIC_GROUP_PERIOD_PAIR_CAP = 4
TABLE_PERIOD_PAIR_CAP = 24


def _period_gap_years(period_a_raw: str, period_b_raw: str) -> int:
    return abs(period_sort_key(period_a_raw)[0] - period_sort_key(period_b_raw)[0])


def build_period_pairs_for_table(facts: Sequence[TableFact]) -> list[tuple[TableFact, TableFact, str]]:
    """Period pairs within a normalized-metric group, any two periods (not
    only chronologically adjacent ones) -- capped per metric group and per
    table (``METRIC_GROUP_PERIOD_PAIR_CAP`` / ``TABLE_PERIOD_PAIR_CAP``)
    instead of relying on adjacency alone to bound pool size. Within each
    cap, pairs with the smallest year gap are kept first (deterministic,
    no randomness) -- chronologically closer periods are both the most
    naturally confusable and, by construction, always survive first;
    larger-gap ("skip-period", e.g. 2017 vs 2019) pairs fill any remaining
    slots under the cap. ``construction_rule`` records which case a pair
    is, so downstream analysis can separate them.
    """

    by_metric: dict[str, list[TableFact]] = defaultdict(list)
    for fact in facts:
        by_metric[fact.metric_norm].append(fact)

    table_candidates: list[tuple[int, str, TableFact, TableFact, str]] = []
    for metric_norm, group in by_metric.items():
        by_period = _dedupe_by_key(group, lambda f: f.period_raw)
        ordered = sorted(by_period.values(), key=lambda f: period_sort_key(f.period_raw))

        metric_candidates: list[tuple[int, TableFact, TableFact, str]] = []
        for fact_a, fact_b in itertools.combinations(ordered, 2):
            if fact_a.value_norm == fact_b.value_norm:
                continue
            if fact_a.unit_class != fact_b.unit_class:
                continue
            if temporal_remainder(fact_a.period_raw) != temporal_remainder(fact_b.period_raw):
                continue
            gap = _period_gap_years(fact_a.period_raw, fact_b.period_raw)
            rule = "same_metric_adjacent_period_different_value" if gap == 1 else "same_metric_nonadjacent_period_different_value"
            metric_candidates.append((gap, fact_a, fact_b, rule))

        metric_candidates.sort(key=lambda item: (item[0], item[1].period_raw, item[2].period_raw))
        for gap, fact_a, fact_b, rule in metric_candidates[:METRIC_GROUP_PERIOD_PAIR_CAP]:
            table_candidates.append((gap, metric_norm, fact_a, fact_b, rule))

    table_candidates.sort(key=lambda item: (item[0], item[1], item[2].period_raw, item[3].period_raw))
    return [(fact_a, fact_b, rule) for _, _, fact_a, fact_b, rule in table_candidates[:TABLE_PERIOD_PAIR_CAP]]


def build_metric_candidates_for_table(
    facts: Sequence[TableFact],
    stats: Any = None,
) -> list[tuple[TableFact, TableFact, str, float]]:
    """Metric pairs accepted by the v2 threshold gate. Returns
    ``(fact_a, fact_b, matched_via, jaccard_score)``. Acceptance is decided
    entirely by this function (jaccard >= ``METRIC_RELATEDNESS_JACCARD_
    THRESHOLD`` or a basis-qualifier match, footnote-suffix duplicates
    excluded) -- there is no separate human/model adjudication step.

    ``stats``, if given a ``collections.Counter``-like object, is
    incremented for footnote-suffix duplicates rejected
    (``rejected_footnote_suffix_duplicate``).
    """

    by_period: dict[str, list[TableFact]] = defaultdict(list)
    for fact in facts:
        by_period[fact.period_raw].append(fact)

    candidates: list[tuple[TableFact, TableFact, str, float]] = []
    for group in by_period.values():
        by_metric = _dedupe_by_key(group, lambda f: f.metric_norm)
        metric_norms = sorted(by_metric.keys())
        for metric_a, metric_b in itertools.combinations(metric_norms, 2):
            fact_a, fact_b = by_metric[metric_a], by_metric[metric_b]
            if fact_a.value_norm == fact_b.value_norm:
                continue
            if fact_a.unit_class != fact_b.unit_class:
                continue
            if fact_a.scope != fact_b.scope:
                continue  # unresolved scope difference -- different section of the table
            if is_footnote_suffix_duplicate(metric_a, metric_b):
                if stats is not None:
                    stats["rejected_footnote_suffix_duplicate"] += 1
                continue

            base_a, qualifier_a = strip_basis_qualifiers(metric_a)
            base_b, qualifier_b = strip_basis_qualifiers(metric_b)
            score = lexical_jaccard(metric_a, metric_b)
            if base_a and base_a == base_b and qualifier_a != qualifier_b:
                matched_via = "basis_qualifier"
            elif score >= METRIC_RELATEDNESS_JACCARD_THRESHOLD:
                matched_via = "lexical_jaccard"
            else:
                continue

            candidates.append((fact_a, fact_b, matched_via, round(score, 4)))

    return candidates


QUESTION_TEMPLATE = "What was {metric} in {period}?"


def _fact_evidence_view(fact: TableFact) -> dict[str, Any]:
    return {
        "cell_id": fact.cell.cell_id,
        "row_idx": fact.cell.row_idx,
        "col_idx": fact.cell.col_idx,
        "row_header_ids": list(fact.row_header_ids),
        "column_header_ids": list(fact.column_header_ids),
    }


def build_derived_pair_record(
    fact_a: TableFact,
    fact_b: TableFact,
    pair_type: str,
    construction_rule: str,
    pair_index: int,
    subset: str,
    id_prefix: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one table-derived (Strategy B) pair record.

    Question text is filled from ``QUESTION_TEMPLATE`` only -- no
    generative model chooses target facts or phrasing, per the frozen
    protocol's deterministic-template-only rule.
    """

    document_id = safe_slug(fact_a.table_uid)
    pair_id = f"{id_prefix}_{pair_index:06d}"

    def side(fact: TableFact) -> dict[str, Any]:
        return {
            "question_id": fact.fact_id,
            "question": QUESTION_TEMPLATE.format(metric=fact.metric_raw, period=fact.period_raw),
            "answer": fact.value_raw,
            "normalized_answer": fact.value_norm,
            "metric": fact.metric_raw,
            "period": fact.period_raw,
            "unit_class": fact.unit_class,
            "scope": fact.scope,
            "evidence": _fact_evidence_view(fact),
        }

    side_a, side_b = side(fact_a), side(fact_b)
    record: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "subset": subset,
        "protocol_version": PROTOCOL_VERSION,
        "split": fact_a.split,
        "document_id": document_id,
        # TAT-QA has no page concept independent of the table: one doc == one
        # table, so there is no separate page identifier to record here.
        "page_id": None,
        "table_id": document_id,
        "question_a_id": side_a["question_id"],
        "question_a": side_a["question"],
        "answer_a": side_a["answer"],
        "normalized_answer_a": side_a["normalized_answer"],
        "metric_a": side_a["metric"],
        "period_a": side_a["period"],
        "unit_class_a": side_a["unit_class"],
        "scope_a": side_a["scope"],
        "evidence_a": side_a["evidence"],
        "question_b_id": side_b["question_id"],
        "question_b": side_b["question"],
        "answer_b": side_b["answer"],
        "normalized_answer_b": side_b["normalized_answer"],
        "metric_b": side_b["metric"],
        "period_b": side_b["period"],
        "unit_class_b": side_b["unit_class"],
        "scope_b": side_b["scope"],
        "evidence_b": side_b["evidence"],
        "changed_dimension": pair_type,
        "construction_rule": construction_rule,
        "question_generation_method": "deterministic_template",
        "validation_status": "structurally_valid",
    }
    if extra:
        record.update(extra)
    return record


@dataclass(frozen=True)
class EligibleQuestion:
    table_uid: str
    question_uid: str
    question_index: int
    question_text: str
    answer_raw: str
    normalized_answer: str
    answer_cell: GridCell
    metric: str | None
    period: str | None
    scale: str | None
    evidence_packet: EvidencePacket
    split: str


def extract_eligible_question(
    doc: Mapping[str, Any],
    question: Mapping[str, Any],
    question_index: int,
    cells: Sequence[GridCell],
    split: str,
) -> tuple[EligibleQuestion | None, str]:
    """Apply the reused eligibility filters, in the same order/vocabulary
    as ``scripts/audit_tatqa_source_rerender_cf.py``'s funnel, stopping at
    "uniquely locatable" -- this module does not require CF-plan
    feasibility (no header-swap partner or irrelevant cell needed).

    ``split`` (e.g. ``"test_gold"``/``"dev"``) is carried through to the
    frozen pair record so provenance stays inspectable -- pairs are only
    ever formed between two questions from the *same* split (see
    ``classify_pair``), since different splits were exposed to the trained
    models to different degrees during training/LR selection.
    """

    answer_from = question.get("answer_from")
    if answer_from not in ("table", "table-text"):
        return None, "rejected_not_table_grounded"

    ok, reason = is_direct_extraction_question(question)
    if not ok:
        return None, f"rejected_{reason}"

    located = locate_answer_cell(question, cells)
    if located is None:
        return None, "rejected_answer_cell_not_located"

    evidence_packet = build_evidence_packet(cells, located.answer_cell)
    metric, period = metric_and_period(evidence_packet)
    scale = str(question.get("scale") or "").strip() or None
    table_uid = str((doc.get("table") or {}).get("uid") or "")

    return (
        EligibleQuestion(
            table_uid=table_uid,
            question_uid=located.question_id,
            question_index=question_index,
            question_text=located.question,
            answer_raw=located.answer,
            normalized_answer=normalize_number_text(located.answer),
            answer_cell=located.answer_cell,
            metric=metric,
            period=period,
            scale=scale,
            evidence_packet=evidence_packet,
            split=split,
        ),
        "accepted",
    )


def normalize_metric_text(text: str) -> str:
    return " ".join(text.strip().casefold().split())


def strip_basis_qualifiers(normalized_metric: str) -> tuple[str, bool]:
    tokens = normalized_metric.split()
    kept = [token for token in tokens if token not in BASIS_QUALIFIER_WORDS]
    had_qualifier = len(kept) != len(tokens)
    return " ".join(kept), had_qualifier


def confusable_metric_flag(metric_a: str, metric_b: str) -> bool:
    tokens_a = set(normalize_metric_text(metric_a).split())
    tokens_b = set(normalize_metric_text(metric_b).split())
    for group in CONFUSABLE_METRIC_GROUPS:
        if tokens_a & group and tokens_b & group:
            return True
    return False


def classify_pair(qa: EligibleQuestion, qb: EligibleQuestion) -> tuple[str | None, str]:
    """Classify an unordered pair of eligible questions.

    Returns ``(pair_type, rule_or_rejection_reason)``. ``pair_type`` is one
    of ``PAIR_TYPES`` on acceptance, ``None`` on rejection.
    """

    if qa.table_uid != qb.table_uid:
        return None, "rejected_different_document"
    if qa.split != qb.split:
        return None, "rejected_different_split"
    if qa.question_uid == qb.question_uid:
        return None, "rejected_self_pair"
    if qa.normalized_answer == qb.normalized_answer:
        return None, "rejected_gold_answers_identical"

    metric_a = normalize_metric_text(qa.metric) if qa.metric else None
    metric_b = normalize_metric_text(qb.metric) if qb.metric else None
    period_a, period_b = qa.period, qb.period

    same_metric = metric_a is not None and metric_a == metric_b
    same_period = period_a is not None and period_a == period_b

    # Type C: basis -- same base metric (qualifier words stripped), exactly
    # one side carries a qualifier, period fixed (equal, or absent on both).
    if metric_a is not None and metric_b is not None and not same_metric:
        base_a, qualifier_a = strip_basis_qualifiers(metric_a)
        base_b, qualifier_b = strip_basis_qualifiers(metric_b)
        if base_a and base_a == base_b and (qualifier_a != qualifier_b):
            period_fixed = same_period or (period_a is None and period_b is None)
            if period_fixed:
                return "basis", "same_base_metric_qualifier_differs"

    # Type A: period -- same metric, different (both present) period.
    if same_metric and period_a is not None and period_b is not None and period_a != period_b:
        return "period", "same_metric_same_table_different_period"

    # Type B: metric -- same period, different (both present) metric.
    if same_period and metric_a is not None and metric_b is not None and metric_a != metric_b:
        return "metric", "same_period_same_table_different_metric"

    # Type D: unit -- same fact (metric+period both equal), differing,
    # non-empty scale annotation. Expected to be rare/absent in practice
    # since TAT-QA tables don't usually restate one fact under two scales;
    # implemented for completeness rather than forced.
    if same_metric and same_period and qa.scale and qb.scale and qa.scale != qb.scale:
        return "unit", "same_fact_different_scale_annotation"

    if metric_a is None or metric_b is None:
        return None, "rejected_metric_ambiguous"
    if period_a is None or period_b is None:
        return None, "rejected_period_ambiguous"
    if metric_a != metric_b and period_a != period_b:
        return None, "rejected_more_than_one_dimension_changed"
    return None, "rejected_no_matching_rule"


def _evidence_view(question: EligibleQuestion) -> dict[str, Any]:
    packet = question.evidence_packet
    return {
        "cell_id": question.answer_cell.cell_id,
        "row_idx": question.answer_cell.row_idx,
        "col_idx": question.answer_cell.col_idx,
        "row_header_ids": [cell.cell_id for cell in packet.row_headers],
        "column_header_ids": [cell.cell_id for cell in packet.column_headers],
    }


def build_pair_record(
    qa: EligibleQuestion,
    qb: EligibleQuestion,
    pair_type: str,
    rule: str,
    pair_index: int,
) -> dict[str, Any]:
    document_id = safe_slug(qa.table_uid)
    pair_id = f"tatqa_{pair_type}_{pair_index:06d}"
    flag = confusable_metric_flag(qa.metric, qb.metric) if pair_type == "metric" and qa.metric and qb.metric else False

    # Relatedness metadata (Step 7 of the construction diagnosis): most
    # human-authored metric pairs turn out NOT to pass the same relatedness
    # bar used to gate table-derived metric candidates -- record the score
    # rather than silently dropping the pair, so downstream analysis can
    # filter to a high-confidence sub-slice without discarding the rest.
    jaccard_score: float | None = None
    matched_via: str | None = None
    if pair_type == "metric" and qa.metric and qb.metric:
        metric_a_norm = normalize_metric_text(qa.metric)
        metric_b_norm = normalize_metric_text(qb.metric)
        base_a, qualifier_a = strip_basis_qualifiers(metric_a_norm)
        base_b, qualifier_b = strip_basis_qualifiers(metric_b_norm)
        jaccard_score = round(lexical_jaccard(metric_a_norm, metric_b_norm), 4)
        if base_a and base_a == base_b and qualifier_a != qualifier_b:
            matched_via = "basis_qualifier"
        elif jaccard_score >= METRIC_RELATEDNESS_JACCARD_THRESHOLD:
            matched_via = "lexical_jaccard"
        else:
            matched_via = "below_threshold"

    def side(question: EligibleQuestion) -> dict[str, Any]:
        return {
            "question_id": question.question_uid,
            "question": question.question_text,
            "answer": question.answer_raw,
            "normalized_answer": question.normalized_answer,
            "metric": question.metric,
            "period": question.period,
            "basis": question.metric,
            "unit": question.scale,
            "evidence": _evidence_view(question),
        }

    side_a = side(qa)
    side_b = side(qb)
    record: dict[str, Any] = {
        "pair_id": pair_id,
        "pair_type": pair_type,
        "subset": "human_authored",
        "protocol_version": PROTOCOL_VERSION,
        "split": qa.split,
        "document_id": document_id,
        # TAT-QA has no page concept independent of the table: one doc == one
        # table, so there is no separate page identifier to record here.
        "page_id": None,
        "table_id": document_id,
        "question_a_id": side_a["question_id"],
        "question_a": side_a["question"],
        "answer_a": side_a["answer"],
        "normalized_answer_a": side_a["normalized_answer"],
        "metric_a": side_a["metric"],
        "period_a": side_a["period"],
        "basis_a": side_a["basis"],
        "unit_a": side_a["unit"],
        "evidence_a": side_a["evidence"],
        "question_b_id": side_b["question_id"],
        "question_b": side_b["question"],
        "answer_b": side_b["answer"],
        "normalized_answer_b": side_b["normalized_answer"],
        "metric_b": side_b["metric"],
        "period_b": side_b["period"],
        "basis_b": side_b["basis"],
        "unit_b": side_b["unit"],
        "evidence_b": side_b["evidence"],
        "changed_dimension": pair_type,
        "confusable_metric_flag": flag,
        "relatedness_jaccard": jaccard_score,
        "relatedness_matched_via": matched_via,
        "construction_rule": rule,
        "question_generation_method": "existing_tatqa_question",
        "validation_status": "auto_high_confidence",
    }
    return record


def validate_natural_pairs(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Re-verify the frozen pair set. Raises ValueError on critical violations."""

    seen_pair_ids: set[str] = set()
    seen_unordered: set[frozenset[str]] = set()
    checks: dict[str, int] = {
        "total": len(records),
        "duplicate_pair_id": 0,
        "self_pair": 0,
        "identical_gold_answers": 0,
        "cross_document_pair": 0,
        "duplicate_or_reversed_pair": 0,
        "missing_evidence": 0,
        "pair_type_mismatch_on_recheck": 0,
    }

    for record in records:
        pair_id = str(record["pair_id"])
        if pair_id in seen_pair_ids:
            checks["duplicate_pair_id"] += 1
            raise ValueError(f"duplicate pair_id: {pair_id!r}")
        seen_pair_ids.add(pair_id)

        question_a_id = str(record["question_a_id"])
        question_b_id = str(record["question_b_id"])
        if question_a_id == question_b_id:
            checks["self_pair"] += 1
            raise ValueError(f"pair {pair_id!r}: question paired with itself")

        unordered = frozenset({question_a_id, question_b_id})
        if unordered in seen_unordered:
            checks["duplicate_or_reversed_pair"] += 1
            raise ValueError(f"pair {pair_id!r}: duplicate or reversed pair already seen")
        seen_unordered.add(unordered)

        if record["normalized_answer_a"] == record["normalized_answer_b"]:
            checks["identical_gold_answers"] += 1
            raise ValueError(f"pair {pair_id!r}: gold answers identical after normalisation")

        if record["document_id"] != record["table_id"]:
            checks["cross_document_pair"] += 1
            raise ValueError(f"pair {pair_id!r}: document_id/table_id mismatch")

        evidence_a = record.get("evidence_a") or {}
        evidence_b = record.get("evidence_b") or {}
        if not evidence_a.get("cell_id") or not evidence_b.get("cell_id"):
            checks["missing_evidence"] += 1
            raise ValueError(f"pair {pair_id!r}: missing evidence cell id")

        expected_type = record["pair_type"]
        if expected_type not in PAIR_TYPES:
            checks["pair_type_mismatch_on_recheck"] += 1
            raise ValueError(f"pair {pair_id!r}: unknown pair_type {expected_type!r}")

    return {
        "total_pairs": len(records),
        "unique_pair_ids": len(seen_pair_ids),
        "unique_question_pairs": len(seen_unordered),
        "checks_run": checks,
        "status": "passed",
    }


# --- Primary period-pair subset selection (structural sampling on top of
# the frozen v4 period_derived pool -- does not change v4's construction
# rules, only which of its already-valid pairs join a smaller "primary"
# benchmark) ------------------------------------------------------------
#
# Versioned separately from PROTOCOL_VERSION: this is a sampling layer on
# an already-frozen pair pool, not a construction-rule change. Mirrors the
# private company benchmark's structural sampling (per-page / per-metric-
# family / per-company caps): per (table, normalized metric) group, at
# most 1 adjacent + 1 skip-period pair; per table, an overall cap; then an
# overall stratified sample to hit the target adjacent/skip split. Uses
# only fields already in the frozen pair records (pair_id, table_id,
# metric_a, construction_rule) -- no model prediction is read anywhere in
# this selection.
SELECTION_PROTOCOL_VERSION = "v1"
ADJACENT_RULE = "same_metric_adjacent_period_different_value"
NONADJACENT_RULE = "same_metric_nonadjacent_period_different_value"


def select_primary_period_pairs(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    table_cap: int,
    target_adjacent: int,
    target_skip: int,
) -> dict[str, Any]:
    """Deterministically sample a structurally-balanced "primary" subset
    from a frozen period_derived pool.

    Stage 1 -- per (table_id, metric_a) group, keep at most 1 adjacent and
    1 skip-period candidate (seeded random choice when a group has more
    than one of a kind).
    Stage 2 -- per table, if stage 1's survivors exceed ``table_cap``,
    seeded-sample down to it (prevents a large multi-metric table from
    dominating).
    Stage 3 -- overall stratified seeded sample from stage 2's adjacent
    pool and skip pool down to ``target_adjacent`` / ``target_skip``
    (fewer than the target is kept as-is if the pool is smaller).

    Every stage iterates over a *sorted* candidate order before drawing
    from the RNG, so the result is reproducible given the same ``records``
    and ``seed`` regardless of input ordering or dict iteration order.

    Returns a dict with ``pair_ids`` (the frozen selection, sorted),
    ``seed``, ``table_cap``, ``target_adjacent``, ``target_skip``, and a
    ``stats`` breakdown at each stage, for the audit file.
    """

    rng = random.Random(seed)

    by_group: dict[tuple[str, str], dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: {"adjacent": [], "skip": []})
    for record in records:
        rule = record["construction_rule"]
        if rule not in (ADJACENT_RULE, NONADJACENT_RULE):
            continue
        kind = "adjacent" if rule == ADJACENT_RULE else "skip"
        by_group[(record["table_id"], record["metric_a"])][kind].append(record)

    stage1: list[Mapping[str, Any]] = []
    for key in sorted(by_group):
        group = by_group[key]
        for kind in ("adjacent", "skip"):
            candidates = sorted(group[kind], key=lambda r: r["pair_id"])
            if candidates:
                stage1.append(rng.choice(candidates))

    by_table: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in stage1:
        by_table[record["table_id"]].append(record)

    stage2: list[Mapping[str, Any]] = []
    for table_id in sorted(by_table):
        candidates = sorted(by_table[table_id], key=lambda r: r["pair_id"])
        if len(candidates) > table_cap:
            candidates = rng.sample(candidates, table_cap)
        stage2.extend(candidates)

    adjacent_pool = sorted((r for r in stage2 if r["construction_rule"] == ADJACENT_RULE), key=lambda r: r["pair_id"])
    skip_pool = sorted((r for r in stage2 if r["construction_rule"] == NONADJACENT_RULE), key=lambda r: r["pair_id"])
    final_adjacent = rng.sample(adjacent_pool, min(target_adjacent, len(adjacent_pool)))
    final_skip = rng.sample(skip_pool, min(target_skip, len(skip_pool)))

    pair_ids = sorted(r["pair_id"] for r in final_adjacent + final_skip)
    return {
        "selection_protocol_version": SELECTION_PROTOCOL_VERSION,
        "seed": seed,
        "table_cap": table_cap,
        "target_adjacent": target_adjacent,
        "target_skip": target_skip,
        "pair_ids": pair_ids,
        "stats": {
            "input_pairs": len(records),
            "stage1_per_group_capped": len(stage1),
            "stage1_adjacent": sum(1 for r in stage1 if r["construction_rule"] == ADJACENT_RULE),
            "stage1_skip": sum(1 for r in stage1 if r["construction_rule"] == NONADJACENT_RULE),
            "stage2_per_table_capped": len(stage2),
            "stage2_adjacent": len(adjacent_pool),
            "stage2_skip": len(skip_pool),
            "final_adjacent": len(final_adjacent),
            "final_skip": len(final_skip),
            "final_total": len(pair_ids),
            "unique_tables": len({r["table_id"] for r in final_adjacent + final_skip}),
        },
    }
