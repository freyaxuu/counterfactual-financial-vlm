"""Turn automated-filter-passing period/basis-pair candidates
(`financial_vlm.evaluation.kpi_confusion_pairs.PairCandidate`) into actual
question-pair records, per `docs/company-benchmark-diagnosis-report.md`
section 13's schema.

Per the repository owner's decision (chat, 2026-08-12; see the diagnosis
report sections 9/14/17), no human verification pass is applied before
freezing this benchmark. Instead, every fact entering a pair must pass the
calibrated OCR-proximity proxy check
(`financial_vlm.evaluation.ocr_quality_proxy`, calibrated across two rounds
against real examples -- report sections 0.1/0.2) on the specific metric
field being asked about. This is a materially weaker guarantee than human
review -- the calibration rounds found 0 confirmed genuine annotation
defects in 11 checked examples, which is reassuring but not proof of a zero
error rate -- and that tradeoff must be stated as a limitation wherever this
benchmark's results are reported, not silently assumed away.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from financial_vlm.evaluation.kpi_confusion_pairs import normalize_month, normalize_year

_MONTH_NAMES: dict[str, str] = {
    "01": "January", "02": "February", "03": "March", "04": "April",
    "05": "May", "06": "June", "07": "July", "08": "August",
    "09": "September", "10": "October", "11": "November", "12": "December",
}  # fmt: skip

# A full date crammed into a single field ("31.12.20", "31/12/2020") --
# confirmed real raw-vocabulary pattern, 2026-08-12.
_FULL_DATE_RE = re.compile(r"^(\d{1,2})[./-](\d{1,2})[./-](\d{2,4})$")
# A month/year pair crammed into a single field ("12/20", "12.20") without a
# day component -- same confirmed pattern, distinguished from the full-date
# case by having only two numeric groups.
_MONTH_YEAR_RE = re.compile(r"^(\d{1,2})[./-](\d{2,4})$")


def _format_month_year(month_raw: str, year_raw: str) -> str | None:
    month_norm = normalize_month(month_raw)
    year_norm = normalize_year(year_raw)
    if month_norm is None or year_norm is None:
        return None
    return f"{_MONTH_NAMES[month_norm]} {year_norm}"


def _try_parse_embedded_date(text: str) -> str | None:
    """If `text` (e.g. a raw `period` value) is itself a full date or a
    month/year pair in one field, reformat it as "Month YYYY". Returns
    `None` if it doesn't match either pattern -- callers fall back to
    treating `text` as an ordinary label, not a date."""

    text = text.strip()
    match = _FULL_DATE_RE.match(text)
    if match:
        _day, month, year = match.groups()
        return _format_month_year(month, year)
    match = _MONTH_YEAR_RE.match(text)
    if match:
        month, year = match.groups()
        return _format_month_year(month, year)
    return None


# Which OCR-proxy outcomes count as "this side of the pair is confirmed."
# Deliberately conservative: "mismatch_low_ocr_confidence" is excluded even
# though calibration (report section 0.2) found both examples of it were
# genuine OCR failures, not annotation errors -- with only 2 examples
# checked, treating "OCR couldn't confirm this" as equivalent to "OCR
# confirmed this" would be a bigger assumption than this benchmark's
# verification story can currently support. "annotation_unparseable" and
# "no_ocr_nearby" are excluded for the same reason: absence of positive
# confirmation is not confirmation.
ACCEPTED_OCR_OUTCOMES = frozenset({"match", "match_word_set", "match_scale_or_sign"})


def period_label_text(period: str | None, month: str | None, year: str | None) -> str:
    """Human-readable period label -- prefers a normalized "Month YYYY" form
    over raw concatenation of whatever format the source table happened to
    use. Confirmed real issue, 2026-08-12: naive `" ".join(...)` produced
    unreadable question text like "What was sales in 12 20?" instead of
    "...in December 2020?", and some raw values cram a full date or a
    month/year pair into a single field ("31.12.20", "12/20") rather than
    splitting across the `period`/`month`/`year` fields at all.

    Falls back to raw, unnormalized text when nothing parses cleanly --
    never fabricates a date component it can't confirm.
    """

    if period is not None:
        embedded_date = _try_parse_embedded_date(period)
        if embedded_date is not None:
            return embedded_date

    date_part = _format_month_year(month, year) if (month is not None and year is not None) else None
    if date_part is None:
        year_norm = normalize_year(year) if year is not None else None
        date_part = year_norm

    if period is not None and date_part is not None:
        return f"{period} {date_part}"
    if date_part is not None:
        return date_part
    if period is not None:
        return period

    parts = [p for p in (month, year) if p]
    return " ".join(parts) if parts else "unspecified period"


def build_question_text(metric: str, period_text: str) -> str:
    return f"What was {metric} in {period_text}?"


def metric_pair_automated_pass(outcome_a: str, outcome_b: str) -> bool:
    """Both sides of a pair's specific metric value must independently pass
    the OCR-proximity proxy check -- see module docstring for why this,
    not human review, is this benchmark's verification gate, and why the
    accepted-outcome set is conservative."""

    return outcome_a in ACCEPTED_OCR_OUTCOMES and outcome_b in ACCEPTED_OCR_OUTCOMES


@dataclass(frozen=True)
class QuestionPairRecord:
    pair_id: str
    pair_type: str  # "period" | "basis"
    document_id: str
    company_key: str
    metric: str
    page_a: str
    page_b: str
    instance_a: int
    instance_b: int
    # (top, left, bottom, right) pixel coordinates on the page image -- not
    # sensitive on their own (no company name or figure), and necessary to
    # re-locate a specific emitted pair on the source page for spot-checking
    # without needing to re-run the whole pairing pipeline.
    bbox_a: tuple[int, int, int, int]
    bbox_b: tuple[int, int, int, int]
    period_text_a: str
    period_text_b: str
    question_a: str
    question_b: str
    gold_answer_a: str
    gold_answer_b: str
    ocr_outcome_a: str
    ocr_outcome_b: str
    verification_status: str  # "automated_pass" -- see module docstring


def stratified_cap_pairs(
    pairs: Sequence[QuestionPairRecord],
    max_per_page: int,
    max_per_company: int,
    max_per_metric: int | None = None,
    shuffle_seed: int = 20260812,
) -> list[QuestionPairRecord]:
    """Greedily select a subset of `pairs` respecting three independent caps
    at once, so no single page/company/metric can dominate the final set --
    confirmed necessary, 2026-08-12: the unfiltered set had one page
    contributing 246 pairs, one document 948, and `sales` alone 41% of
    everything.

    Processes pairs in a fixed-seed shuffled order (not file order, which
    tends to cluster by document/page and would bias which pairs survive
    the caps toward whatever happened to be iterated first) and keeps a
    pair only if accepting it would not exceed any of the three caps.
    `max_per_metric` is optional -- pass `None` to cap only by page and
    company. Deterministic for a given `shuffle_seed`.
    """

    order = list(pairs)
    random.Random(shuffle_seed).shuffle(order)

    page_counts: Counter[tuple[str, str]] = Counter()
    company_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    selected: list[QuestionPairRecord] = []

    for pair in order:
        page_key = (pair.document_id, pair.page_a)
        if page_counts[page_key] >= max_per_page:
            continue
        if company_counts[pair.company_key] >= max_per_company:
            continue
        if max_per_metric is not None and metric_counts[pair.metric] >= max_per_metric:
            continue

        selected.append(pair)
        page_counts[page_key] += 1
        company_counts[pair.company_key] += 1
        metric_counts[pair.metric] += 1

    return selected
