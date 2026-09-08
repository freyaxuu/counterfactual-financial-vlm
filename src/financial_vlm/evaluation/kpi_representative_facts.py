"""Individual (non-paired) facts for the frozen `company_representative_v1`
benchmark -- ordinary-reading / domain-transfer diagnostic, distinct from
`company_confusion_v1`'s confusion-pair reliability diagnostic (chat,
2026-08-13). Deliberately restricted to the `KPIs` surface, the same source
`company_confusion_v1` uses, so a difference between the two sets' Individual
Accuracy is attributable to the confusion-pair structure itself, not to a
different field surface.

Pure functions/dataclasses -- no private-package dependency, testable with
synthetic fixtures.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class RepresentativeFactRecord:
    fact_id: str
    document_id: str
    page_id: str
    instance_id: int
    company_key: str
    metric: str
    bbox: tuple[int, int, int, int]
    period_text: str
    question: str
    gold_answer: str
    ocr_outcome: str
    verification_status: str  # "automated_pass" -- see kpi_question_pairs module docstring


def stratified_cap_facts(
    facts: Sequence[RepresentativeFactRecord],
    max_per_page: int,
    max_per_company: int,
    max_per_metric: int | None = None,
    shuffle_seed: int = 20260813,
) -> list[RepresentativeFactRecord]:
    """Greedily select a subset of `facts` respecting independent per-page/
    per-company/(optionally) per-metric caps -- same rationale and algorithm
    as `kpi_question_pairs.stratified_cap_pairs`: the raw KPIs surface is
    heavily concentrated (a handful of documents/pages/metrics dominate),
    and an uncapped "representative" sample would just reproduce that
    skew, not represent it fairly. Fixed-seed shuffle before greedy
    selection for determinism and to avoid biasing toward whichever
    document happens to be iterated first.
    """

    order = list(facts)
    random.Random(shuffle_seed).shuffle(order)

    page_counts: Counter[tuple[str, str]] = Counter()
    company_counts: Counter[str] = Counter()
    metric_counts: Counter[str] = Counter()
    selected: list[RepresentativeFactRecord] = []

    for fact in order:
        page_key = (fact.document_id, fact.page_id)
        if page_counts[page_key] >= max_per_page:
            continue
        if company_counts[fact.company_key] >= max_per_company:
            continue
        if max_per_metric is not None and metric_counts[fact.metric] >= max_per_metric:
            continue

        selected.append(fact)
        page_counts[page_key] += 1
        company_counts[fact.company_key] += 1
        metric_counts[fact.metric] += 1

    return selected
