"""Canonical clean-record loader helpers for the private company KPIs
surface, for the one-time, narrowly-scoped exploratory training exception
documented in `AGENTS.md`'s "Company data" section.

Modeled directly on `financial_vlm.data.tatdqa_loader.build_tatdqa_clean_record`
-- the closest existing analog (flat, single-variant, no counterfactual
policy), not TAT-QA's group/CF-shaped builder. Differs from TAT-DQA's
version only in `source`/`template_family`/`evidence.type` and in having
exactly one `evidence_units` entry (the target field's own bbox) -- company
KPIs instances carry one bbox per field, not TAT-QA/SynFinTabs' full
row/column header cell graph, and `validate_clean_record` does not require
row/column header units (only a non-empty `evidence_units` list and a valid
`target_id` reference).

Pure functions -- no private-package import, testable with synthetic
fixtures (`tests/unit/test_company_loader.py`), per `AGENTS.md`'s "Make
local tests independent of evolution-ai-datasets" / "Use synthetic
fixtures only in unit tests" rules.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from financial_vlm.data.canonical_schema import normalise_bbox, normalise_numeric_value, validate_clean_record

TEMPLATE_FAMILY = "kpi_table"


@dataclass(frozen=True)
class CompanyTrainCandidate:
    document_id: str
    page_id: str
    instance_index: int
    metric: str
    period_text: str
    gold_answer: str
    question: str
    bbox: tuple[int, int, int, int]


def cap_per_document(
    candidates: Sequence[CompanyTrainCandidate], max_per_document: int, shuffle_seed: int
) -> list[CompanyTrainCandidate]:
    """Same deterministic-shuffle-then-greedy-cap pattern as
    `kpi_representative_facts.stratified_cap_facts` /
    `kpi_question_pairs.stratified_cap_pairs`, keyed by `document_id` --
    the dimension actually skewed in `company_train_v1`'s 5-document
    exception pool (2 of 5 documents otherwise supply 71% of accepted
    facts, per the diagnosis this exception's construction was based on).
    Fixed-seed shuffle before greedy selection for determinism and to
    avoid biasing toward whichever document happens to be iterated
    first."""

    order = list(candidates)
    random.Random(shuffle_seed).shuffle(order)
    doc_counts: Counter[str] = Counter()
    selected: list[CompanyTrainCandidate] = []
    for candidate in order:
        if doc_counts[candidate.document_id] >= max_per_document:
            continue
        selected.append(candidate)
        doc_counts[candidate.document_id] += 1
    return selected


def adapter_bbox_to_xyxy(bbox: Sequence[int]) -> tuple[int, int, int, int]:
    """The private-package adapter's `BoundingBox` is `(top, left, bottom,
    right)` (see `evolution_ai_datasets_adapter.py`'s own docstring) --
    `canonical_schema.normalise_bbox` expects `(x0, y0, x1, y1)` i.e.
    `(left, top, right, bottom)`. Converting here, once, in one named
    function, rather than inline at each call site, so the reordering is
    not silently duplicated or forgotten."""

    top, left, bottom, right = bbox
    return (left, top, right, bottom)


def company_evidence_id(document_id: str, page_id: str, instance_index: int, metric: str) -> str:
    raw = f"{document_id}|{page_id}|{instance_index}|{metric}"
    return "field_" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_company_clean_record(
    *,
    document_id: str,
    page_id: str,
    instance_index: int,
    split: str,
    image_path: str,
    image_size: tuple[int, int],
    metric: str,
    period_text: str,
    gold_answer: str,
    question: str,
    bbox: Sequence[int],
    group_index: int,
) -> dict[str, Any]:
    """Build one `canonical_clean_v1` record for a single (metric, period)
    fact from the company `KPIs` surface. `bbox` is the adapter's raw
    `(top, left, bottom, right)` field bbox; `image_size` is `(width,
    height)` read from the actual page image on disk (company data has no
    pre-recorded image dimensions the way TAT-QA/SynFinTabs' rendering
    pipelines do)."""

    image_width, image_height = image_size
    bbox_xyxy = adapter_bbox_to_xyxy(bbox)
    bbox_normalised = normalise_bbox(bbox_xyxy, image_width, image_height)
    evidence_id = company_evidence_id(document_id, page_id, instance_index, metric)

    record = {
        "group_id": f"company_{document_id}_{page_id}_{instance_index}_{metric}_{group_index:06d}",
        "source": "company",
        "split": split,
        "document_id": document_id,
        "template_family": TEMPLATE_FAMILY,
        "image_path": image_path,
        "question": question,
        "answer": {
            "raw": gold_answer,
            "normalised_value": normalise_numeric_value(gold_answer),
            "unit": None,
            "scale": None,
            "metric": metric,
            "period": period_text,
        },
        "evidence": {
            "type": "region",
            "target_id": evidence_id,
            "bbox_normalised": bbox_normalised,
            "row_header_ids": [],
            "column_header_ids": [],
            "unit_region_ids": [],
        },
        "evidence_units": [
            {
                "id": evidence_id,
                "type": "region",
                "text": gold_answer,
                "bbox_normalised": bbox_normalised,
                "role": "target",
                "metadata": {
                    "page_id": page_id,
                    "instance_index": instance_index,
                },
            }
        ],
        "counterfactual_policy": {
            "allowed_types": [],
        },
    }
    validate_clean_record(record)
    return record
