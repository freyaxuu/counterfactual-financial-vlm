"""Canonical clean-record loader helpers for TAT-DQA pages."""

from __future__ import annotations

import re
from typing import Any, Mapping

from financial_vlm.data.canonical_schema import normalise_bbox, normalise_numeric_value, validate_clean_record
from financial_vlm.data.tatdqa_pilot import MetricPeriodMatch, TatDqaEvidence, scalar_answer_text


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_")[:120] or "unknown"


def tatdqa_evidence_id(evidence: TatDqaEvidence) -> str:
    start, end = evidence.span
    block_prefix = safe_slug(evidence.block.uuid)[:8] or "block"
    return f"region_{evidence.block.page_index}_{block_prefix}_{evidence.span_mode}_{start}_{end}"


def build_tatdqa_clean_record(
    *,
    doc_uid: str,
    split: str,
    image_path: str,
    image_size: tuple[int, int],
    question: Mapping[str, Any],
    question_index: int,
    accepted_index: int,
    evidence: TatDqaEvidence,
    metric_period_match: MetricPeriodMatch | None = None,
) -> dict[str, Any]:
    raw_answer_value = question.get("answer")
    answer_raw = scalar_answer_text(raw_answer_value) or ""
    question_id = str(question.get("uid") or question.get("id") or f"q_{question_index:06d}")
    evidence_id = tatdqa_evidence_id(evidence)
    image_width, image_height = image_size
    metric = metric_period_match.metric_candidate if metric_period_match is not None else None
    period = (
        metric_period_match.matched_periods[0]
        if metric_period_match is not None and metric_period_match.matched_periods
        else None
    )
    record = {
        "group_id": f"tatdqa_{safe_slug(doc_uid)}_{safe_slug(question_id)}_{accepted_index:06d}",
        "source": "tatdqa",
        "split": split,
        "document_id": doc_uid,
        "template_family": "public_page",
        "image_path": image_path,
        "question": str(question.get("question") or ""),
        "answer": {
            "raw": answer_raw,
            "normalised_value": normalise_numeric_value(answer_raw),
            "unit": None,
            "scale": str(question.get("scale") or "").strip() or None,
            "metric": metric,
            "period": period,
        },
        "evidence": {
            "type": "region",
            "target_id": evidence_id,
            "bbox_normalised": normalise_bbox(evidence.bbox, image_width, image_height),
            "row_header_ids": [],
            "column_header_ids": [],
            "unit_region_ids": [],
        },
        "evidence_units": [
            {
                "id": evidence_id,
                "type": "region",
                "text": evidence.text or " ",
                "bbox_normalised": normalise_bbox(evidence.bbox, image_width, image_height),
                "role": "target",
                "metadata": {
                    "page_index": evidence.block.page_index,
                    "block_uuid": evidence.block.uuid,
                    "block_order": evidence.block.order,
                    "span": list(evidence.span),
                    "span_mode": evidence.span_mode,
                    "word_indices": list(evidence.word_indices),
                    "metric_period_match": (
                        {
                            "accepted": metric_period_match.accepted,
                            "reason": metric_period_match.reason,
                            "metric_candidate": metric_period_match.metric_candidate,
                            "metric_tokens": list(metric_period_match.metric_tokens),
                            "matched_metric_tokens": list(metric_period_match.matched_metric_tokens),
                            "metric_overlap_ratio": metric_period_match.metric_overlap_ratio,
                            "period_candidates": list(metric_period_match.period_candidates),
                            "matched_periods": list(metric_period_match.matched_periods),
                            "context_block_ids": list(metric_period_match.context_block_ids),
                        }
                        if metric_period_match is not None
                        else None
                    ),
                },
            }
        ],
        "counterfactual_policy": {
            "allowed_types": [],
        },
    }
    validate_clean_record(record)
    return record
