"""Canonical group-record loader helpers for source-rerendered TAT-QA groups.

TAT-QA groups (unlike TAT-DQA, which only ships a clean record) come with a
full 4-variant counterfactual set from ``tatqa_source_rerender``, so the
correct canonical target is ``validate_group_record`` (one JSONL line per
question, nested by variant) rather than the flat ``validate_clean_record``
used for TAT-DQA's clean-only manifest. See
``docs/tatqa-source-rerender-cf-feasibility.md`` for the feasibility check
this builds on.

This module intentionally has no PIL dependency: callers render each variant
via ``tatqa_source_rerender.render_variant_image`` (which needs Pillow),
save the image, and pass the resulting bboxes/image size/rendered cells in.
Per ``AGENTS.md`` dataset roles, TAT-QA is public real-world data and must
only be used for evaluation, never training or hyperparameter selection.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from financial_vlm.data.canonical_schema import normalise_bbox, normalise_numeric_value, validate_group_record
from financial_vlm.data.synfintabs_loader import unit_and_scale_from_cells
from financial_vlm.data.synfintabs_training import stable_group_seed
from financial_vlm.data.tatqa_source_rerender import Bbox, EvidencePacket, GridCell, VariantPlan, YEAR_RE, build_evidence_packet


COUNTERFACTUAL_TYPES = ["target_value_replace", "header_swap", "irrelevant_cell_replace"]
TEMPLATE_FAMILY = "public_table"
EXPECTED_BEHAVIOR_BY_VARIANT = {
    "clean": "baseline",
    "target_value_replacement": "answer_changes",
    "header_address_swap": "answer_changes",
    "irrelevant_value_replacement": "answer_invariant",
}


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_")[:120] or "unknown"


def group_id_for(table_uid: str, question_uid: str, question_index: int) -> str:
    question_slug = safe_slug(question_uid) if question_uid else f"q{question_index:04d}"
    return f"tatqa_{safe_slug(table_uid)}_{question_slug}"


def evidence_unit_for_cell(
    cell: GridCell,
    bbox: Bbox,
    *,
    role: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    return {
        "id": cell.cell_id,
        "type": "cell",
        "text": cell.text or " ",
        "bbox_normalised": normalise_bbox(bbox, image_width, image_height),
        "role": role,
        "metadata": {"row_idx": cell.row_idx, "col_idx": cell.col_idx},
    }


def evidence_packet_ids(
    evidence_packet: EvidencePacket,
    bboxes: Mapping[str, Bbox],
    image_size: tuple[int, int],
) -> tuple[dict[str, dict[str, Any]], list[str], list[str], list[str], list[str]]:
    image_width, image_height = image_size
    units_by_id: dict[str, dict[str, Any]] = {}
    for role, group_cells in (
        ("spanning_header", evidence_packet.spanning_headers),
        ("column_header", evidence_packet.column_headers),
        ("unit_region", evidence_packet.unit_cells),
        ("row_header", evidence_packet.row_headers),
        ("target", (evidence_packet.value_cell,)),
    ):
        for cell in group_cells:
            units_by_id.setdefault(
                cell.cell_id,
                evidence_unit_for_cell(
                    cell, bboxes[cell.cell_id], role=role, image_width=image_width, image_height=image_height
                ),
            )
    row_header_ids = [cell.cell_id for cell in evidence_packet.row_headers]
    column_header_ids = [cell.cell_id for cell in evidence_packet.column_headers]
    unit_region_ids = [cell.cell_id for cell in evidence_packet.unit_cells]
    parent_header_ids = [cell.cell_id for cell in evidence_packet.spanning_headers]
    return units_by_id, row_header_ids, column_header_ids, unit_region_ids, parent_header_ids


def metric_and_period(evidence_packet: EvidencePacket) -> tuple[str | None, str | None]:
    metric = evidence_packet.row_headers[-1].text if evidence_packet.row_headers else None
    if evidence_packet.column_headers:
        period_cell = next(
            (cell for cell in reversed(evidence_packet.column_headers) if YEAR_RE.search(cell.text)),
            evidence_packet.column_headers[-1],
        )
        period = period_cell.text
    else:
        period = None
    return metric, period


def build_tatqa_variant_payload(
    *,
    plan: VariantPlan,
    rendered_cells: Sequence[GridCell],
    bboxes: Mapping[str, Bbox],
    image_size: tuple[int, int],
    group_id: str,
    image_path: str,
    renderer_seed: int,
    expected_behavior: str,
) -> dict[str, Any]:
    image_width, image_height = image_size
    evidence_packet = build_evidence_packet(rendered_cells, plan.evidence_cell)
    units_by_id, row_header_ids, column_header_ids, unit_region_ids, parent_header_ids = evidence_packet_ids(
        evidence_packet, bboxes, image_size
    )
    metric, period = metric_and_period(evidence_packet)
    unit, scale = unit_and_scale_from_cells(evidence_packet.unit_cells)

    target_id = evidence_packet.value_cell.cell_id
    changed_cell_ids = [patch.cell.cell_id for patch in plan.patches]

    return {
        "variant": plan.variant,
        "intervention_type": plan.intervention_type,
        "image_id": f"{group_id}__{plan.variant}",
        "image_path": image_path,
        "renderer_seed": renderer_seed,
        "answer": {
            "raw": plan.answer,
            "normalised_value": normalise_numeric_value(plan.answer),
            "unit": unit,
            "scale": scale,
            "metric": metric,
            "period": period,
        },
        "evidence": {
            "type": "cell",
            "target_id": target_id,
            "bbox_normalised": normalise_bbox(bboxes[target_id], image_width, image_height),
            "row_header_ids": row_header_ids,
            "column_header_ids": column_header_ids,
            "unit_region_ids": unit_region_ids,
        },
        "evidence_units": list(units_by_id.values()),
        "gold_evidence_ids": {
            "target_value": [target_id],
            "row_headers": row_header_ids,
            "column_headers": column_header_ids,
            "parent_headers": parent_header_ids,
            "unit_cells": unit_region_ids,
        },
        "changed_cell_ids": changed_cell_ids,
        "expected_behavior": expected_behavior,
        "validation_status": plan.validation_status,
        "validation_notes": list(plan.validation_notes),
    }


def build_tatqa_group_record(
    *,
    table_uid: str,
    question_uid: str,
    question_index: int,
    question: str,
    split: str,
    plans: Sequence[VariantPlan],
    variant_rendered_cells: Mapping[str, Sequence[GridCell]],
    variant_bboxes: Mapping[str, Mapping[str, Bbox]],
    variant_image_sizes: Mapping[str, tuple[int, int]],
    variant_image_paths: Mapping[str, str],
    seed: int,
) -> dict[str, Any]:
    """Build one nested group record covering clean + all 3 CF variants.

    Callers must render each variant first (``tatqa_source_rerender.render_variant_image``)
    and pass in its cells/bboxes/size/path per variant name -- this module has
    no PIL dependency of its own.
    """

    group_id = group_id_for(table_uid, question_uid, question_index)
    renderer_seed_base = stable_group_seed(seed, group_id)

    variants_payload: dict[str, Any] = {}
    for plan in plans:
        variant_seed = stable_group_seed(seed, f"{group_id}:{plan.variant}")
        variants_payload[plan.variant] = build_tatqa_variant_payload(
            plan=plan,
            rendered_cells=variant_rendered_cells[plan.variant],
            bboxes=variant_bboxes[plan.variant],
            image_size=variant_image_sizes[plan.variant],
            group_id=group_id,
            image_path=variant_image_paths[plan.variant],
            renderer_seed=variant_seed,
            expected_behavior=EXPECTED_BEHAVIOR_BY_VARIANT[plan.variant],
        )

    record = {
        "group_id": group_id,
        "source": "tatqa",
        "split": split,
        "document_id": safe_slug(table_uid),
        "template_family": TEMPLATE_FAMILY,
        "question": question,
        "counterfactual_policy": {"allowed_types": COUNTERFACTUAL_TYPES},
        "renderer_seed_base": renderer_seed_base,
        "variants": variants_payload,
    }
    validate_group_record(record)
    return record
