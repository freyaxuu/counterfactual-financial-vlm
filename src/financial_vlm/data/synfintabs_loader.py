"""Canonical clean-record loader helpers for SynFinTabs tables."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from financial_vlm.data.canonical_schema import (
    normalise_bbox,
    normalise_numeric_value,
    validate_clean_record,
    validate_group_record,
)
from financial_vlm.data.synfintabs_pilot import (
    CellRef,
    EvidencePacket,
    LocatedQuestion,
    VariantPlan,
    YEAR_RE,
    build_evidence_packet,
    plan_to_json,
    rendered_cells_for_plan,
)
from financial_vlm.data.synfintabs_training import stable_group_seed


COUNTERFACTUAL_TYPES = ["target_value_replace", "header_swap", "irrelevant_cell_replace"]
STYLE_KEYS = ("template_family", "theme", "style", "template")
EXPECTED_BEHAVIOR_BY_VARIANT = {
    "clean": "baseline",
    "target_value_replacement": "answer_changes",
    "header_address_swap": "answer_changes",
    "irrelevant_value_replacement": "answer_invariant",
}


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_")[:120] or "unknown"


def syn_cell_id(cell: CellRef) -> str:
    return f"cell_{cell.row_idx}_{cell.col_idx}"


def template_family_for_table(table: Mapping[str, Any]) -> str:
    for key in STYLE_KEYS:
        value = table.get(key)
        if value is not None and str(value).strip():
            return safe_slug(str(value))
    return "unknown"


def document_id_for_table(table: Mapping[str, Any], source_index: int) -> str:
    source_table_id = str(table.get("id") or "").strip()
    if source_table_id:
        return safe_slug(source_table_id)
    return f"syn_{source_index:06d}"


def question_slug(located: LocatedQuestion, question_index: int) -> str:
    if located.question_id:
        return safe_slug(located.question_id)
    return f"q{question_index:02d}"


def unit_and_scale_from_cells(unit_cells: Sequence[CellRef]) -> tuple[str | None, str | None]:
    if not unit_cells:
        return None, None
    text = " ".join(cell.text for cell in unit_cells).lower()
    unit = None
    if "$" in text or "usd" in text:
        unit = "USD"
    elif "£" in text or "gbp" in text:
        unit = "GBP"
    elif "€" in text or "eur" in text:
        unit = "EUR"
    elif "%" in text or "percent" in text:
        unit = "%"

    scale = None
    if "million" in text or re.search(r"\bmn\b", text):
        scale = "million"
    elif "thousand" in text or "000" in text:
        scale = "thousand"
    elif re.search(r"\bbn\b", text) or "billion" in text:
        scale = "billion"
    return unit, scale


def evidence_unit_for_cell(
    cell: CellRef,
    *,
    role: str,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    return {
        "id": syn_cell_id(cell),
        "type": "cell",
        "text": cell.text or " ",
        "bbox_normalised": normalise_bbox(cell.bbox, image_width, image_height),
        "role": role,
        "metadata": {
            "row_idx": cell.row_idx,
            "col_idx": cell.col_idx,
            "label": cell.label,
            "word_indices": list(cell.word_indices),
        },
    }


def build_synfintabs_clean_record(
    *,
    table: Mapping[str, Any],
    source_index: int,
    question_index: int,
    split: str,
    image_path: str,
    image_size: tuple[int, int],
    located: LocatedQuestion,
    evidence_packet: EvidencePacket,
) -> dict[str, Any]:
    image_width, image_height = image_size
    group_id = f"syn_{source_index:06d}_{question_slug(located, question_index)}"
    units_by_id, row_header_ids, column_header_ids, unit_region_ids, _parent_header_ids = evidence_packet_ids(
        evidence_packet, image_size
    )

    metric = evidence_packet.row_headers[-1].text if evidence_packet.row_headers else None
    if evidence_packet.column_headers:
        period_cell = next(
            (cell for cell in reversed(evidence_packet.column_headers) if YEAR_RE.search(cell.text)),
            evidence_packet.column_headers[-1],
        )
        period = period_cell.text
    else:
        period = None
    unit, scale = unit_and_scale_from_cells(evidence_packet.unit_cells)

    record = {
        "group_id": group_id,
        "source": "synfintabs",
        "split": split,
        "document_id": document_id_for_table(table, source_index),
        "template_family": template_family_for_table(table),
        "image_path": image_path,
        "question": located.question,
        "answer": {
            "raw": located.answer,
            "normalised_value": normalise_numeric_value(located.answer),
            "unit": unit,
            "scale": scale,
            "metric": metric,
            "period": period,
        },
        "evidence": {
            "type": "cell",
            "target_id": syn_cell_id(evidence_packet.value_cell),
            "bbox_normalised": normalise_bbox(evidence_packet.value_cell.bbox, image_width, image_height),
            "row_header_ids": row_header_ids,
            "column_header_ids": column_header_ids,
            "unit_region_ids": unit_region_ids,
        },
        "evidence_units": list(units_by_id.values()),
        "counterfactual_policy": {
            "allowed_types": COUNTERFACTUAL_TYPES,
        },
    }
    validate_clean_record(record)
    return record


def evidence_packet_ids(evidence_packet: EvidencePacket, image_size: tuple[int, int]) -> tuple[
    dict[str, dict[str, Any]], list[str], list[str], list[str], list[str]
]:
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
                syn_cell_id(cell),
                evidence_unit_for_cell(cell, role=role, image_width=image_width, image_height=image_height),
            )
    row_header_ids = [syn_cell_id(cell) for cell in evidence_packet.row_headers]
    column_header_ids = [syn_cell_id(cell) for cell in evidence_packet.column_headers]
    unit_region_ids = [syn_cell_id(cell) for cell in evidence_packet.unit_cells]
    parent_header_ids = [syn_cell_id(cell) for cell in evidence_packet.spanning_headers]
    return units_by_id, row_header_ids, column_header_ids, unit_region_ids, parent_header_ids


def build_synfintabs_variant_payload(
    *,
    plan: VariantPlan,
    cells: Sequence[CellRef],
    group_id: str,
    image_path: str,
    image_size: tuple[int, int],
    renderer_seed: int,
    expected_behavior: str,
) -> dict[str, Any]:
    image_width, image_height = image_size
    rendered_cells = rendered_cells_for_plan(cells, plan)
    evidence_packet = build_evidence_packet(rendered_cells, plan.evidence_cell)

    units_by_id, row_header_ids, column_header_ids, unit_region_ids, parent_header_ids = evidence_packet_ids(
        evidence_packet, image_size
    )

    metric = evidence_packet.row_headers[-1].text if evidence_packet.row_headers else None
    if evidence_packet.column_headers:
        period_cell = next(
            (cell for cell in reversed(evidence_packet.column_headers) if YEAR_RE.search(cell.text)),
            evidence_packet.column_headers[-1],
        )
        period = period_cell.text
    else:
        period = None
    unit, scale = unit_and_scale_from_cells(evidence_packet.unit_cells)

    target_id = syn_cell_id(evidence_packet.value_cell)
    evidence = {
        "type": "cell",
        "target_id": target_id,
        "bbox_normalised": normalise_bbox(evidence_packet.value_cell.bbox, image_width, image_height),
        "row_header_ids": row_header_ids,
        "column_header_ids": column_header_ids,
        "unit_region_ids": unit_region_ids,
    }

    plan_json = plan_to_json(plan)

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
        "evidence": evidence,
        "evidence_units": list(units_by_id.values()),
        "gold_evidence_ids": {
            "target_value": [target_id],
            "row_headers": row_header_ids,
            "column_headers": column_header_ids,
            "parent_headers": parent_header_ids,
            "unit_cells": unit_region_ids,
        },
        "changed_cell_ids": plan_json["box_mapping"]["changed_source_cell_ids"],
        "expected_behavior": expected_behavior,
        "validation_status": plan.validation_status,
        "validation_notes": list(plan.validation_notes),
    }


def build_synfintabs_group_record(
    *,
    table: Mapping[str, Any],
    source_index: int,
    question_index: int,
    split: str,
    located: LocatedQuestion,
    cells: Sequence[CellRef],
    plans: Sequence[VariantPlan],
    variant_image_paths: Mapping[str, str],
    variant_image_sizes: Mapping[str, tuple[int, int]],
    seed: int,
) -> dict[str, Any]:
    """Build one nested group record covering all counterfactual variants.

    Unlike ``build_synfintabs_clean_record`` (one flat clean-only record),
    this writes a single JSONL line per group with a ``variants`` mapping
    keyed by variant name, each carrying its own image, renderer seed, and
    evidence -- notably ``header_address_swap``'s evidence correctly points
    at the peer cell, not the clean answer cell.
    """

    group_id = f"syn_{source_index:06d}_{question_slug(located, question_index)}"
    renderer_seed_base = stable_group_seed(seed, group_id)

    variants_payload: dict[str, Any] = {}
    for plan in plans:
        variant_seed = stable_group_seed(seed, f"{group_id}:{plan.variant}")
        variants_payload[plan.variant] = build_synfintabs_variant_payload(
            plan=plan,
            cells=cells,
            group_id=group_id,
            image_path=variant_image_paths[plan.variant],
            image_size=variant_image_sizes[plan.variant],
            renderer_seed=variant_seed,
            expected_behavior=EXPECTED_BEHAVIOR_BY_VARIANT[plan.variant],
        )

    record = {
        "group_id": group_id,
        "source": "synfintabs",
        "split": split,
        "document_id": document_id_for_table(table, source_index),
        "template_family": template_family_for_table(table),
        "question": located.question,
        "counterfactual_policy": {"allowed_types": COUNTERFACTUAL_TYPES},
        "renderer_seed_base": renderer_seed_base,
        "variants": variants_payload,
    }
    validate_group_record(record)
    return record
