"""Canonical JSONL schema helpers for grounded financial QA records."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence


CLEAN_RECORD_KEYS = {
    "group_id",
    "source",
    "split",
    "document_id",
    "template_family",
    "image_path",
    "question",
    "answer",
    "evidence",
    "evidence_units",
    "counterfactual_policy",
}
ANSWER_KEYS = {"raw", "normalised_value", "unit", "scale", "metric", "period"}
EVIDENCE_KEYS = {
    "type",
    "target_id",
    "bbox_normalised",
    "row_header_ids",
    "column_header_ids",
    "unit_region_ids",
}
EVIDENCE_UNIT_KEYS = {"id", "type", "text", "bbox_normalised", "role", "metadata"}
COUNTERFACTUAL_POLICY_KEYS = {"allowed_types"}
NUMERIC_TEXT_RE = re.compile(r"^\(?-?[$£€]?\s*\d[\d,]*(?:\.\d+)?%?\)?$")

GROUP_RECORD_KEYS = {
    "group_id",
    "source",
    "split",
    "document_id",
    "template_family",
    "question",
    "counterfactual_policy",
    "renderer_seed_base",
    "variants",
}
REQUIRED_VARIANTS = {
    "clean",
    "target_value_replacement",
    "header_address_swap",
    "irrelevant_value_replacement",
}
VARIANT_PAYLOAD_KEYS = {
    "variant",
    "intervention_type",
    "image_id",
    "image_path",
    "renderer_seed",
    "answer",
    "evidence",
    "evidence_units",
    "gold_evidence_ids",
    "changed_cell_ids",
    "expected_behavior",
    "validation_status",
    "validation_notes",
}
GOLD_EVIDENCE_ID_KEYS = {"target_value", "row_headers", "column_headers", "parent_headers", "unit_cells"}
EXPECTED_BEHAVIOURS = {"baseline", "answer_changes", "answer_invariant"}


def normalise_numeric_value(raw: Any) -> float | None:
    text = "" if raw is None else str(raw).strip()
    if not text:
        return None
    compact = " ".join(text.split())
    if not NUMERIC_TEXT_RE.match(compact):
        return None

    negative = compact.startswith("(") and compact.endswith(")")
    if negative:
        compact = compact[1:-1]
    compact = compact.replace(",", "")
    compact = compact.replace("$", "").replace("£", "").replace("€", "")
    compact = compact.replace("%", "")
    compact = compact.strip()
    if not compact:
        return None

    try:
        value = float(compact)
    except ValueError:
        return None
    return -value if negative else value


def normalise_bbox(
    bbox: Sequence[int | float],
    image_width: int,
    image_height: int,
) -> list[float]:
    if len(bbox) != 4:
        raise ValueError(f"Expected bbox with four coordinates, got {bbox!r}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Image size must be positive, got {(image_width, image_height)!r}")

    x0, y0, x1, y1 = (float(value) for value in bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid bbox {bbox!r}")
    return [
        round(x0 / image_width, 6),
        round(y0 / image_height, 6),
        round(x1 / image_width, 6),
        round(y1 / image_height, 6),
    ]


def validate_bbox_normalised(value: Any, field_name: str) -> None:
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError(f"{field_name} must be a list of four numbers")
    previous = None
    for index, coordinate in enumerate(value):
        if isinstance(coordinate, bool) or not isinstance(coordinate, int | float):
            raise ValueError(f"{field_name}[{index}] must be numeric")
        if coordinate < 0 or coordinate > 1:
            raise ValueError(f"{field_name}[{index}] must be in [0, 1]")
        if previous is not None and index in {2, 3}:
            pass
        previous = coordinate
    if value[2] <= value[0] or value[3] <= value[1]:
        raise ValueError(f"{field_name} must have x1>x0 and y1>y0")


def require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    keys = set(value)
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{name} keys mismatch: {', '.join(details)}")


def validate_answer_block(answer: Any, name: str) -> None:
    if not isinstance(answer, Mapping):
        raise ValueError(f"{name} must be a mapping")
    require_exact_keys(answer, ANSWER_KEYS, name)
    if not isinstance(answer["raw"], str) or not answer["raw"].strip():
        raise ValueError(f"{name}.raw must be a non-empty string")
    if isinstance(answer["normalised_value"], bool) or not isinstance(answer["normalised_value"], int | float):
        raise ValueError(f"{name}.normalised_value must be numeric")
    for key in ("unit", "scale", "metric", "period"):
        if answer[key] is not None and not isinstance(answer[key], str):
            raise ValueError(f"{name}.{key} must be null or string")


def validate_evidence_units_block(evidence_units: Any, name: str) -> set[str]:
    if not isinstance(evidence_units, list) or not evidence_units:
        raise ValueError(f"{name} must be a non-empty list")
    unit_ids: set[str] = set()
    for index, unit in enumerate(evidence_units):
        if not isinstance(unit, Mapping):
            raise ValueError(f"{name}[{index}] must be a mapping")
        require_exact_keys(unit, EVIDENCE_UNIT_KEYS, f"{name}[{index}]")
        for key in ("id", "type", "role"):
            if not isinstance(unit[key], str) or not unit[key].strip():
                raise ValueError(f"{name}[{index}].{key} must be a non-empty string")
        if not isinstance(unit["text"], str):
            raise ValueError(f"{name}[{index}].text must be a string")
        if unit["id"] in unit_ids:
            raise ValueError(f"Duplicate evidence unit id {unit['id']!r}")
        unit_ids.add(unit["id"])
        validate_bbox_normalised(unit["bbox_normalised"], f"{name}[{index}].bbox_normalised")
        if not isinstance(unit["metadata"], Mapping):
            raise ValueError(f"{name}[{index}].metadata must be a mapping")
    return unit_ids


def validate_evidence_block(evidence: Any, unit_ids: set[str], name: str) -> None:
    if not isinstance(evidence, Mapping):
        raise ValueError(f"{name} must be a mapping")
    require_exact_keys(evidence, EVIDENCE_KEYS, name)
    if not isinstance(evidence["type"], str) or not evidence["type"].strip():
        raise ValueError(f"{name}.type must be a non-empty string")
    if evidence["target_id"] not in unit_ids:
        raise ValueError(f"{name}.target_id must exist in evidence_units")
    validate_bbox_normalised(evidence["bbox_normalised"], f"{name}.bbox_normalised")
    for key in ("row_header_ids", "column_header_ids", "unit_region_ids"):
        ids = evidence[key]
        if not isinstance(ids, list):
            raise ValueError(f"{name}.{key} must be a list")
        missing = [item for item in ids if item not in unit_ids]
        if missing:
            raise ValueError(f"{name}.{key} contains unknown ids: {missing}")


def validate_counterfactual_policy_block(policy: Any, name: str) -> None:
    if not isinstance(policy, Mapping):
        raise ValueError(f"{name} must be a mapping")
    require_exact_keys(policy, COUNTERFACTUAL_POLICY_KEYS, name)
    allowed_types = policy["allowed_types"]
    if not isinstance(allowed_types, list) or any(not isinstance(item, str) or not item for item in allowed_types):
        raise ValueError(f"{name}.allowed_types must be a list of non-empty strings")


def validate_clean_record(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, CLEAN_RECORD_KEYS, "clean record")

    for key in ("group_id", "source", "split", "document_id", "template_family", "image_path", "question"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")

    validate_answer_block(record["answer"], "answer")
    unit_ids = validate_evidence_units_block(record["evidence_units"], "evidence_units")
    validate_evidence_block(record["evidence"], unit_ids, "evidence")
    validate_counterfactual_policy_block(record["counterfactual_policy"], "counterfactual_policy")


def validate_variant_payload(payload: Mapping[str, Any], *, name: str, expect_changed_cells: bool) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{name} must be a mapping")
    require_exact_keys(payload, VARIANT_PAYLOAD_KEYS, name)

    for key in ("variant", "intervention_type", "image_id", "image_path", "expected_behavior", "validation_status"):
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"{name}.{key} must be a non-empty string")
    if payload["expected_behavior"] not in EXPECTED_BEHAVIOURS:
        raise ValueError(f"{name}.expected_behavior must be one of {sorted(EXPECTED_BEHAVIOURS)}")
    if isinstance(payload["renderer_seed"], bool) or not isinstance(payload["renderer_seed"], int):
        raise ValueError(f"{name}.renderer_seed must be an int")
    if not isinstance(payload["validation_notes"], list) or any(
        not isinstance(item, str) for item in payload["validation_notes"]
    ):
        raise ValueError(f"{name}.validation_notes must be a list of strings")

    validate_answer_block(payload["answer"], f"{name}.answer")
    unit_ids = validate_evidence_units_block(payload["evidence_units"], f"{name}.evidence_units")
    validate_evidence_block(payload["evidence"], unit_ids, f"{name}.evidence")

    gold_evidence_ids = payload["gold_evidence_ids"]
    if not isinstance(gold_evidence_ids, Mapping):
        raise ValueError(f"{name}.gold_evidence_ids must be a mapping")
    require_exact_keys(gold_evidence_ids, GOLD_EVIDENCE_ID_KEYS, f"{name}.gold_evidence_ids")
    for role, ids in gold_evidence_ids.items():
        if not isinstance(ids, list) or any(item not in unit_ids for item in ids):
            raise ValueError(f"{name}.gold_evidence_ids.{role} must be a list of known evidence unit ids")

    changed_cell_ids = payload["changed_cell_ids"]
    if not isinstance(changed_cell_ids, list) or any(not isinstance(item, str) for item in changed_cell_ids):
        raise ValueError(f"{name}.changed_cell_ids must be a list of strings")
    if expect_changed_cells and not changed_cell_ids:
        raise ValueError(f"{name}.changed_cell_ids must be non-empty for a counterfactual variant")
    if not expect_changed_cells and changed_cell_ids:
        raise ValueError(f"{name}.changed_cell_ids must be empty for the clean variant")


def validate_group_record(record: Mapping[str, Any]) -> None:
    require_exact_keys(record, GROUP_RECORD_KEYS, "group record")

    for key in ("group_id", "source", "split", "document_id", "template_family", "question"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} must be a non-empty string")
    if isinstance(record["renderer_seed_base"], bool) or not isinstance(record["renderer_seed_base"], int):
        raise ValueError("renderer_seed_base must be an int")

    validate_counterfactual_policy_block(record["counterfactual_policy"], "counterfactual_policy")

    variants = record["variants"]
    if not isinstance(variants, Mapping):
        raise ValueError("variants must be a mapping")
    require_exact_keys(variants, REQUIRED_VARIANTS, "variants")
    for variant_name, payload in variants.items():
        if isinstance(payload, Mapping) and payload.get("variant") != variant_name:
            raise ValueError(f"variants.{variant_name}.variant must equal {variant_name!r}")
        validate_variant_payload(
            payload,
            name=f"variants.{variant_name}",
            expect_changed_cells=(variant_name != "clean"),
        )


def write_jsonl_record(handle: Any, record: Mapping[str, Any], *, validator: Any = validate_clean_record) -> None:
    validator(record)
    handle.write(json.dumps(record, sort_keys=True) + "\n")
