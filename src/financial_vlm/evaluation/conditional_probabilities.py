"""Group-level conditional counterfactual failure probabilities."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from financial_vlm.evaluation.pilot_accuracy import (
    canonical_variant,
    record_exact_correct,
    record_numeric_correct,
)


REQUIRED_VARIANTS = ("clean", "target_value", "header_address_swap", "irrelevant_value")
VARIANT_SYMBOLS = {
    "clean": "C",
    "target_value": "T",
    "header_address_swap": "H",
    "irrelevant_value": "I",
}


def record_correct(record: Mapping[str, Any], correctness: str) -> bool:
    if correctness == "numeric":
        return record_numeric_correct(record)
    if correctness == "exact":
        return record_exact_correct(record)
    if correctness == "is_correct":
        if "is_correct" in record:
            return bool(record["is_correct"])
        return record_numeric_correct(record)
    raise ValueError(f"Unsupported correctness mode: {correctness}")


def group_predictions(
    records: Iterable[Mapping[str, Any]],
    correctness: str,
) -> tuple[dict[str, dict[str, dict[str, bool]]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, bool]]] = defaultdict(lambda: defaultdict(dict))
    duplicates: list[dict[str, Any]] = []

    for record in records:
        setting = str(record.get("setting") or "")
        group_id = str(record.get("group_id") or "")
        variant = canonical_variant(str(record.get("variant") or ""))
        if not setting or not group_id or variant not in REQUIRED_VARIANTS:
            continue

        variants = grouped[setting][group_id]
        if variant in variants:
            duplicates.append({"setting": setting, "group_id": group_id, "variant": variant})
            continue
        variants[variant] = record_correct(record, correctness)

    return {setting: dict(groups) for setting, groups in grouped.items()}, duplicates


def rate(count: int, denominator: int) -> float | None:
    return count / denominator if denominator else None


def summarize_setting(groups: Mapping[str, Mapping[str, bool]]) -> dict[str, Any]:
    complete = {
        group_id: values
        for group_id, values in groups.items()
        if all(variant in values for variant in REQUIRED_VARIANTS)
    }
    clean_correct = {
        group_id: values
        for group_id, values in complete.items()
        if bool(values["clean"])
    }

    denominator = len(clean_correct)
    target_failures = sum(not values["target_value"] for values in clean_correct.values())
    header_failures = sum(not values["header_address_swap"] for values in clean_correct.values())
    targeted_failures = sum(
        (not values["target_value"]) or (not values["header_address_swap"])
        for values in clean_correct.values()
    )
    irrelevant_failures = sum(not values["irrelevant_value"] for values in clean_correct.values())

    p_t_fail = rate(target_failures, denominator)
    p_h_fail = rate(header_failures, denominator)
    p_targeted_fail = rate(targeted_failures, denominator)
    p_i_fail = rate(irrelevant_failures, denominator)

    h_minus_i = None if p_h_fail is None or p_i_fail is None else p_h_fail - p_i_fail
    t_minus_i = None if p_t_fail is None or p_i_fail is None else p_t_fail - p_i_fail
    targeted_minus_i = (
        None
        if p_targeted_fail is None or p_i_fail is None
        else p_targeted_fail - p_i_fail
    )

    return {
        "groups_total": len(groups),
        "groups_complete": len(complete),
        "groups_missing_required_variants": len(groups) - len(complete),
        "clean_correct_groups": denominator,
        "condition": "C=1",
        "statistical_unit": "group",
        "counts": {
            "target_value_failures_given_clean_correct": target_failures,
            "header_address_failures_given_clean_correct": header_failures,
            "any_targeted_failures_given_clean_correct": targeted_failures,
            "irrelevant_instability_given_clean_correct": irrelevant_failures,
        },
        "probabilities": {
            "P(T=0|C=1)": p_t_fail,
            "P(H=0|C=1)": p_h_fail,
            "P(T=0_or_H=0|C=1)": p_targeted_fail,
            "P(I=0|C=1)": p_i_fail,
        },
        "differences": {
            "P(H=0|C=1)-P(I=0|C=1)": h_minus_i,
            "P(T=0|C=1)-P(I=0|C=1)": t_minus_i,
            "P(T=0_or_H=0|C=1)-P(I=0|C=1)": targeted_minus_i,
        },
        "targeted_failure_exceeds_nuisance": {
            "header_gt_irrelevant": None if h_minus_i is None else h_minus_i > 0,
            "target_value_gt_irrelevant": None if t_minus_i is None else t_minus_i > 0,
            "any_targeted_gt_irrelevant": None if targeted_minus_i is None else targeted_minus_i > 0,
        },
    }


def summarize_conditional_probabilities(
    records: Iterable[Mapping[str, Any]],
    *,
    correctness: str = "numeric",
) -> dict[str, Any]:
    grouped, duplicates = group_predictions(records, correctness)
    return {
        "correctness": correctness,
        "statistical_unit": "group",
        "variant_symbols": VARIANT_SYMBOLS,
        "definitions": {
            "C": "clean prediction is correct",
            "T": "target_value_replacement prediction is correct",
            "H": "header_address_swap prediction is correct",
            "I": "irrelevant_value_replacement prediction is correct",
        },
        "duplicate_records": duplicates,
        "by_setting": {
            setting: summarize_setting(groups)
            for setting, groups in sorted(grouped.items())
        },
    }
