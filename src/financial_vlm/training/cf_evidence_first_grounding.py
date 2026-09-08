"""CF augmentation + Evidence-first Grounding: combines the two prior ablations.

Follow-up named explicitly in the Evidence-first Grounding spec's closing
note: "using exactly the same CF examples and schedule as the existing
CF-augmentation baseline." Every group is visited 3 times (clean + one
counterfactual per visit, cycling target_value / header_address /
irrelevant_value in a per-group shuffled-balanced order) -- identical
schedule to CF-augmentation (financial_vlm.training.cf_cycle_sampler). The
difference is the supervision target on *each side* of the pair: instead of
plain-QA (`gold_answer` only), both the clean side and the counterfactual
side use the Evidence-first bbox-before-answer target
(`<bbox>x1 y1 x2 y2</bbox>\\n<answer>...</answer>`), resolved independently
from each side's own evidence block.

This module only *combines* the two existing modules -- it does not modify
either. CF-augmentation's cycle/shuffle logic (``EpochCycleSampler``,
``materialize_cycles`` semantics for group/cf_type selection) is reused
unmodified via composition: it is run once to get the (group_id, cf_type,
cycle_index, position_in_cycle) sequence, then each pair's *content* is
rebuilt here from the underlying ``GroupRecord`` using
``grounding_target.build_grounding_example`` (unmodified) so the bbox is
available on both sides -- ``cf_cycle_sampler.build_training_pair``'s
``_example_view`` doesn't carry bbox, which is why a thin wrapper is needed.

Pure stdlib, no torch/PIL -- consistent with both modules it composes.
"""

from __future__ import annotations

from typing import Any, Mapping

from financial_vlm.data.grounding_target import (
    GroundingExample,
    build_grounding_example,
    validate_grounding_payload,
)
from financial_vlm.training.cf_cycle_sampler import (
    CF_TYPES,
    VARIANT_BY_CF_TYPE,
    EpochCycleSampler,
    GroupRecord,
)


def validate_and_filter_groups(
    groups: Mapping[str, GroupRecord],
) -> tuple[dict[str, GroupRecord], dict[str, int]]:
    """Keep only groups where clean AND all 3 counterfactual variants pass grounding validation.

    Grounding-only / Evidence-first only ever train on the ``clean`` variant,
    so they filter on clean alone. Here every visit pairs clean with one of
    three CF types chosen at cycle-shuffle time, so a group must be
    grounding-valid on *all four* variants to guarantee
    ``build_evidence_first_cf_pair`` never raises later, regardless of which
    CF type a given visit draws. Never silently drops a group: every
    exclusion is counted under an ``invalid__<variant>__<reason>`` key.
    """

    stats: dict[str, int] = {"total_groups": 0, "valid_groups": 0}
    valid: dict[str, GroupRecord] = {}

    for group_id, group in groups.items():
        stats["total_groups"] += 1
        reasons: list[str] = []

        clean_reason = validate_grounding_payload(group.clean)
        if clean_reason is not None:
            reasons.append(f"clean__{clean_reason}")
        for cf_type in CF_TYPES:
            cf_reason = validate_grounding_payload(group.counterfactuals[cf_type])
            if cf_reason is not None:
                reasons.append(f"{cf_type}__{cf_reason}")

        if reasons:
            key = f"invalid__{reasons[0]}"
            stats[key] = stats.get(key, 0) + 1
            continue

        valid[group_id] = group
        stats["valid_groups"] += 1

    return valid, stats


def build_evidence_first_cf_pair(
    group: GroupRecord,
    cf_type: str,
    *,
    cycle_index: int,
    position_in_cycle: int,
) -> dict[str, Any]:
    """Build one {clean, counterfactual} pair, each side an Evidence-first GroundingExample.

    Raises ValueError (propagated from ``build_grounding_example``) if either
    side fails grounding validation -- callers should filter groups with
    ``validate_and_filter_groups`` first so this never happens at
    materialization time.
    """

    if cf_type not in CF_TYPES:
        raise ValueError(f"cf_type must be one of {CF_TYPES}, got {cf_type!r}")

    clean_example = build_grounding_example(group.group_id, "clean", group.question, group.clean)
    cf_variant_name = VARIANT_BY_CF_TYPE[cf_type]
    cf_example = build_grounding_example(group.group_id, cf_variant_name, group.question, group.counterfactuals[cf_type])

    return {
        "group_id": group.group_id,
        "cf_type": cf_type,
        "clean": clean_example,
        "counterfactual": cf_example,
        "cycle_index": cycle_index,
        "position_in_cycle": position_in_cycle,
    }


def materialize_evidence_first_cf_pairs(
    groups: Mapping[str, GroupRecord],
    *,
    global_seed: int,
    num_cycles: int,
) -> list[dict[str, Any]]:
    """Same group/cf_type schedule as ``cf_cycle_sampler.materialize_cycles`` (same
    ``global_seed`` -> byte-identical group order and cf_type assignment), but
    each returned pair carries full ``GroundingExample`` objects (bbox_1000,
    target_cell_id, Evidence-first target text) for both sides instead of the
    plain-QA ``_example_view``. ``num_cycles=1`` visits each group exactly 3
    times (once per CF type) -- matching CF-augmentation's training budget.
    """

    if num_cycles < 1:
        raise ValueError("num_cycles must be at least 1")

    sampler = EpochCycleSampler(groups, global_seed=global_seed)
    all_pairs: list[dict[str, Any]] = []
    for cycle in range(num_cycles):
        for offset in range(len(CF_TYPES)):
            block_index = cycle * len(CF_TYPES) + offset
            for raw_pair in sampler.epoch(block_index):
                group = groups[raw_pair["group_id"]]
                cf_type = raw_pair["counterfactual"]["cf_type"]
                all_pairs.append(
                    build_evidence_first_cf_pair(
                        group,
                        cf_type,
                        cycle_index=raw_pair["cycle_index"],
                        position_in_cycle=raw_pair["position_in_cycle"],
                    )
                )
    return all_pairs


def pair_to_log_record(pair: Mapping[str, Any], *, run_id: str, epoch: int, global_step: int) -> dict[str, Any]:
    clean: GroundingExample = pair["clean"]
    counterfactual: GroundingExample = pair["counterfactual"]
    return {
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "group_id": pair["group_id"],
        "cycle_index": pair["cycle_index"],
        "position_in_cycle": pair["position_in_cycle"],
        "cf_type": pair["cf_type"],
        "clean_target_cell_id": clean.target_cell_id,
        "cf_target_cell_id": counterfactual.target_cell_id,
        "clean_gold_answer": clean.answer_raw,
        "cf_gold_answer": counterfactual.answer_raw,
        "clean_bbox_1000": list(clean.bbox_1000),
        "cf_bbox_1000": list(counterfactual.bbox_1000),
    }
