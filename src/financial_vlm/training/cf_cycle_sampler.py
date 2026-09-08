"""Group-aware shuffled balanced cycle sampler for counterfactual training.

Consumes a ``canonical_group_cf_v1`` manifest (one JSONL line per group, with
``clean`` plus three counterfactual variants nested under ``variants``) and
returns, for every visit to a group, ``clean + exactly one counterfactual``.
The counterfactual type cycles through all three types in a random-but-
balanced order per group: each type exactly once before any repeats, then an
independent reshuffle for the next cycle.

This module is intentionally framework-agnostic (pure stdlib, no torch/PIL):
it returns plain dicts referencing image paths, not loaded pixels. A torch
``Dataset``/``DataLoader`` wrapper built on top of it (part of the deferred
training-script integration) is responsible for actually loading images and
computing losses.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Mapping, Sequence

from financial_vlm.data.canonical_schema import validate_group_record
from financial_vlm.data.synfintabs_training import load_jsonl, stable_group_seed


CF_TYPES = ["target_value", "header_address", "irrelevant_value"]

VARIANT_BY_CF_TYPE = {
    "target_value": "target_value_replacement",
    "header_address": "header_address_swap",
    "irrelevant_value": "irrelevant_value_replacement",
}


@dataclass(frozen=True)
class GroupRecord:
    group_id: str
    question: str
    clean: Mapping[str, Any]
    counterfactuals: Mapping[str, Mapping[str, Any]]


def _gold_evidence_cell_ids(variant_payload: Mapping[str, Any]) -> set[str]:
    evidence = variant_payload["evidence"]
    ids = {evidence["target_id"]}
    ids.update(evidence["row_header_ids"])
    ids.update(evidence["column_header_ids"])
    ids.update(evidence["unit_region_ids"])
    return ids


def parse_group_records(records: Iterable[Mapping[str, Any]]) -> dict[str, GroupRecord]:
    """Validate and convert canonical_group_cf_v1 records into GroupRecords.

    Raises ValueError, tagged with the offending group_id, for any missing
    counterfactual variant, duplicate group_id, or an irrelevant_value edit
    that touches the clean variant's gold evidence -- before any sampler is
    constructed, so these problems are caught before training starts.
    """

    groups: dict[str, GroupRecord] = {}
    for index, record in enumerate(records):
        group_id = record.get("group_id")
        try:
            validate_group_record(record)
        except ValueError as exc:
            raise ValueError(f"record[{index}] (group_id={group_id!r}) failed schema validation: {exc}") from exc

        if group_id in groups:
            raise ValueError(f"duplicate group_id {group_id!r} in manifest")

        variants = record["variants"]
        clean = variants["clean"]
        counterfactuals = {
            cf_type: variants[variant_name] for cf_type, variant_name in VARIANT_BY_CF_TYPE.items()
        }

        gold_ids = _gold_evidence_cell_ids(clean)
        changed_ids = set(counterfactuals["irrelevant_value"]["changed_cell_ids"])
        overlap = changed_ids & gold_ids
        if overlap:
            raise ValueError(
                f"group_id {group_id!r}: irrelevant_value changed_cell_ids {sorted(overlap)} "
                "overlap the clean variant's gold evidence ids"
            )

        groups[group_id] = GroupRecord(
            group_id=group_id,
            question=record["question"],
            clean=clean,
            counterfactuals=counterfactuals,
        )

    return groups


def load_group_records(manifest_path: Path) -> dict[str, GroupRecord]:
    return parse_group_records(load_jsonl(manifest_path))


def deterministic_permutation(global_seed: int, group_id: str, cycle_index: int) -> tuple[str, str, str]:
    """A reproducible shuffle of CF_TYPES for one (group, cycle).

    Deterministic from (global_seed, group_id, cycle_index) via the same
    SHA-256-based stable hash used for split assignment elsewhere in this
    project -- never Python's unstable built-in hash().
    """

    seed = stable_group_seed(global_seed, f"{group_id}:{cycle_index}")
    return tuple(random.Random(seed).sample(CF_TYPES, len(CF_TYPES)))  # type: ignore[return-value]


def _example_view(variant_payload: Mapping[str, Any], question: str) -> dict[str, Any]:
    return {
        "image_id": variant_payload["image_id"],
        "image": variant_payload["image_path"],
        "question": question,
        "gold_answer": variant_payload["answer"]["raw"],
        "target_cell_id": variant_payload["evidence"]["target_id"],
        "gold_evidence_ids": variant_payload["gold_evidence_ids"],
    }


def build_training_pair(
    group: GroupRecord,
    cf_type: str,
    *,
    cycle_index: int,
    position_in_cycle: int,
) -> dict[str, Any]:
    if cf_type not in CF_TYPES:
        raise ValueError(f"cf_type must be one of {CF_TYPES}, got {cf_type!r}")

    cf_payload = group.counterfactuals[cf_type]
    counterfactual = _example_view(cf_payload, group.question)
    counterfactual["cf_type"] = cf_type
    counterfactual["expected_behavior"] = cf_payload["expected_behavior"]

    return {
        "group_id": group.group_id,
        "clean": _example_view(group.clean, group.question),
        "counterfactual": counterfactual,
        "cycle_index": cycle_index,
        "position_in_cycle": position_in_cycle,
    }


class GroupCycleSampler:
    """Visit-based, stateful, checkpointable shuffled balanced cycle sampler.

    ``current_order`` (the shuffled permutation of CF_TYPES for a cycle) is a
    pure function of (global_seed, group_id, cycle_index), and cycle_index is
    itself derived from visit_count. So the only state that needs to persist
    across a checkpoint is visit_count per group -- the order is always
    recomputed on demand, never cached, which makes resume trivially correct
    regardless of where in a cycle training stopped.
    """

    def __init__(
        self,
        groups: Mapping[str, GroupRecord],
        *,
        global_seed: int,
        on_pair: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._groups = dict(groups)
        self._global_seed = global_seed
        self._on_pair = on_pair
        self._visit_counts: dict[str, int] = {group_id: 0 for group_id in self._groups}

    def visit(self, group_id: str) -> dict[str, Any]:
        if group_id not in self._groups:
            raise KeyError(f"Unknown group_id {group_id!r}")

        visit_count = self._visit_counts[group_id]
        position = visit_count % len(CF_TYPES)
        cycle_index = visit_count // len(CF_TYPES)
        order = deterministic_permutation(self._global_seed, group_id, cycle_index)
        cf_type = order[position]

        pair = build_training_pair(
            self._groups[group_id], cf_type, cycle_index=cycle_index, position_in_cycle=position
        )
        self._visit_counts[group_id] = visit_count + 1
        if self._on_pair is not None:
            self._on_pair(pair)
        return pair

    def state_dict(self) -> dict[str, int]:
        return dict(self._visit_counts)

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        for group_id, visit_count in state.items():
            if group_id not in self._visit_counts:
                raise KeyError(f"Unknown group_id in checkpoint state: {group_id!r}")
            self._visit_counts[group_id] = visit_count


class EpochCycleSampler:
    """Epoch-based sampler: one visit per group per epoch.

    Group order is shuffled per epoch (deterministically, from
    global_seed + epoch_index). No persisted state is needed to resume --
    epoch(N) is a pure function of N.
    """

    def __init__(
        self,
        groups: Mapping[str, GroupRecord],
        *,
        global_seed: int,
        on_pair: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._groups = dict(groups)
        self._global_seed = global_seed
        self._on_pair = on_pair

    def epoch(self, epoch_index: int) -> list[dict[str, Any]]:
        cycle_index = epoch_index // len(CF_TYPES)
        position_in_cycle = epoch_index % len(CF_TYPES)

        group_ids = list(self._groups)
        shuffle_seed = stable_group_seed(self._global_seed, f"epoch:{epoch_index}")
        random.Random(shuffle_seed).shuffle(group_ids)

        pairs = []
        for group_id in group_ids:
            order = deterministic_permutation(self._global_seed, group_id, cycle_index)
            cf_type = order[position_in_cycle]
            pair = build_training_pair(
                self._groups[group_id], cf_type, cycle_index=cycle_index, position_in_cycle=position_in_cycle
            )
            pairs.append(pair)
            if self._on_pair is not None:
                self._on_pair(pair)
        return pairs


def verify_cycle_coverage(pairs: Iterable[dict[str, Any]]) -> None:
    """Raise ValueError unless every group in ``pairs`` saw each CF type exactly once.

    ``pairs`` must cover exactly one complete cycle (3 visits/epochs) per
    group -- e.g. all pairs from 3 consecutive epochs, or 3 consecutive
    visits to each group.
    """

    cf_types_by_group: dict[str, list[str]] = {}
    for pair in pairs:
        cf_types_by_group.setdefault(pair["group_id"], []).append(pair["counterfactual"]["cf_type"])

    violations = {
        group_id: types
        for group_id, types in cf_types_by_group.items()
        if sorted(types) != sorted(CF_TYPES)
    }
    if violations:
        raise ValueError(f"cycle coverage violated for groups: {violations}")


def summarize_epoch(
    pairs: Sequence[dict[str, Any]],
    *,
    all_group_ids: Collection[str] | None = None,
) -> dict[str, int]:
    visited = Counter(pair["group_id"] for pair in pairs)
    cf_type_counts = Counter(pair["counterfactual"]["cf_type"] for pair in pairs)
    duplicate_visits = sum(count - 1 for count in visited.values() if count > 1)
    missing_groups = len(set(all_group_ids) - set(visited)) if all_group_ids is not None else 0

    return {
        "groups_visited": len(visited),
        "target_value_count": cf_type_counts.get("target_value", 0),
        "header_address_count": cf_type_counts.get("header_address", 0),
        "irrelevant_value_count": cf_type_counts.get("irrelevant_value", 0),
        "missing_groups": missing_groups,
        "duplicate_visits": duplicate_visits,
    }


def materialize_cycles(
    groups: Mapping[str, GroupRecord],
    *,
    global_seed: int,
    num_cycles: int,
) -> list[dict[str, Any]]:
    """Flatten ``num_cycles`` full shuffled-balanced cycles into one pair list.

    Each cycle is exactly ``len(CF_TYPES)`` epoch-blocks (one visit per group
    per block). ``verify_cycle_coverage`` is checked per cycle before its
    pairs are appended, so a broken cycle raises before any of its pairs
    reach training. With ``num_cycles=1`` every group is visited exactly 3
    times (once per CF type) -- the same total per-group visit count as the
    clean-only baseline's ``num_train_epochs: 3``, for a like-for-like
    training-budget comparison.
    """

    if num_cycles < 1:
        raise ValueError("num_cycles must be at least 1")

    sampler = EpochCycleSampler(groups, global_seed=global_seed)
    all_pairs: list[dict[str, Any]] = []
    for cycle in range(num_cycles):
        cycle_pairs: list[dict[str, Any]] = []
        for offset in range(len(CF_TYPES)):
            cycle_pairs.extend(sampler.epoch(cycle * len(CF_TYPES) + offset))
        verify_cycle_coverage(cycle_pairs)
        all_pairs.extend(cycle_pairs)
    return all_pairs


def pair_to_log_record(pair: Mapping[str, Any], *, run_id: str, epoch: int, global_step: int) -> dict[str, Any]:
    clean = pair["clean"]
    counterfactual = pair["counterfactual"]
    return {
        "run_id": run_id,
        "epoch": epoch,
        "global_step": global_step,
        "group_id": pair["group_id"],
        "cycle_index": pair["cycle_index"],
        "position_in_cycle": pair["position_in_cycle"],
        "cf_type": counterfactual["cf_type"],
        "clean_image_id": clean["image_id"],
        "cf_image_id": counterfactual["image_id"],
        "clean_gold_answer": clean["gold_answer"],
        "cf_gold_answer": counterfactual["gold_answer"],
        "clean_target_cell_id": clean["target_cell_id"],
        "cf_target_cell_id": counterfactual["target_cell_id"],
    }
