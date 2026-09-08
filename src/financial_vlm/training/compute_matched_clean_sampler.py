"""Compute-matched clean-only pairing: the fair-compute control for CF-augmentation.

Pairs clean render A (the frozen dataset's existing "clean" image) with clean
render B (a second, pixel-distinct render of the identical table/answer -- see
scripts/generate_compute_matched_clean_b.py) for every group, repeated across
``num_cycles`` blocks with the same per-block group-order shuffle style as
financial_vlm.training.cf_cycle_sampler.EpochCycleSampler. Each cycle visits
every group exactly once (unlike EpochCycleSampler's 3-type cycle, there is
only one pair type here), so ``num_cycles=3`` is what matches CF-augmentation's
``num_cycles=1`` (which internally visits each group 3x, once per CF type):
3000 pairs for 1000 groups, 6000 image presentations -- the same totals as
the CF-augmentation sampler, isolating "two images per step" from "one of
them is a counterfactual".

No CF content, no cell edits: both sides carry the identical answer/evidence,
so unlike financial_vlm.training.cf_cycle_sampler this needs no schema
validation beyond "every group has an A and a B".
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Iterable, Mapping

from financial_vlm.data.synfintabs_training import load_jsonl, stable_group_seed


@dataclass(frozen=True)
class CleanPairGroup:
    group_id: str
    question: str
    clean_a: Mapping[str, Any]
    clean_b: Mapping[str, Any]


def _example_view(image_path: str, image_id: str, question: str, gold_answer: str) -> dict[str, Any]:
    return {
        "image_id": image_id,
        "image": image_path,
        "question": question,
        "gold_answer": gold_answer,
    }


def load_clean_pair_groups(
    train_manifest_path: Path,
    clean_b_manifest_path: Path,
) -> dict[str, CleanPairGroup]:
    """Join the frozen train manifest's clean variant with the clean_b manifest.

    Raises ValueError, tagged with the offending group_id, for any train group
    missing a clean_b counterpart -- before any sampler is constructed.
    """

    clean_b_by_group: dict[str, dict[str, Any]] = {}
    for record in load_jsonl(clean_b_manifest_path):
        clean_b_by_group[str(record["group_id"])] = record

    groups: dict[str, CleanPairGroup] = {}
    for record in load_jsonl(train_manifest_path):
        group_id = str(record["group_id"])
        clean_a_payload = record["variants"]["clean"]
        clean_b_record = clean_b_by_group.get(group_id)
        if clean_b_record is None:
            raise ValueError(f"group_id {group_id!r}: no clean_b render found in {clean_b_manifest_path}")

        question = str(record["question"])
        groups[group_id] = CleanPairGroup(
            group_id=group_id,
            question=question,
            clean_a=_example_view(
                clean_a_payload["image_path"], clean_a_payload["image_id"], question, clean_a_payload["answer"]["raw"]
            ),
            clean_b=_example_view(
                clean_b_record["image_path"], f"{group_id}__clean_b", question, clean_a_payload["answer"]["raw"]
            ),
        )

    return groups


def build_clean_pair(group: CleanPairGroup, *, cycle_index: int, position_in_cycle: int) -> dict[str, Any]:
    return {
        "group_id": group.group_id,
        "clean_a": dict(group.clean_a),
        "clean_b": dict(group.clean_b),
        "cycle_index": cycle_index,
        "position_in_cycle": position_in_cycle,
    }


def materialize_compute_matched_pairs(
    groups: Mapping[str, CleanPairGroup],
    *,
    global_seed: int,
    num_cycles: int,
) -> list[dict[str, Any]]:
    """Flatten ``num_cycles`` full passes into one (clean_a, clean_b) pair list.

    Every group is visited exactly once per cycle (3 visits total for
    num_cycles=3, matching CF-augmentation's num_cycles=1), always with the
    SAME fixed clean_a/clean_b pair -- matching
    how CF-augmentation reuses the same single clean image across all 3
    visits. Group order is shuffled per cycle, deterministically, from
    global_seed + cycle_index, mirroring EpochCycleSampler.
    """

    if num_cycles < 1:
        raise ValueError("num_cycles must be at least 1")

    all_pairs: list[dict[str, Any]] = []
    for cycle in range(num_cycles):
        group_ids = list(groups)
        shuffle_seed = stable_group_seed(global_seed, f"compute_matched_cycle:{cycle}")
        random.Random(shuffle_seed).shuffle(group_ids)
        for group_id in group_ids:
            all_pairs.append(build_clean_pair(groups[group_id], cycle_index=cycle, position_in_cycle=0))
    return all_pairs
