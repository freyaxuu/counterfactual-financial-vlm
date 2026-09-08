"""Utilities for SynFinTabs training split and epoch sampling manifests."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_COUNTERFACTUAL_VARIANTS = (
    "target_value",
    "target_value_replacement",
    "header_swap",
    "header_address_swap",
    "irrelevant_cell",
    "irrelevant_value",
    "irrelevant_value_replacement",
)


@dataclass(frozen=True)
class SplitRequest:
    name: str
    target_groups: int


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("_")[:120] or "unknown"


def source_table_key(source_table_id: str | None, source_index: int) -> str:
    if source_table_id:
        return f"id:{source_table_id}"
    return f"index:{source_index}"


def read_excluded_table_keys(paths: Iterable[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        with path.expanduser().open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                source_table_id = str(record.get("source_table_id") or "")
                source_index = int(record.get("source_index", -1))
                excluded.add(source_table_key(source_table_id, source_index))
    return excluded


def choose_split(
    remaining_groups: Mapping[str, int],
    rng: random.Random,
) -> str | None:
    active = [(name, count) for name, count in remaining_groups.items() if count > 0]
    if not active:
        return None
    total = sum(count for _, count in active)
    cursor = rng.randrange(total)
    for name, count in active:
        if cursor < count:
            return name
        cursor -= count
    return active[-1][0]


def choose_split_with_template_pool(
    template_family: str,
    remaining_groups: Mapping[str, int],
    rng: random.Random,
    *,
    ood_template_ids: frozenset[str] = frozenset(),
    ood_split_name: str = "test",
) -> str | None:
    """Table-level split assignment that reserves OOD templates for one split.

    Tables whose ``template_family`` is in ``ood_template_ids`` are only
    eligible for ``ood_split_name``'s remaining quota; every other table is
    eligible for every split except ``ood_split_name``. When
    ``ood_template_ids`` is empty and ``remaining_groups`` has no
    ``ood_split_name`` key, this is exactly equivalent to
    ``choose_split(remaining_groups, rng)``.
    """
    if template_family in ood_template_ids:
        eligible = {ood_split_name: remaining_groups.get(ood_split_name, 0)}
    else:
        eligible = {name: count for name, count in remaining_groups.items() if name != ood_split_name}
    return choose_split(eligible, rng)


def stable_group_seed(base_seed: int, group_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{group_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


GROUP_SHARED_KEYS = (
    "group_id",
    "source",
    "split",
    "document_id",
    "template_family",
    "question",
    "counterfactual_policy",
)


def flatten_group_records(
    records: Iterable[Mapping[str, Any]],
    variants: Sequence[str],
) -> list[dict[str, Any]]:
    """Flatten ``canonical_group_cf_v1`` records into one flat record per variant.

    Each output record merges a group's shared fields (``group_id``,
    ``question``, ...) with one selected variant's payload, reproducing the
    flat top-level ``variant``/``image_path``/``answer`` shape that
    ``filter_records_by_variant``/``answer_target``/``CanonicalQADataset``
    already consume. Records with no ``variants`` mapping (flat
    ``canonical_clean_v1``-style manifests) pass through unchanged, so this
    is safe to call unconditionally regardless of which schema a manifest
    was written with.
    """

    wanted = set(variants)
    flattened: list[dict[str, Any]] = []
    for record in records:
        variant_payloads = record.get("variants")
        if not isinstance(variant_payloads, Mapping):
            flattened.append(dict(record))
            continue
        shared = {key: record[key] for key in GROUP_SHARED_KEYS if key in record}
        for variant_name, payload in variant_payloads.items():
            if variant_name not in wanted:
                continue
            flat = dict(shared)
            flat.update(payload)
            flattened.append(flat)
    return flattened


def group_records_by_id(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        group_id = str(record["group_id"])
        variant = str(record["variant"])
        grouped[group_id][variant] = dict(record)
    return dict(grouped)


def sample_epoch_records(
    records: Iterable[Mapping[str, Any]],
    *,
    epoch: int,
    seed: int,
    cf_variants: Sequence[str] = DEFAULT_COUNTERFACTUAL_VARIANTS,
) -> list[dict[str, Any]]:
    """Return clean plus one deterministic random counterfactual per group.

    The same ``seed`` and ``epoch`` always produce the same sampled manifest, while
    changing the epoch resamples counterfactual variants per group.
    """

    sampled: list[dict[str, Any]] = []
    grouped = group_records_by_id(records)
    for group_id in sorted(grouped):
        variants = grouped[group_id]
        clean = variants.get("clean")
        if clean is None:
            raise ValueError(f"Group {group_id!r} is missing the clean variant")

        available_cf = [variant for variant in cf_variants if variant in variants]
        if not available_cf:
            raise ValueError(f"Group {group_id!r} has no available counterfactual variants")

        rng = random.Random(stable_group_seed(seed + epoch, group_id))
        cf_variant = rng.choice(available_cf)
        for record in (clean, variants[cf_variant]):
            item = dict(record)
            item["epoch"] = epoch
            item["epoch_seed"] = seed
            item["epoch_sampled_counterfactual"] = cf_variant
            sampled.append(item)
    return sampled


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def split_dev_records_by_group(
    records: Iterable[Mapping[str, Any]],
    *,
    quick_groups: int = 20,
    select_groups: int = 80,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a dev manifest into deterministic quick/select group subsets."""

    if quick_groups < 0 or select_groups < 0:
        raise ValueError("quick_groups and select_groups must be non-negative")

    copied = [dict(record) for record in records]
    group_ids = sorted({str(record["group_id"]) for record in copied})
    required = quick_groups + select_groups
    if len(group_ids) < required:
        raise ValueError(f"Need at least {required} groups, found {len(group_ids)}")

    quick_ids = set(group_ids[:quick_groups])
    select_ids = set(group_ids[quick_groups:required])
    quick_records = [record for record in copied if str(record["group_id"]) in quick_ids]
    select_records = [record for record in copied if str(record["group_id"]) in select_ids]
    return quick_records, select_records


def select_records_by_group_prefix(
    records: Iterable[Mapping[str, Any]],
    *,
    group_count: int,
) -> list[dict[str, Any]]:
    """Select all records from the first ``group_count`` sorted groups."""

    if group_count < 0:
        raise ValueError("group_count must be non-negative")

    copied = [dict(record) for record in records]
    group_ids = sorted({str(record["group_id"]) for record in copied})
    if len(group_ids) < group_count:
        raise ValueError(f"Need at least {group_count} groups, found {len(group_ids)}")

    selected_ids = set(group_ids[:group_count])
    return [record for record in copied if str(record["group_id"]) in selected_ids]
