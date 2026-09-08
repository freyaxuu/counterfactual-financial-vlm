#!/usr/bin/env python3
"""Render a second, pixel-distinct "clean B" image per existing train group.

This is the prerequisite for the compute-matched clean LoRA control: pairing
clean render A (the frozen dataset's existing "clean" image) with clean
render B, matching CF-augmentation's image count and optimizer-step budget
with zero counterfactual content.

For each group already in manifest_train.jsonl, this replays the exact same
flatten_table -> locate_answer_cell -> build_variant_plans steps used to
originally build that group (matched by source_index parsed from group_id
plus the stored question text), takes only the deterministic "clean" plan
(plans[0], which never depends on the rng passed to build_variant_plans),
and re-renders it with a per-group-seeded jitter on cell_padding (the only
knob render_source_table_image exposes) so clean_B is genuinely
pixel-distinct from clean_A while the table content and answer are
identical. Every regenerated group is verified before rendering: the
relocated answer cell's text must match the original record's answer, or
this fails loudly rather than silently rendering a mismatched table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.synfintabs_pilot import (  # noqa: E402
    build_variant_plans,
    flatten_table,
    locate_answer_cell,
    render_source_table_image,
)
from financial_vlm.data.synfintabs_training import load_jsonl, stable_group_seed  # noqa: E402


DEFAULT_PADDING_JITTER = (-2, -1, 1, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=REPO_ROOT / "configs" / "synfintabs" / "synfintabs_train_dev_test_ood.yaml",
        help="Config providing dataset.id/revision/split (the one used to build the frozen dataset).",
    )
    parser.add_argument("--train-manifest", type=Path, required=True, help="Existing manifest_train.jsonl.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where to write clean_b images + manifest.")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument(
        "--padding-jitter",
        type=int,
        nargs="+",
        default=list(DEFAULT_PADDING_JITTER),
        help="Candidate cell_padding offsets from the default of 4. 0 should never be included -- "
        "it would make clean_B pixel-identical to clean_A.",
    )
    parser.add_argument("--limit-groups", type=int, help="Only generate the first N sorted groups (smoke tests).")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def iter_dataset(dataset_cfg: dict[str, Any]) -> Any:
    from datasets import load_dataset

    return load_dataset(
        dataset_cfg["id"],
        split=dataset_cfg.get("split", "train"),
        revision=dataset_cfg.get("revision"),
        streaming=bool(dataset_cfg.get("streaming", True)),
    )


def parse_source_index(group_id: str) -> int:
    # group_id format: syn_{source_index:06d}_{slug} (see synfintabs_loader.build_synfintabs_group_record)
    parts = group_id.split("_")
    if len(parts) < 3 or parts[0] != "syn":
        raise ValueError(f"Unrecognized group_id format: {group_id!r}")
    return int(parts[1])


def main() -> int:
    args = parse_args()
    if 0 in args.padding_jitter:
        raise ValueError("--padding-jitter must not include 0 (would make clean_B identical to clean_A)")

    dataset_cfg = load_yaml(args.dataset_config)["dataset"]

    train_records = load_jsonl(args.train_manifest)
    if args.limit_groups is not None:
        group_ids = sorted({str(record["group_id"]) for record in train_records})[: args.limit_groups]
        keep = set(group_ids)
        train_records = [record for record in train_records if str(record["group_id"]) in keep]

    targets_by_source_index: dict[int, list[dict[str, Any]]] = {}
    for record in train_records:
        source_index = parse_source_index(str(record["group_id"]))
        targets_by_source_index.setdefault(source_index, []).append(record)
    total_targets = sum(len(records) for records in targets_by_source_index.values())
    if total_targets == 0:
        raise ValueError(f"No groups found in {args.train_manifest}")
    max_source_index = max(targets_by_source_index)

    output_dir = args.output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "clean_b_manifest.jsonl"

    found = 0
    mismatched: list[str] = []
    dataset = iter_dataset(dataset_cfg)

    with manifest_path.open("w", encoding="utf-8") as out:
        for source_index, table in enumerate(dataset):
            if source_index > max_source_index:
                break
            targets = targets_by_source_index.get(source_index)
            if not targets:
                continue

            cells, words = flatten_table(table.get("rows") or [])
            questions_by_text = {str(q.get("question") or ""): q for q in (table.get("questions") or [])}

            for record in targets:
                group_id = str(record["group_id"])
                question_text = str(record["question"])
                question = questions_by_text.get(question_text)
                if question is None:
                    raise ValueError(
                        f"group_id={group_id!r}: question text not found in source table "
                        f"(source_index={source_index}); cannot regenerate cells deterministically"
                    )

                located = locate_answer_cell(question, cells, words)
                if located is None:
                    raise ValueError(f"group_id={group_id!r}: locate_answer_cell failed on replay")

                expected_answer = str(record["variants"]["clean"]["answer"]["raw"])
                if located.answer.strip() != expected_answer.strip():
                    mismatched.append(group_id)
                    continue

                group_rng = random.Random(stable_group_seed(args.seed, f"{group_id}:clean_b"))
                plans = build_variant_plans(located, cells, group_rng)
                if plans is None:
                    raise ValueError(f"group_id={group_id!r}: build_variant_plans failed on replay")
                clean_plan = plans[0]
                if clean_plan.variant != "clean":
                    raise ValueError(f"group_id={group_id!r}: plans[0].variant != 'clean' ({clean_plan.variant!r})")

                cell_padding = 4 + group_rng.choice(args.padding_jitter)
                image = render_source_table_image(cells, table["bbox"], clean_plan, cell_padding=cell_padding)

                image_path = images_dir / f"{group_id}__clean_b.png"
                image.save(image_path)

                out.write(
                    json.dumps(
                        {
                            "group_id": group_id,
                            "image_path": str(image_path.relative_to(output_dir)),
                            "image_size": list(image.size),
                            "cell_padding": cell_padding,
                            "renderer_seed": stable_group_seed(args.seed, f"{group_id}:clean_b"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                out.flush()
                found += 1

            if found + len(mismatched) >= total_targets:
                break

    print(f"targets={total_targets} found={found} mismatched={len(mismatched)} manifest={manifest_path}")
    if mismatched:
        print(f"MISMATCHED group_ids (answer changed on replay, skipped): {mismatched}", file=sys.stderr)
    if found != total_targets:
        print(f"WARNING: only rendered {found}/{total_targets} target groups", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
