#!/usr/bin/env python3
"""Freeze the structurally-sampled "primary" period-pair benchmark: a
~500-pair, adjacent/skip-balanced subset of the frozen v4 period_derived
pool, selected purely from construction-time fields (pair_id, table_id,
metric_a, construction_rule) via
``financial_vlm.data.tatqa_natural_pairs.select_primary_period_pairs``.

Does not read any model prediction. Does not change v4's own construction
rules or pair pool -- this only decides which of v4's already-valid pairs
join a smaller, structurally-capped primary set:

- at most 1 adjacent-period pair per (table, normalized metric) group
- at most 1 skip-period pair per (table, normalized metric) group
- at most --table-cap pairs per table overall
- an overall stratified sample to --target-adjacent / --target-skip

Run once, commit the frozen pair_ids + this script's exact arguments, and
do not re-run with different arguments against the same output path --
per the project's protocol-versioning convention, a different selection
is a new version (SELECTION_PROTOCOL_VERSION), not a silent re-freeze.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.tatqa_natural_pairs import (  # noqa: E402
    ADJACENT_RULE,
    NONADJACENT_RULE,
    SELECTION_PROTOCOL_VERSION,
    select_primary_period_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period-derived-jsonl", type=Path, required=True, help="tatqa_natural_pairs_period_derived_v4.jsonl")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--table-cap", type=int, default=5)
    parser.add_argument("--target-adjacent", type=int, default=375)
    parser.add_argument("--target-skip", type=int, default=125)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(args.period_derived_jsonl)
    by_pair_id = {r["pair_id"]: r for r in records}

    result = select_primary_period_pairs(
        records,
        seed=args.seed,
        table_cap=args.table_cap,
        target_adjacent=args.target_adjacent,
        target_skip=args.target_skip,
    )

    frozen_records = [by_pair_id[pid] for pid in result["pair_ids"]]

    manifest = {
        "selection_protocol_version": result["selection_protocol_version"],
        "source_file": str(args.period_derived_jsonl),
        "seed": result["seed"],
        "table_cap": result["table_cap"],
        "target_adjacent": result["target_adjacent"],
        "target_skip": result["target_skip"],
        "pair_ids": result["pair_ids"],
        "no_model_prediction_used": True,
    }
    manifest_path = output_dir / f"tatqa_period_primary_{SELECTION_PROTOCOL_VERSION}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    audit_path = output_dir / f"tatqa_period_primary_{SELECTION_PROTOCOL_VERSION}_audit.json"
    audit = dict(result["stats"])
    audit["by_construction_rule"] = {
        "adjacent": sum(1 for r in frozen_records if r["construction_rule"] == ADJACENT_RULE),
        "nonadjacent_skip": sum(1 for r in frozen_records if r["construction_rule"] == NONADJACENT_RULE),
    }
    pairs_per_table: dict[str, int] = {}
    for r in frozen_records:
        pairs_per_table[r["table_id"]] = pairs_per_table.get(r["table_id"], 0) + 1
    counts = sorted(pairs_per_table.values())
    audit["pairs_per_table"] = {
        "min": counts[0] if counts else 0,
        "max": counts[-1] if counts else 0,
        "median": counts[len(counts) // 2] if counts else 0,
        "unique_tables": len(counts),
    }
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")

    frozen_jsonl_path = output_dir / f"tatqa_period_primary_{SELECTION_PROTOCOL_VERSION}.jsonl"
    with frozen_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in frozen_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"selected {len(result['pair_ids'])} pairs ({audit['by_construction_rule']})")
    print(f"unique tables: {audit['pairs_per_table']['unique_tables']}, pairs/table min={audit['pairs_per_table']['min']} max={audit['pairs_per_table']['max']} median={audit['pairs_per_table']['median']}")
    print(f"manifest={manifest_path}")
    print(f"audit={audit_path}")
    print(f"frozen_pairs={frozen_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
