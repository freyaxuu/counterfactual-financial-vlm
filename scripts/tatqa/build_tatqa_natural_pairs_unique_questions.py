#!/usr/bin/env python3
"""Build the deduplicated unique-question index for the frozen TAT-QA
Natural Pair benchmark, per Task section 3 (Deduplicate inference).

Reads the three frozen subsets (human_authored, period_derived,
metric_candidate), extracts both sides of every pair via
``financial_vlm.evaluation.tatqa_natural_pairs_eval.pair_sides``, and
de-duplicates by ``question_key`` (document_id, question_text,
target_cell_id) so a shared question instance is only ever sent to a model
once per checkpoint. Attaches each unique question's resolved image path
from the image manifest built by build_tatqa_natural_pairs_image_manifest.py.

Also reports coverage statistics (total pairs, unique questions, unique
tables, unique documents) per subset, per Task section 3/13.

Does not run any model. Does not modify the frozen benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.tatqa_natural_pairs import PROTOCOL_VERSION  # noqa: E402
from financial_vlm.evaluation.tatqa_natural_pairs_eval import pair_sides  # noqa: E402


def subset_files(protocol_version: str) -> dict[str, str]:
    return {
        "human_authored": f"tatqa_natural_pairs_human_authored_{protocol_version}.jsonl",
        "period_derived": f"tatqa_natural_pairs_period_derived_{protocol_version}.jsonl",
        "metric_candidate": f"tatqa_natural_pairs_metric_candidates_{protocol_version}.jsonl",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--natural-pairs-root", type=Path, required=True)
    parser.add_argument("--image-manifest", type=Path, required=True, help="image_manifest.jsonl from build_tatqa_natural_pairs_image_manifest.py")
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-version", default=PROTOCOL_VERSION, help="Frozen benchmark filename suffix (default: current library PROTOCOL_VERSION).")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_image_by_table_uid(manifest_path: Path) -> dict[str, str | None]:
    by_table_uid: dict[str, str | None] = {}
    for record in load_jsonl(manifest_path):
        by_table_uid[record["table_uid"]] = record["image_path"]
    return by_table_uid


def percentile(values: list[int], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(int(pct * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[idx]


def coverage_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    unique_questions: set[str] = set()
    unique_tables: set[str] = set()
    unique_documents: set[str] = set()
    pairs_per_table: dict[str, int] = {}
    for record in records:
        side_a, side_b = pair_sides(record)
        unique_questions.add(side_a.question_key)
        unique_questions.add(side_b.question_key)
        unique_tables.add(record["table_id"])
        unique_documents.add(record["document_id"])
        pairs_per_table[record["table_id"]] = pairs_per_table.get(record["table_id"], 0) + 1
    counts = list(pairs_per_table.values())
    return {
        "num_pairs": len(records),
        "num_unique_questions": len(unique_questions),
        "num_unique_tables": len(unique_tables),
        "num_unique_documents": len(unique_documents),
        "median_pairs_per_table": statistics.median(counts) if counts else None,
        "max_pairs_per_table": max(counts) if counts else None,
        "p25_pairs_per_table": percentile(counts, 0.25),
        "p75_pairs_per_table": percentile(counts, 0.75),
    }


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    image_by_table_uid = load_image_by_table_uid(args.image_manifest)

    subsets: dict[str, list[dict[str, Any]]] = {}
    for subset, filename in subset_files(args.protocol_version).items():
        records = [r for r in load_jsonl(args.natural_pairs_root / filename) if r["split"] == args.split]
        subsets[subset] = records

    unique_by_key: dict[str, dict[str, Any]] = {}
    unresolved_images = 0
    for subset, records in subsets.items():
        for record in records:
            for side in pair_sides(record):
                if side.question_key in unique_by_key:
                    continue
                image_path = image_by_table_uid.get(side.document_id)
                if image_path is None:
                    unresolved_images += 1
                unique_by_key[side.question_key] = {
                    "question_key": side.question_key,
                    "document_id": side.document_id,
                    "table_id": side.table_id,
                    "question": side.question,
                    "answer": {"raw": side.answer_raw},
                    "target_cell_id": side.target_cell_id,
                    "image_path": image_path,
                    "group_id": side.question_key,  # required field name for build_prediction_record reuse
                }

    unique_path = output_dir / "unique_questions.jsonl"
    with unique_path.open("w", encoding="utf-8") as handle:
        for record in sorted(unique_by_key.values(), key=lambda r: r["question_key"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    coverage = {subset: coverage_stats(records) for subset, records in subsets.items()}
    coverage_path = output_dir / "coverage_stats.json"
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")

    print(f"unique questions total: {len(unique_by_key)}")
    print(f"unresolved images (missing from image manifest): {unresolved_images}")
    for subset, stats in coverage.items():
        print(f"{subset}: {stats}")
    print(f"unique_questions={unique_path}")
    print(f"coverage_stats={coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
