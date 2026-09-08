#!/usr/bin/env python3
"""Build the unique-question index for the frozen `company_representative_v1`
benchmark, so it can be run through `scripts/common/evaluate_qwen3_vl_natural_pairs.py`
UNCHANGED -- one row per fact (no A/B sides, unlike company_confusion_v1).

Requires no private-package access itself -- reads the already-frozen
`company_representative_v1.jsonl` (real question/answer data) and writes a
reshaped copy of the same private data. Keep both off Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-representative-v1", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/path/to/private-dataset"))
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def image_path_for(dataset_root: Path, document_id: str, page_id: str) -> str:
    return str(dataset_root / "files" / document_id / "pages" / page_id / "image.png")


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory already exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    facts = load_jsonl(args.company_representative_v1.expanduser().resolve())
    if not facts:
        raise ValueError(f"No records found in: {args.company_representative_v1}")

    records = []
    missing_images = 0
    for fact in facts:
        image_path = image_path_for(dataset_root, fact["document_id"], fact["page_id"])
        if not Path(image_path).exists():
            missing_images += 1
        records.append(
            {
                "question_key": fact["fact_id"],
                "document_id": fact["document_id"],
                "question": fact["question"],
                "answer": {"raw": fact["gold_answer"]},
                "image_path": image_path,
                "group_id": fact["fact_id"],  # required field name for build_prediction_record reuse
            }
        )

    unique_path = output_dir / "unique_questions.jsonl"
    with unique_path.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda r: r["question_key"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"facts: {len(facts)}")
    print(f"missing images (should be 0): {missing_images}")
    print(f"unique_questions={unique_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
