#!/usr/bin/env python3
"""Build the deduplicated unique-question index for the frozen
`company_confusion_v1` benchmark (200 period/basis confusion pairs -- see
`docs/company-benchmark-diagnosis-report.md` section 16), so it can be run
through `scripts/common/evaluate_qwen3_vl_natural_pairs.py` UNCHANGED.

Each pair's two sides become two independent question records with the
exact field names that script (and `financial_vlm.evaluation.pilot_accuracy
.build_prediction_record`) expects: `question_key` (also used as
`group_id`), `document_id`, `question`, `answer.raw`, `image_path`. No
model is run here; this only reshapes the frozen benchmark.

Image paths are resolved directly (no manifest lookup needed, unlike the
TAT-QA Natural Pair build script this mirrors) since `company_confusion_v1`
records already carry `document_id`/`page_a`/`page_b`, and the on-disk
layout is documented in `docs/evolution-ai-datasets.md`:
`<dataset_root>/files/<document_id>/pages/<page_id>/image.png`.

Requires no private-package access itself -- reads the already-frozen
`company_confusion_v1.jsonl` (real question/answer data) and writes a
reshaped copy of the same private data. Keep both off Git.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--company-confusion-v1", type=Path, required=True, help="company_confusion_v1.jsonl")
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

    pairs = load_jsonl(args.company_confusion_v1.expanduser().resolve())
    if not pairs:
        raise ValueError(f"No records found in: {args.company_confusion_v1}")

    unique_by_key: dict[str, dict[str, Any]] = {}
    missing_images = 0
    for pair in pairs:
        for side, page_field, question_field, answer_field in (
            ("A", "page_a", "question_a", "gold_answer_a"),
            ("B", "page_b", "question_b", "gold_answer_b"),
        ):
            question_key = f"{pair['pair_id']}_{side}"
            image_path = image_path_for(dataset_root, pair["document_id"], pair[page_field])
            if not Path(image_path).exists():
                missing_images += 1
            unique_by_key[question_key] = {
                "question_key": question_key,
                "pair_id": pair["pair_id"],
                "pair_type": pair["pair_type"],
                "side": side,
                "document_id": pair["document_id"],
                "question": pair[question_field],
                "answer": {"raw": pair[answer_field]},
                "image_path": image_path,
                "group_id": question_key,  # required field name for build_prediction_record reuse
            }

    unique_path = output_dir / "unique_questions.jsonl"
    with unique_path.open("w", encoding="utf-8") as handle:
        for record in sorted(unique_by_key.values(), key=lambda r: r["question_key"]):
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    print(f"pairs: {len(pairs)}")
    print(f"unique questions: {len(unique_by_key)}")
    print(f"missing images (should be 0): {missing_images}")
    print(f"unique_questions={unique_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
