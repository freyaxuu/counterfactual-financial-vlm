#!/usr/bin/env python3
"""Run one trained checkpoint over the deduplicated TAT-QA Natural Pair
unique-question index, caching one prediction per unique question.

Reuses evaluate_qwen3_vl_cf.py's model loading, prompt construction, image
loading, generation, and prediction-recording logic UNCHANGED (same
imports: load_model / prompt_for / load_image / generate_answer /
git_commit / prepare_output_dir, plus build_prediction_record from
pilot_accuracy) so decoding and answer normalization stay byte-identical
to the existing clean/TVFR/HFR/ISR/CGS evaluation. Only the input pipeline
differs: a flat per-question unique-question index (built by
build_tatqa_natural_pairs_unique_questions.py) instead of a
canonical_group_cf_v1 CF-variant manifest -- there is no CF-variant loop
and no group flattening here.

Each question in the index is sent to the model independently -- pair
members are never combined into one prompt (pairing only happens later, in
a separate aggregation step over the cached predictions.jsonl).

This script does not compute CGS or any aggregate accuracy: per Task
instructions, headline metrics for this benchmark are Individual
Accuracy / Pair Accuracy / capture rates, computed separately per subset
by scripts/aggregate_tatqa_natural_pairs_results.py from the raw
predictions this script writes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts" / "common"
for path in (SRC_ROOT, SCRIPTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_qwen3_vl_cf import (  # noqa: E402
    generate_answer,
    git_commit,
    load_image,
    load_model,
    prepare_output_dir,
    prompt_for,
)
from financial_vlm.evaluation.pilot_accuracy import build_prediction_record  # noqa: E402

DEFAULT_MODEL_PATH = Path("/path/to/models/Qwen3-VL-4B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unique-questions", type=Path, required=True, help="unique_questions.jsonl from build_tatqa_natural_pairs_unique_questions.py")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, help="Optional PEFT LoRA adapter path. Omit for a base-model-only run.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt-style", choices=["hinted", "standard_qa"], default="standard_qa")
    parser.add_argument("--limit", type=int, help="Smoke-test: only run the first N unique questions (sorted order).")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "run_metadata.json"

    records = load_jsonl(args.unique_questions.expanduser().resolve())
    if not records:
        raise ValueError(f"No records found in: {args.unique_questions}")
    records.sort(key=lambda r: r["question_key"])
    if args.limit is not None:
        records = records[: args.limit]

    missing_images = [r["question_key"] for r in records if not r.get("image_path")]
    if missing_images:
        raise ValueError(f"{len(missing_images)} unique questions have no resolved image_path (e.g. {missing_images[:3]}) -- fix the image manifest first.")

    model, processor, torch = load_model(args.model_path.expanduser().resolve(), args)

    total_items = len(records)
    completed = 0
    failed = 0
    with predictions_path.open("w", encoding="utf-8") as output:
        for record in records:
            completed += 1
            prediction: str | None = None
            error: str | None = None
            try:
                image = load_image(Path(record["image_path"]))
                prediction = generate_answer(
                    model,
                    processor,
                    torch,
                    image,
                    prompt_for(record, args.prompt_style),
                    args.max_new_tokens,
                    args.temperature,
                )
            except Exception as exc:  # keep the sample counted, never drop it silently
                error = f"{type(exc).__name__}: {exc}"
                failed += 1

            result = build_prediction_record(record, setting="natural_pairs", prediction=prediction, error=error)
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()

            if completed % 100 == 0 or completed == total_items:
                print(f"completed={completed}/{total_items} failed={failed}", flush=True)

    metadata = {
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "unique_questions_path": str(args.unique_questions),
        "prompt_style": args.prompt_style,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "limit": args.limit,
        "git_commit": git_commit(),
        "predictions_path": str(predictions_path),
        "total_records": total_items,
        "failed_records": failed,
    }
    metrics_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"predictions={predictions_path}")
    print(f"run_metadata={metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
