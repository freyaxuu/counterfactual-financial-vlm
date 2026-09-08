#!/usr/bin/env python3
"""Evaluate the Grounding-only Qwen3-VL LoRA (Grounding-SFT) on full-page
clean/target_value/header_address/irrelevant_value variants.

Reports the same QA metrics as evaluate_qwen3_vl_cf.py (Clean, TVFR, HFR,
ISR, CGS, P(T=0|C=1), P(H=0|C=1), P(T=0 or H=0|C=1), P(I=0|C=1)) for direct
comparability with Standard LoRA / CF-augmentation, PLUS the explicit
grounding metrics: row/column-header accuracy, bbox IoU, CellAcc@1, and
P(cell_correct | answer_correct) -- whether answers that look correct are
actually grounded in the right visual cell.

Uses variant-specific gold labels throughout (row/column/target-cell/bbox),
never the clean variant's labels for a counterfactual variant -- this matters
most for header_address_swap, where the correct target cell moves.

No oracle hints at eval time: all four variants are full-page images, same
as training. See scripts/train_grounding_lora.py for the training pipeline
this evaluates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.canonical_schema import REQUIRED_VARIANTS  # noqa: E402
from financial_vlm.data.grounding_target import (  # noqa: E402
    bbox01_to_pixels,
    bbox_iou,
    build_grounding_example,
    grounding_prompt,
    normalize_bbox_1000,
    normalize_header_text,
    parse_grounding_prediction,
    resolve_cell_at_point,
    summarize_grounding_metrics,
    validate_grounding_payload,
)
from financial_vlm.data.synfintabs_training import load_jsonl, select_records_by_group_prefix  # noqa: E402
from financial_vlm.evaluation.conditional_probabilities import (  # noqa: E402
    summarize_conditional_probabilities,
)
from financial_vlm.evaluation.pilot_accuracy import (  # noqa: E402
    canonical_variant,
    is_exact_correct,
    is_numeric_correct,
    summarize_accuracy,
)


DEFAULT_MODEL_PATH = Path("/path/to/models/Qwen3-VL-4B-Instruct")
DEFAULT_VARIANTS = sorted(REQUIRED_VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Dataset root_dir from prepare_synfintabs_min_train.py (contains images/ and manifest_*.jsonl).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="canonical_group_cf_v1 manifest, e.g. manifest_dev.jsonl. "
        "Do not use manifest_test.jsonl for model/checkpoint selection.",
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--adapter-path",
        type=Path,
        help="Optional PEFT LoRA adapter path (e.g. a train_grounding_lora.py final_adapter/).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", type=str, help="Defaults to --output-dir's name.")
    parser.add_argument("--seed", type=int, help="Training seed of the checkpoint being evaluated, for logging.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=DEFAULT_VARIANTS,
        help="Which of the 4 canonical_group_cf_v1 variants to evaluate.",
    )
    parser.add_argument(
        "--limit-groups",
        type=int,
        help="Evaluate only the first N sorted groups (smoke tests); each still gets every requested --variants.",
    )
    parser.add_argument(
        "--print-samples",
        type=int,
        default=0,
        help="Print this many decoded gold-vs-prediction comparisons to stdout (sanity-check aid).",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def prepare_output_dir(path: Path) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(model_path: Path, args: argparse.Namespace) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model_class = AutoModelForVision2Seq

    kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "trust_remote_code": True,
    }
    kwargs["torch_dtype"] = getattr(torch, args.torch_dtype) if args.torch_dtype != "auto" else "auto"
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = model_class.from_pretrained(model_path, local_files_only=True, **kwargs)
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path.expanduser().resolve(),
            local_files_only=True,
        )
    model.eval()
    return model, processor, torch


def load_image(image_path: Path) -> Any:
    from PIL import Image

    with Image.open(image_path) as loaded:
        return loaded.convert("RGB")


def generate_prediction(
    model: Any,
    processor: Any,
    torch: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if hasattr(model, "device"):
        inputs = inputs.to(model.device)

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0}
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated = generated[:, input_len:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def build_result_row(
    *,
    run_id: str,
    seed: int | None,
    checkpoint: str,
    group_id: str,
    table_id: str,
    variant: str,
    question: str,
    example: Any,
    image_size: tuple[int, int],
    evidence_units: list[dict[str, Any]],
    prediction_raw: str,
) -> dict[str, Any]:
    parsed = parse_grounding_prediction(prediction_raw)

    prediction_answer = parsed.answer or ""
    exact_correct = is_exact_correct(prediction_answer, example.answer_raw) if prediction_answer else False
    numeric_correct = is_numeric_correct(prediction_answer, example.answer_raw) if prediction_answer else False

    row_correct = normalize_header_text(parsed.row) == normalize_header_text(example.row_header)
    column_correct = normalize_header_text(parsed.column) == normalize_header_text(example.column_header)

    predicted_cell_id: str | None = None
    bbox_iou_value = 0.0
    if parsed.bbox_valid and parsed.bbox is not None:
        cx = (parsed.bbox[0] + parsed.bbox[2]) / 2
        cy = (parsed.bbox[1] + parsed.bbox[3]) / 2
        predicted_cell_id = resolve_cell_at_point(cx, cy, evidence_units, target_id=example.target_cell_id)
        bbox_iou_value = bbox_iou(parsed.bbox, example.bbox_1000)
    cell_correct = predicted_cell_id == example.target_cell_id

    gold_bbox_pixel = bbox01_to_pixels(example.bbox_normalised_01, image_size[0], image_size[1])

    return {
        "run_id": run_id,
        "seed": seed,
        "checkpoint": checkpoint,
        "group_id": group_id,
        "table_id": table_id,
        "variant": canonical_variant(variant),
        "question": question,
        "gold_answer": example.answer_raw,
        "prediction_raw": prediction_raw,
        "prediction_answer": prediction_answer,
        "answer_correct": numeric_correct,
        "exact_correct": exact_correct,
        "numeric_correct": numeric_correct,
        "gold_row": example.row_header,
        "predicted_row": parsed.row,
        "row_correct": row_correct,
        "gold_column": example.column_header,
        "predicted_column": parsed.column,
        "column_correct": column_correct,
        "gold_target_cell_id": example.target_cell_id,
        "gold_bbox_pixel": list(gold_bbox_pixel),
        "gold_bbox_normalized": list(example.bbox_1000),
        "predicted_bbox": list(parsed.bbox) if parsed.bbox is not None else None,
        "predicted_bbox_valid": parsed.bbox_valid,
        "bbox_iou": bbox_iou_value,
        "predicted_cell_id": predicted_cell_id,
        "cell_correct": cell_correct,
        # Compatibility fields for financial_vlm.evaluation.pilot_accuracy /
        # conditional_probabilities, which key off "setting"/"prediction"/"target_answer".
        "setting": "full_page",
        "prediction": prediction_answer,
        "target_answer": example.answer_raw,
    }


def print_sample(row: dict[str, Any]) -> None:
    print(f"--- group={row['group_id']} variant={row['variant']} ---")
    print(f"  answer:  gold={row['gold_answer']!r:20} pred={row['prediction_answer']!r:20} correct={row['answer_correct']}")
    print(f"  row:     gold={row['gold_row']!r:20} pred={row['predicted_row']!r:20} correct={row['row_correct']}")
    print(f"  column:  gold={row['gold_column']!r:20} pred={row['predicted_column']!r:20} correct={row['column_correct']}")
    print(f"  bbox:    gold={row['gold_bbox_normalized']} pred={row['predicted_bbox']} iou={row['bbox_iou']:.3f}")
    print(f"  cell:    gold={row['gold_target_cell_id']!r} pred={row['predicted_cell_id']!r} correct={row['cell_correct']}")


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = prepare_output_dir(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    run_id = args.run_id or output_dir.name
    checkpoint = str(args.adapter_path) if args.adapter_path is not None else str(args.model_path)

    variants = sorted(set(args.variants))
    group_records = load_jsonl(manifest_path)
    if not group_records:
        raise ValueError(f"No records found in manifest: {manifest_path}")
    if args.limit_groups is not None:
        group_records = select_records_by_group_prefix(group_records, group_count=args.limit_groups)

    model, processor, torch = load_model(args.model_path.expanduser().resolve(), args)

    rows: list[dict[str, Any]] = []
    failed = 0
    skipped_invalid = 0
    total_items = len(group_records) * len(variants)
    completed = 0
    printed = 0

    with predictions_path.open("w", encoding="utf-8") as output:
        for record in group_records:
            group_id = str(record["group_id"])
            table_id = str(record.get("document_id", ""))
            question = str(record["question"])

            for variant in variants:
                completed += 1
                payload = record["variants"][variant]

                reason = validate_grounding_payload(payload)
                if reason is not None:
                    skipped_invalid += 1
                    continue
                example = build_grounding_example(group_id, variant, question, payload)

                prediction_raw = ""
                error: str | None = None
                image_size = (0, 0)
                try:
                    image_path = data_root / example.image_path
                    image = load_image(image_path)
                    image_size = image.size
                    prediction_raw = generate_prediction(
                        model,
                        processor,
                        torch,
                        image,
                        grounding_prompt(question),
                        args.max_new_tokens,
                        args.temperature,
                    )
                except Exception as exc:  # keep the sample counted, never drop it silently
                    error = f"{type(exc).__name__}: {exc}"
                    failed += 1

                row = build_result_row(
                    run_id=run_id,
                    seed=args.seed,
                    checkpoint=checkpoint,
                    group_id=group_id,
                    table_id=table_id,
                    variant=variant,
                    question=question,
                    example=example,
                    image_size=image_size if image_size != (0, 0) else (1, 1),
                    evidence_units=list(payload["evidence_units"]),
                    prediction_raw=prediction_raw,
                )
                row["error"] = error
                output.write(json.dumps(row, sort_keys=True) + "\n")
                output.flush()
                rows.append(row)

                if printed < args.print_samples:
                    print_sample(row)
                    printed += 1

                if completed % 25 == 0 or completed == total_items:
                    print(f"completed={completed}/{total_items} failed={failed}", flush=True)

    qa_summary = summarize_accuracy(rows)
    conditional_summary = summarize_conditional_probabilities(rows, correctness="numeric")
    grounding_summary = summarize_grounding_metrics(rows)

    metrics = {
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "run_id": run_id,
        "seed": args.seed,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "variants": variants,
        "limit_groups": args.limit_groups,
        "git_commit": git_commit(),
        "predictions_path": str(predictions_path),
        "total_records": total_items,
        "failed_records": failed,
        "skipped_invalid_records": skipped_invalid,
        "qa_metrics": qa_summary,
        "conditional_probabilities": conditional_summary,
        "grounding_metrics_by_variant": grounding_summary,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(json.dumps(qa_summary.get("counterfactual_test_by_setting", {}), indent=2, sort_keys=True))
    print(json.dumps(grounding_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
