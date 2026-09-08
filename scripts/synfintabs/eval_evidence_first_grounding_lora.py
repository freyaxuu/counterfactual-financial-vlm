#!/usr/bin/env python3
"""Evaluate the Evidence-first Grounding LoRA (bbox generated before answer)
on full-page clean/target_value/header_address/irrelevant_value variants.

Reports the same QA metrics as evaluate_qwen3_vl_cf.py / eval_grounding_lora.py
(Clean, TVFR, HFR, ISR, CGS, P(T=0|C=1), P(H=0|C=1), P(T=0 or H=0|C=1),
P(I=0|C=1)) for direct comparability, plus: CellAcc@1, bbox IoU, and the
full answer x cell 2x2 confusion analysis (P(cell|answer), P(answer|cell),
P(answer|cell_wrong)) -- this target has no row/column supervision, so no
row/column accuracy is reported.

For header_address_swap specifically, buckets every example into one of
A/B/C/D (correct-cell x correct-answer) and saves group_ids per bucket for
manual inspection.

Ends by printing the Case A/B/C decision-rule interpretation (comparing
against Standard LoRA, Grounding-only answer-first, and CF-augmentation
reference numbers) and the two final comparison tables. Does not launch any
further training -- interpretation only.

Uses variant-specific gold labels throughout; never the clean variant's
labels for a counterfactual variant. No oracle hints at eval time.
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
    categorize_hfr_errors,
    evidence_first_prompt,
    parse_evidence_first_prediction,
    resolve_cell_at_point,
    summarize_evidence_first_grounding_metrics,
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
DEFAULT_GROUNDING_ANSWER_FIRST_METRICS = Path(
    "~/runs/cf-vlm-financial-grounding/grounding_eval_dev_lr2e-4/metrics.json"
).expanduser()


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
        help="Optional PEFT LoRA adapter path (e.g. a train_evidence_first_grounding_lora.py final_adapter/).",
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
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    # Decision-rule / comparison-table reference values (defaults from the existing report).
    parser.add_argument(
        "--grounding-answer-first-metrics",
        type=Path,
        default=DEFAULT_GROUNDING_ANSWER_FIRST_METRICS,
        help="Path to eval_grounding_lora.py's dev metrics.json for the answer-first 2e-4 run "
        "(used for the decision rule and comparison tables). Skipped if missing.",
    )
    parser.add_argument("--ref-standard-clean", type=float, default=0.950)
    parser.add_argument("--ref-standard-tvfr", type=float, default=0.970)
    parser.add_argument("--ref-standard-hfr", type=float, default=0.905)
    parser.add_argument("--ref-standard-isr", type=float, default=0.955)
    parser.add_argument("--ref-standard-cgs", type=float, default=0.9429)
    parser.add_argument("--ref-cfaug-clean", type=float, default=0.960)
    parser.add_argument("--ref-cfaug-tvfr", type=float, default=0.965)
    parser.add_argument("--ref-cfaug-hfr", type=float, default=0.970)
    parser.add_argument("--ref-cfaug-isr", type=float, default=0.960)
    parser.add_argument("--ref-cfaug-cgs", type=float, default=0.9650)
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
    parsed = parse_evidence_first_prediction(prediction_raw)

    prediction_answer = parsed.answer or ""
    exact_correct = is_exact_correct(prediction_answer, example.answer_raw) if prediction_answer else False
    numeric_correct = is_numeric_correct(prediction_answer, example.answer_raw) if prediction_answer else False

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
        "gold_target_cell_id": example.target_cell_id,
        "gold_bbox_pixel": list(gold_bbox_pixel),
        "gold_bbox_normalized": list(example.bbox_1000),
        "predicted_bbox": list(parsed.bbox) if parsed.bbox is not None else None,
        "predicted_bbox_valid": parsed.bbox_valid,
        "bbox_before_answer": parsed.bbox_before_answer,
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
    print(f"  bbox:    gold={row['gold_bbox_normalized']} pred={row['predicted_bbox']} iou={row['bbox_iou']:.3f}")
    print(f"  cell:    gold={row['gold_target_cell_id']!r} pred={row['predicted_cell_id']!r} correct={row['cell_correct']}")
    print(f"  order:   bbox_before_answer={row['bbox_before_answer']}")


def print_decision_rule(
    *,
    new_clean: float,
    new_hfr: float,
    new_cgs: float,
    new_p_target_fail: float,
    grounding_answer_first: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    print("\n=== Decision rule interpretation ===")
    print(f"Evidence-first:              Clean={new_clean:.3f} HFR={new_hfr:.3f} CGS={new_cgs:.4f} P(T=0orH=0|C=1)={new_p_target_fail:.4f}")
    if grounding_answer_first is not None:
        print(
            f"Grounding-only answer-first: Clean={grounding_answer_first['clean']:.3f} "
            f"HFR={grounding_answer_first['hfr']:.3f} CGS={grounding_answer_first['cgs']:.4f} "
            f"P(T=0orH=0|C=1)={grounding_answer_first['p_target_fail']:.4f}"
        )
    print(f"Standard LoRA:               HFR={args.ref_standard_hfr:.3f} CGS={args.ref_standard_cgs:.4f}")
    print(f"CF-augmentation:             HFR={args.ref_cfaug_hfr:.3f} CGS={args.ref_cfaug_cgs:.4f}")

    # Case C first: the most surprising outcome, checked before anything else.
    if new_cgs >= args.ref_cfaug_cgs - 0.005 or new_hfr >= args.ref_cfaug_hfr - 0.01:
        print(
            "\nCase C: Evidence-first unexpectedly matches or exceeds CF-augmentation.\n"
            "Flag this result for manual inspection before any further training."
        )
        return

    if grounding_answer_first is not None:
        clean_comparable = new_clean >= grounding_answer_first["clean"] - 0.02
        hfr_improved = new_hfr >= grounding_answer_first["hfr"] + 0.02
        fail_improved = new_p_target_fail <= grounding_answer_first["p_target_fail"] - 0.02
        if clean_comparable and (hfr_improved or fail_improved):
            print(
                "\nCase A: Evidence-first clearly improves over answer-first grounding.\n"
                "Evidence ordering appears important.\n"
                "Proceed next to CF + Evidence-first Grounding."
            )
            return

    print(
        "\nCase B: Evidence-first remains similar to or worse than answer-first.\n"
        "Clean grounding supervision alone does not train counterfactual semantic re-binding.\n"
        "Proceed next to CF + Evidence-first Grounding to test whether explicit localization adds "
        "value once counterfactual re-binding examples are present."
    )


def print_comparison_tables(
    *,
    new_clean: float,
    new_tvfr: float,
    new_hfr: float,
    new_isr: float,
    new_cgs: float,
    new_p_target_fail: float,
    new_clean_cellacc: float | None,
    new_hfr_cellacc: float | None,
    new_hfr_p_ans_given_cell: float | None,
    grounding_answer_first: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    print("\n=== Comparison table: QA metrics ===")
    header = f"{'Model':<30}{'Clean':>8}{'TVFR':>8}{'HFR':>8}{'ISR':>8}{'CGS':>9}{'AnyTargetFail':>15}"
    print(header)
    print(f"{'Standard LoRA':<30}{args.ref_standard_clean:>8.3f}{args.ref_standard_tvfr:>8.3f}{args.ref_standard_hfr:>8.3f}{args.ref_standard_isr:>8.3f}{args.ref_standard_cgs:>9.4f}{'--':>15}")
    if grounding_answer_first is not None:
        g = grounding_answer_first
        print(
            f"{'Grounding-only answer-first':<30}{g['clean']:>8.3f}{g['tvfr']:>8.3f}{g['hfr']:>8.3f}"
            f"{g['isr']:>8.3f}{g['cgs']:>9.4f}{g['p_target_fail']:>15.4f}"
        )
    print(
        f"{'Evidence-first Grounding':<30}{new_clean:>8.3f}{new_tvfr:>8.3f}{new_hfr:>8.3f}"
        f"{new_isr:>8.3f}{new_cgs:>9.4f}{new_p_target_fail:>15.4f}"
    )
    print(f"{'CF-augmentation':<30}{args.ref_cfaug_clean:>8.3f}{args.ref_cfaug_tvfr:>8.3f}{args.ref_cfaug_hfr:>8.3f}{args.ref_cfaug_isr:>8.3f}{args.ref_cfaug_cgs:>9.4f}{'--':>15}")

    print("\n=== Comparison table: grounding metrics ===")
    header2 = f"{'Model':<30}{'Clean CellAcc':>16}{'HFR CellAcc':>14}{'HFR P(ans|cell)':>18}"
    print(header2)
    if grounding_answer_first is not None and grounding_answer_first.get("clean_cellacc") is not None:
        print(
            f"{'Grounding answer-first':<30}{grounding_answer_first['clean_cellacc']:>16.3f}"
            f"{grounding_answer_first['hfr_cellacc']:>14.3f}"
            f"{'n/a':>18}"
        )
    fmt = lambda v: f"{v:.3f}" if v is not None else "n/a"  # noqa: E731
    print(
        f"{'Evidence-first Grounding':<30}{fmt(new_clean_cellacc):>16}{fmt(new_hfr_cellacc):>14}"
        f"{fmt(new_hfr_p_ans_given_cell):>18}"
    )


def load_grounding_answer_first_reference(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        print(f"NOTE: --grounding-answer-first-metrics not found at {path}; skipping that reference in the report.")
        return None
    data = json.loads(path.read_text())
    cf = data["qa_metrics"]["counterfactual_test_by_setting"]["full_page"]
    cond = data["conditional_probabilities"]["by_setting"]["full_page"]["probabilities"]
    grounding = data.get("grounding_metrics_by_variant", {})
    clean_grounding = grounding.get("clean", {})
    hfr_grounding = grounding.get("header_address_swap", {})
    return {
        "clean": cf["clean_accuracy_numeric"],
        "tvfr": cf["target_value_following_rate_numeric"],
        "hfr": cf["header_following_rate_numeric"],
        "isr": cf["irrelevant_stability_rate_numeric"],
        "cgs": cf["counterfactual_grounding_score_numeric"],
        "p_target_fail": cond["P(T=0_or_H=0|C=1)"],
        "clean_cellacc": clean_grounding.get("cell_acc_at_1"),
        "hfr_cellacc": hfr_grounding.get("cell_acc_at_1"),
    }


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
                        evidence_first_prompt(question),
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
    grounding_summary = summarize_evidence_first_grounding_metrics(rows)
    hfr_error_categories = categorize_hfr_errors(rows, variant="header_address_swap")

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
        "hfr_error_categories": hfr_error_categories,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(json.dumps(qa_summary.get("counterfactual_test_by_setting", {}), indent=2, sort_keys=True))
    print(json.dumps(grounding_summary, indent=2, sort_keys=True))

    print("\n=== HFR (header_address_swap) error categories ===")
    hfr_grounding = grounding_summary.get("header_address_swap", {})
    print(f"HFR answer accuracy: {hfr_grounding.get('answer_accuracy')}")
    print(f"HFR CellAcc@1: {hfr_grounding.get('cell_acc_at_1')}")
    print(f"HFR P(answer_correct|cell_correct): {hfr_grounding.get('p_answer_correct_given_cell_correct')}")
    print(f"HFR P(answer_correct|cell_wrong): {hfr_grounding.get('p_answer_correct_given_cell_wrong')}")
    for category, count in hfr_error_categories["counts"].items():
        print(f"  {category}: {count} groups -- {hfr_error_categories['group_ids'][category]}")

    cf = qa_summary["counterfactual_test_by_setting"]["full_page"]
    cond = conditional_summary["by_setting"]["full_page"]["probabilities"]
    clean_grounding = grounding_summary.get("clean", {})

    grounding_answer_first = load_grounding_answer_first_reference(args.grounding_answer_first_metrics)

    required_for_decision_rule = (
        cf["clean_accuracy_numeric"],
        cf["header_following_rate_numeric"],
        cf["counterfactual_grounding_score_numeric"],
        cond["P(T=0_or_H=0|C=1)"],
    )
    if any(value is None for value in required_for_decision_rule):
        print(
            "\nNOTE: skipping decision-rule interpretation and comparison tables -- "
            "HFR/CGS/P(T=0 or H=0|C=1) require the header_address_swap (and target_value) "
            "variants to have been evaluated (got --variants="
            f"{variants}). Re-run with the default --variants (all four) for the real report."
        )
        return 0

    print_decision_rule(
        new_clean=cf["clean_accuracy_numeric"],
        new_hfr=cf["header_following_rate_numeric"],
        new_cgs=cf["counterfactual_grounding_score_numeric"],
        new_p_target_fail=cond["P(T=0_or_H=0|C=1)"],
        grounding_answer_first=grounding_answer_first,
        args=args,
    )
    print_comparison_tables(
        new_clean=cf["clean_accuracy_numeric"],
        new_tvfr=cf["target_value_following_rate_numeric"],
        new_hfr=cf["header_following_rate_numeric"],
        new_isr=cf["irrelevant_stability_rate_numeric"],
        new_cgs=cf["counterfactual_grounding_score_numeric"],
        new_p_target_fail=cond["P(T=0_or_H=0|C=1)"],
        new_clean_cellacc=clean_grounding.get("cell_acc_at_1"),
        new_hfr_cellacc=hfr_grounding.get("cell_acc_at_1"),
        new_hfr_p_ans_given_cell=hfr_grounding.get("p_answer_correct_given_cell_correct"),
        grounding_answer_first=grounding_answer_first,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
