#!/usr/bin/env python3
"""Evaluate Qwen3-VL (optionally with a LoRA adapter) on a canonical_group_cf_v1
manifest, producing clean accuracy plus counterfactual-faithfulness metrics
(target-value-following rate, header-following rate, irrelevant-stability rate,
counterfactual-grounding-score) in the same shape the SynFinTabs pilot used.

Do not point --manifest at manifest_test.jsonl for model or checkpoint
selection -- that split is reserved for a final, one-time OOD report.
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
from financial_vlm.data.synfintabs_training import (  # noqa: E402
    flatten_group_records,
    load_jsonl,
    select_records_by_group_prefix,
)
from financial_vlm.evaluation.pilot_accuracy import (  # noqa: E402
    build_prediction_record,
    summarize_accuracy,
)
from financial_vlm.training.standard_qa import standard_qa_prompt  # noqa: E402


DEFAULT_MODEL_PATH = Path("/path/to/models/Qwen3-VL-4B-Instruct")
DEFAULT_VARIANTS = sorted(REQUIRED_VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Optional PEFT LoRA adapter path (e.g. a train_qwen3_vl_lora.py final_adapter/).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
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
        help="Evaluate only the first N sorted groups (smoke tests); "
        "each still gets every requested --variants.",
    )
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-style",
        choices=["hinted", "standard_qa"],
        default="standard_qa",
        help="Match the prompt style the adapter was trained with.",
    )
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


def prompt_for(record: dict[str, Any], prompt_style: str) -> str:
    question = str(record["question"])
    if prompt_style == "standard_qa":
        return standard_qa_prompt(question)

    answer = record.get("answer") or {}
    hints = [
        f"{label}: {value}"
        for label, value in (("metric", answer.get("metric")), ("period", answer.get("period")), ("scale", answer.get("scale")))
        if value
    ]
    hint_line = f"\nEvidence address hints: {'; '.join(hints)}." if hints else ""
    return (
        "Read the financial document image and answer the question."
        f"{hint_line}\n"
        "Return only the answer value, without explanation or units unless the value itself includes a percent sign.\n"
        f"Question: {question}"
    )


def load_image(image_path: Path) -> Any:
    from PIL import Image

    with Image.open(image_path) as loaded:
        return loaded.convert("RGB")


def generate_answer(
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


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    output_dir = prepare_output_dir(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    variants = sorted(set(args.variants))
    group_records = load_jsonl(manifest_path)
    if not group_records:
        raise ValueError(f"No records found in manifest: {manifest_path}")
    if args.limit_groups is not None:
        group_records = select_records_by_group_prefix(group_records, group_count=args.limit_groups)
    records = flatten_group_records(group_records, variants)
    if not records:
        raise ValueError(f"No records selected for variants={variants} from manifest: {manifest_path}")

    model, processor, torch = load_model(args.model_path.expanduser().resolve(), args)

    predictions: list[dict[str, Any]] = []
    failed = 0
    total_items = len(records)
    completed = 0
    with predictions_path.open("w", encoding="utf-8") as output:
        for record in records:
            completed += 1
            prediction: str | None = None
            error: str | None = None
            try:
                image = load_image(data_root / record["image_path"])
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

            result = build_prediction_record(record, setting="full_page", prediction=prediction, error=error)
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            predictions.append(result)

            if completed % 25 == 0 or completed == total_items:
                print(f"completed={completed}/{total_items} failed={failed}", flush=True)

    metrics = {
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "variants": variants,
        "prompt_style": args.prompt_style,
        "limit_groups": args.limit_groups,
        "git_commit": git_commit(),
        "predictions_path": str(predictions_path),
        "total_records": total_items,
        "failed_records": failed,
        **summarize_accuracy(predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(json.dumps(metrics["counterfactual_test_by_setting"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
