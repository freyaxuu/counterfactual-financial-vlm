#!/usr/bin/env python3
"""Evaluate Qwen3-VL on canonical grounded financial QA manifests."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.evaluation.pilot_accuracy import (  # noqa: E402
    AccuracySummary,
    is_exact_correct,
    is_numeric_correct,
)
from financial_vlm.training.standard_qa import standard_qa_prompt  # noqa: E402


DEFAULT_DATA_ROOT = Path(
    "/path/to/runs/cf-vlm-financial-grounding/tatdqa_test_direct_extract_seed_20260802_n200_3d6e185"
)
DEFAULT_MODEL_PATH = Path("/path/to/models/Qwen3-VL-4B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, help="Defaults to <data-root>/manifest.jsonl.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, help="Optional PEFT LoRA adapter path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/runs/cf-vlm-financial-grounding/qwen3_vl_4b_tatdqa_lookup_eval"),
    )
    parser.add_argument("--limit", type=int, help="Limit records for smoke tests.")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-style",
        choices=["hinted", "standard_qa"],
        default="hinted",
        help="Use standard_qa for clean answer-only LoRA baselines without evidence hints.",
    )
    return parser.parse_args()


def load_manifest(path: Path, limit: int | None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    return records


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
    if args.torch_dtype != "auto":
        kwargs["torch_dtype"] = getattr(torch, args.torch_dtype)
    else:
        kwargs["torch_dtype"] = "auto"
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = model_class.from_pretrained(
        model_path,
        local_files_only=True,
        **kwargs,
    )
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path.expanduser().resolve(),
            local_files_only=True,
        )
    model.eval()
    return model, processor, torch


def prompt_for(record: Mapping[str, Any], prompt_style: str = "hinted") -> str:
    question = str(record["question"])
    if prompt_style == "standard_qa":
        return standard_qa_prompt(question)

    answer = record.get("answer") or {}
    metric = answer.get("metric")
    period = answer.get("period")
    scale = answer.get("scale")
    hints = []
    if metric:
        hints.append(f"metric: {metric}")
    if period:
        hints.append(f"period: {period}")
    if scale:
        hints.append(f"scale: {scale}")

    context = "Read the financial document image and answer the question."
    if prompt_style == "standard_qa":
        hint_line = ""
    else:
        hint_line = f"\nEvidence address hints: {'; '.join(hints)}." if hints else ""
    return (
        f"{context}{hint_line}\n"
        "Return only the answer value, without explanation or units unless the value itself includes a percent sign.\n"
        f"Question: {question}"
    )


def build_messages(image: Any, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def encode_inputs(processor: Any, messages: list[dict[str, Any]], image: Any) -> Any:
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )


def move_inputs_to_device(inputs: Any, model: Any) -> Any:
    if hasattr(model, "device"):
        return inputs.to(model.device)
    return inputs


def generate_answer(
    model: Any,
    processor: Any,
    torch: Any,
    image: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = build_messages(image, prompt)
    inputs = encode_inputs(processor, messages, image)
    inputs = move_inputs_to_device(inputs, model)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    generated = generated[:, input_len:]
    decoded = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


def iter_eval_items(
    data_root: Path,
    records: Iterable[dict[str, Any]],
) -> Iterable[tuple[dict[str, Any], str, Any]]:
    from PIL import Image

    for record in records:
        page_path = data_root / record["image_path"]
        with Image.open(page_path) as loaded:
            page_image = loaded.convert("RGB")
        yield record, "full_page", page_image.copy()


def accuracy_summary(items: list[Mapping[str, Any]]) -> dict[str, Any]:
    return AccuracySummary(
        total=len(items),
        exact_correct=sum(1 for item in items if bool(item["exact_correct"])),
        numeric_correct=sum(1 for item in items if bool(item["numeric_correct"])),
    ).to_json()


def summarize_predictions(predictions: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_setting: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_setting_scale: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        setting = str(prediction["setting"])
        scale = str(prediction.get("scale") or "none")
        by_setting[setting].append(prediction)
        by_setting_scale[(setting, scale)].append(prediction)
    return {
        "overall": accuracy_summary(predictions),
        "by_setting": {
            setting: accuracy_summary(items)
            for setting, items in sorted(by_setting.items())
        },
        "by_setting_scale": {
            f"{setting}/{scale}": accuracy_summary(items)
            for (setting, scale), items in sorted(by_setting_scale.items())
        },
    }


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    manifest_path = (args.manifest or (data_root / "manifest.jsonl")).expanduser().resolve()
    output_dir = prepare_output_dir(args.output_dir)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"

    records = load_manifest(manifest_path, args.limit)
    if not records:
        raise ValueError(f"No records selected from manifest: {manifest_path}")

    model, processor, torch = load_model(args.model_path.expanduser().resolve(), args)
    predictions: list[dict[str, Any]] = []

    total_items = len(records)
    completed = 0
    with predictions_path.open("w", encoding="utf-8") as output:
        for record, setting, image in iter_eval_items(
            data_root,
            records,
        ):
            completed += 1
            answer = record["answer"]
            target = str(answer["raw"])
            prediction = generate_answer(
                model,
                processor,
                torch,
                image,
                prompt_for(record, args.prompt_style),
                args.max_new_tokens,
                args.temperature,
            )
            result = {
                "setting": setting,
                "group_id": record["group_id"],
                "document_id": record["document_id"],
                "question": record["question"],
                "target_answer": target,
                "prediction": prediction,
                "exact_correct": is_exact_correct(prediction, target),
                "numeric_correct": is_numeric_correct(prediction, target),
                "scale": answer.get("scale"),
                "metric": answer.get("metric"),
                "period": answer.get("period"),
                "image_path": record["image_path"],
                "input_image_size": list(image.size),
                "evidence": record["evidence"],
            }
            output.write(json.dumps(result, sort_keys=True) + "\n")
            output.flush()
            predictions.append(result)

            if completed % 25 == 0 or completed == total_items:
                print(f"completed={completed}/{total_items}", flush=True)

    metrics = {
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path) if args.adapter_path is not None else None,
        "data_root": str(data_root),
        "manifest_path": str(manifest_path),
        "settings": ["full_page"],
        "prompt_style": args.prompt_style,
        "limit": args.limit,
        "predictions_path": str(predictions_path),
        **summarize_predictions(predictions),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(f"predictions={predictions_path}")
    print(f"metrics={metrics_path}")
    print(json.dumps(metrics["by_setting"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
