#!/usr/bin/env python3
"""Train the Grounding-only Qwen3-VL LoRA baseline (Grounding-SFT).

Tests whether explicit evidence-grounding supervision alone -- with NO
counterfactual training data -- reduces the residual header/address
grounding failures observed with Standard LoRA. Training uses clean
full-page images ONLY: no target-value/header/irrelevant-value
counterfactuals, no oracle evidence packets, no highlighted or cropped
images, no CF-pair or consistency losses. The only difference from Standard
LoRA is the supervision target: instead of just the answer, the assistant
target also includes the resolved row header, column header, and target-cell
bbox (normalized to [0, 1000]), in a fixed, deterministic format:

    <answer>...</answer>
    <row>...</row>
    <column>...</column>
    <bbox>x1 y1 x2 y2</bbox>

Loss is the ordinary autoregressive causal-LM loss over that target string
(prompt tokens masked, exactly as in train_qwen3_vl_lora.py) -- no separate
grounding loss term, no lambda weighting, no bbox regression loss.

Model loading, LoRA config, optimizer/scheduler, image preprocessing,
checkpointing, and seed handling are duplicated from train_qwen3_vl_lora.py
rather than imported, matching this repo's existing convention (see
train_qwen3_vl_cf_lora.py, train_qwen3_vl_compute_matched_clean_lora.py) of
self-contained scripts. Standard LoRA and CF-augmentation are untouched.

Smoke test / overfit sanity check: run with --limit-groups 20 (or 30-50),
then run eval_grounding_lora.py --limit-groups <same N> --print-samples 10
against the resulting adapter to inspect decoded predictions before
launching the full 1000-group run.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from financial_vlm.data.grounding_target import (  # noqa: E402
    GroundingExample,
    check_split_leakage,
    grounding_prompt,
    validate_and_build_examples,
)
from financial_vlm.data.synfintabs_training import load_jsonl  # noqa: E402
from financial_vlm.training.standard_qa import normalize_interval_strategy  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "synfintabs" / "synfintabs_train_dev_test_ood_cfv1_grounding_lora.yaml"
TRAIN_VARIANT = "clean"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, help="Override data.root_dir.")
    parser.add_argument("--train-manifest", type=Path, help="Override data.train_manifest.")
    parser.add_argument("--dev-manifest", type=Path, help="Override data.dev_manifest (leakage check only).")
    parser.add_argument("--test-manifest", type=Path, help="Override data.test_manifest (leakage check only).")
    parser.add_argument(
        "--skip-leakage-check",
        action="store_true",
        help="Skip the train/dev/test source-table leakage check (requires dev/test manifests to exist).",
    )
    parser.add_argument("--output-dir", type=Path, help="Override output.root_dir.")
    parser.add_argument("--model-path", type=Path, help="Override model.base_model_path.")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate.")
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        help="Override training.num_train_epochs. Useful with --limit-groups for an overfit "
        "sanity check, since the production epoch count won't reach many optimizer steps on a tiny subset.",
    )
    parser.add_argument("--limit-groups", type=int, help="Train on only the first N sorted groups (smoke tests).")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read training configs.") from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return loaded


def path_from_config(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        path = base / path
    return path.resolve()


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


def prepare_output_dir(path: Path, config_path: Path, config: Mapping[str, Any]) -> Path:
    output_dir = path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory exists and is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    metadata = {
        "git_commit": git_commit(),
        "config_path": str(config_path.resolve()),
        "experiment": config.get("experiment", {}),
        "model": config.get("model", {}),
        "data": config.get("data", {}),
        "training": config.get("training", {}),
        "lora": config.get("lora", {}),
        "baseline": "Grounding-only LoRA (Grounding-SFT): clean images only, no CF data",
        "objective": "causal_lm_loss(<answer>...</answer><row>...</row><column>...</column><bbox>x1 y1 x2 y2</bbox>)",
        "selection_metric": "numeric_accuracy",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_dir


def _source_index_from_group_id(group_id: str) -> str:
    parts = group_id.split("_")
    if len(parts) < 3 or parts[0] != "syn":
        raise ValueError(f"Unrecognized group_id format: {group_id!r}")
    return parts[1]


def print_validation_summary(stats: Mapping[str, int]) -> None:
    print("=== Grounding validation summary ===")
    print(f"  total_groups: {stats.get('total_groups', 0)}")
    print(f"  valid_groups: {stats.get('valid_groups', 0)}")
    for key in sorted(stats):
        if key in {"total_groups", "valid_groups"}:
            continue
        print(f"  {key}: {stats[key]}")


def build_messages(image: Any, question: str, target: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": grounding_prompt(question)},
            ],
        }
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": target})
    return messages


class GroundingQADataset:
    def __init__(self, examples: Sequence[GroundingExample], data_root: Path, processor: Any):
        self.examples = list(examples)
        self.data_root = data_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image
        import torch

        example = self.examples[index]
        image_path = Path(example.image_path).expanduser()
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")

        prompt_inputs = self.processor.apply_chat_template(
            build_messages(image, example.question),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            build_messages(image, example.question, example.target_text),
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = full_inputs["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        if prompt_len >= labels.shape[-1]:
            prompt_len = max(0, labels.shape[-1] - 1)
        labels[:prompt_len] = -100

        example_tensors: dict[str, Any] = {}
        for key, value in full_inputs.items():
            if torch.is_tensor(value) and key in {"input_ids", "attention_mask"}:
                example_tensors[key] = value.squeeze(0)
            else:
                example_tensors[key] = value
        example_tensors["labels"] = labels
        return example_tensors


class VisionQACollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        import torch
        from torch.nn.utils.rnn import pad_sequence

        batch = {
            "input_ids": pad_sequence(
                [example["input_ids"] for example in examples],
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                [example["attention_mask"] for example in examples],
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(
                [example["labels"] for example in examples],
                batch_first=True,
                padding_value=-100,
            ),
        }
        extra_keys = sorted(set().union(*(set(example) for example in examples)) - set(batch))
        for key in extra_keys:
            values = [example.get(key) for example in examples]
            if any(value is None for value in values):
                continue
            if all(torch.is_tensor(value) for value in values):
                try:
                    batch[key] = torch.cat(values, dim=0)
                except RuntimeError:
                    batch[key] = torch.stack(values, dim=0)
        return batch


def training_arguments(output_dir: Path, cfg: Mapping[str, Any]) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": float(cfg.get("num_train_epochs", 3)),
        "per_device_train_batch_size": int(cfg.get("per_device_train_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 16)),
        "learning_rate": float(cfg.get("learning_rate", 2e-4)),
        "weight_decay": float(cfg.get("weight_decay", 0.0)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.03)),
        "lr_scheduler_type": str(cfg.get("lr_scheduler_type", "cosine")),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "save_steps": int(cfg.get("save_steps", 100)),
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "bf16": bool(cfg.get("bf16", True)),
        "fp16": bool(cfg.get("fp16", False)),
        "gradient_checkpointing": bool(cfg.get("gradient_checkpointing", True)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 0)),
        "remove_unused_columns": False,
        "report_to": cfg.get("report_to", "none"),
        "save_strategy": str(cfg.get("save_strategy", "steps")),
        "seed": int(cfg.get("seed", 20260804)),
    }
    signature = inspect.signature(TrainingArguments)
    eval_strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_strategy_key] = normalize_interval_strategy(None, default="no")
    return TrainingArguments(**kwargs)


def load_model_and_processor(model_path: Path, model_cfg: Mapping[str, Any], lora_cfg: Mapping[str, Any]) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText

        model_class = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq

        model_class = AutoModelForVision2Seq

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "device_map": model_cfg.get("device_map", "auto"),
        "local_files_only": bool(model_cfg.get("local_files_only", True)),
    }
    torch_dtype = model_cfg.get("torch_dtype", "auto")
    kwargs["torch_dtype"] = getattr(torch, torch_dtype) if torch_dtype != "auto" else "auto"
    if model_cfg.get("attn_implementation"):
        kwargs["attn_implementation"] = model_cfg["attn_implementation"]

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=bool(model_cfg.get("local_files_only", True)),
    )
    model = model_class.from_pretrained(model_path, **kwargs)
    if bool(model_cfg.get("gradient_checkpointing", True)):
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("lora_alpha", 32)),
        lora_dropout=float(lora_cfg.get("lora_dropout", 0.05)),
        bias=str(lora_cfg.get("bias", "none")),
        task_type=str(lora_cfg.get("task_type", "CAUSAL_LM")),
        target_modules=list(lora_cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, processor


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = dict(config["training"])
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
    if args.num_train_epochs is not None:
        train_cfg["num_train_epochs"] = args.num_train_epochs
    config["training"] = train_cfg
    lora_cfg = config["lora"]
    output_cfg = config["output"]

    data_root = path_from_config(args.data_root or data_cfg["root_dir"])
    train_manifest = path_from_config(args.train_manifest or data_cfg["train_manifest"], data_root)
    global_seed = int(config["experiment"]["seed"])

    if not args.skip_leakage_check:
        dev_manifest = path_from_config(args.dev_manifest or data_cfg.get("dev_manifest", "manifest_dev.jsonl"), data_root)
        test_manifest = path_from_config(
            args.test_manifest or data_cfg.get("test_manifest", "manifest_test.jsonl"), data_root
        )
        leakage_counts = check_split_leakage(
            load_jsonl(train_manifest), load_jsonl(dev_manifest), load_jsonl(test_manifest)
        )
        print(f"leakage check passed: no overlap ({leakage_counts})")

    group_records = load_jsonl(train_manifest)
    if args.limit_groups is not None:
        keep_ids = sorted({str(r["group_id"]) for r in group_records})[: args.limit_groups]
        keep = set(keep_ids)
        group_records = [r for r in group_records if str(r["group_id"]) in keep]

    examples, stats = validate_and_build_examples(group_records, variant=TRAIN_VARIANT)
    print_validation_summary(stats)
    if not examples:
        raise ValueError("No valid groups survived grounding validation -- nothing to train on.")

    output_dir = prepare_output_dir(
        path_from_config(args.output_dir or output_cfg["root_dir"]),
        config_path,
        config,
    )
    (output_dir / "grounding_validation_summary.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    model_path = path_from_config(args.model_path or model_cfg["base_model_path"])

    from transformers import set_seed

    set_seed(global_seed)
    model, processor = load_model_and_processor(model_path, model_cfg, lora_cfg)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0)

    train_dataset = GroundingQADataset(examples, data_root, processor)

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_arguments(output_dir, {**train_cfg, "seed": global_seed}),
        train_dataset=train_dataset,
        data_collator=VisionQACollator(int(pad_token_id)),
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(output_dir / "processor")
    print(f"adapter={output_dir / 'final_adapter'}")
    print("selection_metric=numeric_accuracy")
    return 0


if __name__ == "__main__":
    import os

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
