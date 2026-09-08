#!/usr/bin/env python3
"""Train an answer-only Qwen3-VL LoRA on canonical financial QA manifests."""

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

from financial_vlm.data.synfintabs_training import flatten_group_records, load_jsonl  # noqa: E402
from financial_vlm.training.standard_qa import (  # noqa: E402
    answer_target,
    filter_records_by_variant,
    normalize_interval_strategy,
    standard_qa_prompt,
)


DEFAULT_CONFIG = REPO_ROOT / "configs" / "synfintabs" / "synfintabs_standard_qa_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", type=Path, help="Override data.root_dir.")
    parser.add_argument("--train-manifest", type=Path, help="Override data.train_manifest.")
    parser.add_argument("--eval-manifest", type=Path, help="Override data.dev20_manifest for loss-only eval.")
    parser.add_argument("--output-dir", type=Path, help="Override output.root_dir.")
    parser.add_argument("--model-path", type=Path, help="Override model.base_model_path.")
    parser.add_argument("--learning-rate", type=float, help="Override training.learning_rate.")
    parser.add_argument("--limit-train", type=int, help="Use only the first N train records for smoke tests.")
    parser.add_argument("--limit-eval", type=int, help="Use only the first N eval records for smoke tests.")
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
        "selection_metric": "numeric_accuracy",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_dir


def load_records(path: Path, allowed_variants: Sequence[str], limit: int | None) -> list[dict[str, Any]]:
    records = flatten_group_records(load_jsonl(path), allowed_variants)
    records = filter_records_by_variant(records, allowed_variants)
    if limit is not None:
        records = records[:limit]
    if not records:
        raise ValueError(f"No records selected from manifest: {path}")
    return records


def build_messages(image: Any, question: str, answer: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": standard_qa_prompt(question)},
            ],
        }
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


class CanonicalQADataset:
    def __init__(self, records: Sequence[Mapping[str, Any]], data_root: Path, processor: Any):
        self.records = [dict(record) for record in records]
        self.data_root = data_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        from PIL import Image
        import torch

        record = self.records[index]
        image_path = Path(record["image_path"]).expanduser()
        if not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")

        question = str(record["question"])
        target = answer_target(record)
        prompt_inputs = self.processor.apply_chat_template(
            build_messages(image, question),
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs = self.processor.apply_chat_template(
            build_messages(image, question, target),
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

        example: dict[str, Any] = {}
        for key, value in full_inputs.items():
            if torch.is_tensor(value) and key in {"input_ids", "attention_mask"}:
                example[key] = value.squeeze(0)
            else:
                example[key] = value
        example["labels"] = labels
        return example


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
        "per_device_eval_batch_size": int(cfg.get("per_device_eval_batch_size", 1)),
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
        "seed": int(cfg.get("seed", 20260802)),
    }
    signature = inspect.signature(TrainingArguments)
    eval_strategy_key = "eval_strategy" if "eval_strategy" in signature.parameters else "evaluation_strategy"
    kwargs[eval_strategy_key] = normalize_interval_strategy(cfg.get("eval_strategy"), default="no")
    if "eval_steps" in cfg:
        kwargs["eval_steps"] = int(cfg["eval_steps"])
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
        config["training"] = train_cfg
    lora_cfg = config["lora"]
    output_cfg = config["output"]

    data_root = path_from_config(args.data_root or data_cfg["root_dir"])
    train_manifest = path_from_config(args.train_manifest or data_cfg["train_manifest"], data_root)
    eval_manifest_value = args.eval_manifest or data_cfg.get("dev20_manifest")
    eval_manifest = path_from_config(eval_manifest_value, data_root) if eval_manifest_value else None
    output_dir = prepare_output_dir(
        path_from_config(args.output_dir or output_cfg["root_dir"]),
        config_path,
        config,
    )
    model_path = path_from_config(args.model_path or model_cfg["base_model_path"])
    variants = list(data_cfg.get("include_variants", ["clean"]))

    train_records = load_records(train_manifest, variants, args.limit_train)
    eval_records = load_records(eval_manifest, variants, args.limit_eval) if eval_manifest is not None else None

    from transformers import set_seed

    set_seed(int(config["experiment"]["seed"]))
    model, processor = load_model_and_processor(model_path, model_cfg, lora_cfg)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0)

    train_dataset = CanonicalQADataset(train_records, data_root, processor)
    eval_dataset = CanonicalQADataset(eval_records, data_root, processor) if eval_records is not None else None

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_arguments(output_dir, {**train_cfg, "seed": config["experiment"]["seed"]}),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VisionQACollator(int(pad_token_id)),
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(output_dir / "processor")
    print(f"adapter={output_dir / 'final_adapter'}")
    print("selection_metric=numeric_accuracy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
