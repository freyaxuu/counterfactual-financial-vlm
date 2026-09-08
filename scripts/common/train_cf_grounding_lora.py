#!/usr/bin/env python3
"""Train CF augmentation + Grounding-only (answer-first) LoRA.

Completes the 2x2 ablation grid alongside CF + Evidence-first Grounding
(train_cf_evidence_first_grounding_lora.py): same CF cycle and pair schedule
as CF-augmentation, but this time both sides of every pair use the
Grounding-only answer-first target

    <answer>...</answer>
    <row>...</row>
    <column>...</column>
    <bbox>x1 y1 x2 y2</bbox>

instead of the Evidence-first bbox-before-answer target. Loss is the same
summed dual-forward-pass form:

    L = L_grounding(clean) + L_grounding(counterfactual)

Purpose: CF + Evidence-first Grounding (dev CGS 0.9465) underperforms plain
CF-augmentation (0.9639). This experiment isolates *why* -- is it the
bbox-before-answer ordering specifically, or does *any* extra grounding
target (regardless of order) dilute the CF signal? If CF + Grounding-only
lands close to CF + Evidence-first, the extra grounding output itself is the
cost, not the ordering. If it lands close to plain CF-augmentation, ordering
specifically is what hurt Evidence-first.

Reuses financial_vlm.training.cf_evidence_first_grounding's
validate_and_filter_groups / materialize_evidence_first_cf_pairs /
pair_to_log_record UNCHANGED -- despite the module's name, that pairing
logic is target-format-agnostic: it returns two full GroundingExample
objects (clean and cf sides) and defers *serialization* entirely to the
caller. train_cf_evidence_first_grounding_lora.py reformats each side via
format_evidence_first_target(); this script instead uses each
GroundingExample's own `.target_text` field directly, which
build_grounding_example already populates in the answer-first Grounding-only
format via format_grounding_target() -- so no reformatting call is needed
here at all. Standard LoRA, CF-augmentation, Grounding-only, Evidence-first
Grounding, and CF + Evidence-first Grounding scripts are all untouched.

Smoke test / overfit sanity check: --limit-groups 20-30 with a large
--num-cycles override (num_train_epochs is fixed at 1.0, same rationale as
the CF-pair scripts above), then eval_grounding_lora.py --limit-groups
<same N> --print-samples 10 -- reused unmodified, same output format as
plain Grounding-only.
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
)
from financial_vlm.data.synfintabs_training import load_jsonl  # noqa: E402
from financial_vlm.training.cf_cycle_sampler import CF_TYPES, load_group_records  # noqa: E402
from financial_vlm.training.cf_evidence_first_grounding import (  # noqa: E402
    materialize_evidence_first_cf_pairs,
    pair_to_log_record,
    validate_and_filter_groups,
)
from financial_vlm.training.standard_qa import normalize_interval_strategy  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "configs" / "synfintabs" / "synfintabs_train_dev_test_ood_cfv1_cf_grounding_lora.yaml"


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
    parser.add_argument("--num-cycles", type=int, help="Override cf_cycle.num_cycles.")
    parser.add_argument(
        "--limit-groups",
        type=int,
        help="Train on only the first N sorted groups (smoke tests).",
    )
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
        "cf_cycle": config.get("cf_cycle", {}),
        "training": config.get("training", {}),
        "lora": config.get("lora", {}),
        "baseline": "CF-augmentation + Grounding-only (answer-first): same shuffled-balanced CF cycle as "
        "CF-augmentation, both sides use the answer-first <answer><row><column><bbox> target",
        "objective": "L = L_grounding(clean) + L_grounding(counterfactual)",
        "selection_metric": "numeric_accuracy",
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return output_dir


def print_validation_summary(stats: Mapping[str, int]) -> None:
    print("=== Grounding validation summary (clean AND all 3 CF variants must pass) ===")
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


class CFGroundingPairDataset:
    """One example per shuffled-balanced-cycle pair: {clean, counterfactual}.

    Each side is a GroundingExample; the training target is
    example.target_text, which build_grounding_example already serialized in
    the answer-first <answer><row><column><bbox> format -- no reformatting
    needed here (contrast with the Evidence-first script, which must
    override target_text with format_evidence_first_target).
    """

    def __init__(self, pairs: Sequence[Mapping[str, Any]], data_root: Path, processor: Any):
        self.pairs = list(pairs)
        self.data_root = data_root
        self.processor = processor

    def __len__(self) -> int:
        return len(self.pairs)

    def _build_example(self, example: GroundingExample) -> dict[str, Any]:
        from PIL import Image
        import torch

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        pair = self.pairs[index]
        return {
            "clean": self._build_example(pair["clean"]),
            "counterfactual": self._build_example(pair["counterfactual"]),
        }


class CFPairCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def _collate_side(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
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

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "clean": self._collate_side([example["clean"] for example in examples]),
            "counterfactual": self._collate_side([example["counterfactual"] for example in examples]),
        }


def training_arguments(output_dir: Path, cfg: Mapping[str, Any]) -> Any:
    from transformers import TrainingArguments

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        # Cycle count (how many times each group is visited) is controlled
        # entirely by cf_cycle.num_cycles / materialize_evidence_first_cf_pairs
        # -- the materialized pair list already IS one full pass, so Trainer
        # always runs exactly one epoch over it.
        "num_train_epochs": 1.0,
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
    cf_cycle_cfg = config.get("cf_cycle", {})
    train_cfg = dict(config["training"])
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate
        config["training"] = train_cfg
    lora_cfg = config["lora"]
    output_cfg = config["output"]

    data_root = path_from_config(args.data_root or data_cfg["root_dir"])
    train_manifest = path_from_config(args.train_manifest or data_cfg["train_manifest"], data_root)
    num_cycles = int(args.num_cycles or cf_cycle_cfg.get("num_cycles", 1))
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

    output_dir = prepare_output_dir(
        path_from_config(args.output_dir or output_cfg["root_dir"]),
        config_path,
        config,
    )
    model_path = path_from_config(args.model_path or model_cfg["base_model_path"])

    groups = load_group_records(train_manifest)
    if not groups:
        raise ValueError(f"No groups found in manifest: {train_manifest}")
    if args.limit_groups is not None:
        keep_ids = sorted(groups)[: args.limit_groups]
        groups = {group_id: groups[group_id] for group_id in keep_ids}

    valid_groups, validation_stats = validate_and_filter_groups(groups)
    print_validation_summary(validation_stats)
    (output_dir / "grounding_validation_summary.json").write_text(
        json.dumps(validation_stats, indent=2, sort_keys=True) + "\n"
    )
    if not valid_groups:
        raise ValueError("No valid groups survived grounding validation -- nothing to train on.")

    pairs = materialize_evidence_first_cf_pairs(valid_groups, global_seed=global_seed, num_cycles=num_cycles)
    print(
        f"groups={len(valid_groups)} num_cycles={num_cycles} pairs={len(pairs)} "
        f"cf_types={CF_TYPES}"
    )

    run_id = str(config["experiment"].get("name", output_dir.name))
    pairs_log_path = output_dir / "cf_pairs_log.jsonl"
    with pairs_log_path.open("w", encoding="utf-8") as handle:
        for step, pair in enumerate(pairs):
            record = pair_to_log_record(pair, run_id=run_id, epoch=pair["cycle_index"], global_step=step)
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    from transformers import set_seed

    set_seed(global_seed)
    model, processor = load_model_and_processor(model_path, model_cfg, lora_cfg)
    pad_token_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(processor.tokenizer, "eos_token_id", 0)

    train_dataset = CFGroundingPairDataset(pairs, data_root, processor)

    from transformers import Trainer

    class CFPairTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            clean_outputs = model(**inputs["clean"])
            cf_outputs = model(**inputs["counterfactual"])
            loss = clean_outputs.loss + cf_outputs.loss
            if return_outputs:
                return loss, {"clean": clean_outputs, "counterfactual": cf_outputs}
            return loss

    trainer = CFPairTrainer(
        model=model,
        args=training_arguments(output_dir, {**train_cfg, "seed": global_seed}),
        train_dataset=train_dataset,
        data_collator=CFPairCollator(int(pad_token_id)),
    )
    trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(output_dir / "final_adapter"))
    processor.save_pretrained(output_dir / "processor")
    print(f"adapter={output_dir / 'final_adapter'}")
    print(f"cf_pairs_log={pairs_log_path}")
    print("selection_metric=numeric_accuracy")
    return 0


if __name__ == "__main__":
    import os

    exit_code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
