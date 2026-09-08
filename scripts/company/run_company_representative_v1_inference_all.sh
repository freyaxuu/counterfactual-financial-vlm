#!/usr/bin/env bash
# Run all 10 models (9 trained checkpoints + frozen zero-shot base) over the
# company_representative_v1 unique-question index, 4-at-a-time (one per
# GPU). Identical structure and checkpoint set to
# run_company_confusion_v1_inference_all.sh -- see that script's header for
# the checkpoint-selection rationale (reused unchanged, never selected
# against company data).
set -uo pipefail

REPO=~/apps/cf-vlm-financial-grounding
RUNS=~/runs/cf-vlm-financial-grounding
INDEX=$RUNS/company_representative_v1_unique_questions/unique_questions.jsonl
OUT_ROOT=$RUNS/company_representative_v1_predictions
mkdir -p "$OUT_ROOT"

cd "$REPO"

declare -a LABELS=(
  "base"
  "standard_lora_seed_20260804"
  "standard_lora_seed_20260806"
  "standard_lora_seed_20260808"
  "compute_matched_clean_seed_20260804"
  "compute_matched_clean_seed_20260806"
  "compute_matched_clean_seed_20260808"
  "cf_augmentation_seed_20260804"
  "cf_augmentation_seed_20260806"
  "cf_augmentation_seed_20260808"
)
declare -a ADAPTERS=(
  ""
  "$RUNS/tatqa_train_dev_cfv1_qa_lora_seed_20260804/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_qa_lora_seed_20260806/final_adapter"
  "$RUNS/tatqa_lr_sweep_20260808/standard_lora_lr0.0001/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_compute_matched_clean_lora_seed_20260804/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_compute_matched_clean_lora_seed_20260806/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_compute_matched_clean_lora_seed_20260808/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_cf_augmentation_lora_seed_20260804/final_adapter"
  "$RUNS/tatqa_train_dev_cfv1_cf_augmentation_lora_seed_20260806/final_adapter"
  "$RUNS/tatqa_lr_sweep_20260808/cf_augmentation_lora_lr0.0003/final_adapter"
)

N=${#LABELS[@]}
GPUS=4
i=0
round=1
while [ "$i" -lt "$N" ]; do
  echo "=== round $round starting at $(date) ==="
  pids=()
  gpu=0
  batch_end=$((i + GPUS))
  if [ "$batch_end" -gt "$N" ]; then batch_end=$N; fi
  for ((j = i; j < batch_end; j++)); do
    label="${LABELS[$j]}"
    adapter="${ADAPTERS[$j]}"
    out_dir="$OUT_ROOT/$label"
    log="$OUT_ROOT/${label}.log"
    adapter_arg=()
    if [ -n "$adapter" ]; then
      adapter_arg=(--adapter-path "$adapter")
    fi
    echo "launching $label on GPU $gpu -> $out_dir"
    CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=src /path/to/python-env/.venv/bin/python3 scripts/common/evaluate_qwen3_vl_natural_pairs.py \
      --unique-questions "$INDEX" \
      "${adapter_arg[@]}" \
      --output-dir "$out_dir" \
      > "$log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  echo "waiting on pids: ${pids[*]}"
  for pid in "${pids[@]}"; do
    wait "$pid"
    echo "pid $pid finished with status $? at $(date)"
  done
  echo "=== round $round done at $(date) ==="
  i=$batch_end
  round=$((round + 1))
done

echo "ALL_DONE at $(date)"
