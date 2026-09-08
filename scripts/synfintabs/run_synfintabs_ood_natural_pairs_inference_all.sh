#!/usr/bin/env bash
# Run base (zero-shot) + the 9 SynFinTabs-trained checkpoints over the new
# synfintabs_ood_natural_pairs_v1 unique-question index (200 pairs, 400
# unique questions, held-out OOD template "5", never touched by any of
# these checkpoints' training). Public synthetic data -- no `bash -c`
# wrapper needed (unlike the company-data scripts).
set -uo pipefail

REPO=~/apps/cf-vlm-financial-grounding
RUNS=~/runs/cf-vlm-financial-grounding
INDEX=$RUNS/synfintabs_ood_natural_pairs_v1/unique_questions.jsonl
OUT_ROOT=$RUNS/synfintabs_ood_natural_pairs_v1_predictions
mkdir -p "$OUT_ROOT"

cd "$REPO"

declare -a LABELS=(
  "base"
  "qa_lora_seed_20260804"
  "qa_lora_seed_20260805"
  "qa_lora_seed_20260806"
  "compute_matched_clean_lora_seed_20260804"
  "compute_matched_clean_lora_seed_20260805"
  "compute_matched_clean_lora_seed_20260806"
  "cf_augmentation_lora_seed_20260804"
  "cf_augmentation_lora_seed_20260805"
  "cf_augmentation_lora_seed_20260806"
)
declare -a ADAPTERS=(
  ""
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_qa_lora_seed_20260804/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_qa_lora_seed_20260805/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_qa_lora_seed_20260806/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_compute_matched_clean_lora_seed_20260804/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_compute_matched_clean_lora_seed_20260805/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_compute_matched_clean_lora_seed_20260806/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_cf_augmentation_lora_seed_20260804/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_cf_augmentation_lora_seed_20260805/final_adapter"
  "$RUNS/synfintabs_train_dev_test_ood_cfv1_cf_augmentation_lora_seed_20260806/final_adapter"
)
declare -a GPU_IDS=(0 1 2 3)

N=${#LABELS[@]}
GPUS=${#GPU_IDS[@]}
i=0
round=1
while [ "$i" -lt "$N" ]; do
  echo "=== round $round starting at $(date) ==="
  pids=()
  gpu_idx=0
  batch_end=$((i + GPUS))
  if [ "$batch_end" -gt "$N" ]; then batch_end=$N; fi
  for ((j = i; j < batch_end; j++)); do
    label="${LABELS[$j]}"
    adapter="${ADAPTERS[$j]}"
    gpu="${GPU_IDS[$gpu_idx]}"
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
    gpu_idx=$((gpu_idx + 1))
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
