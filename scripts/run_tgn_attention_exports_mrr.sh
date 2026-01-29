#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

py="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
exports_root="${EXPORTS_ROOT:-relbench_exports}"

epochs="${EPOCHS:-5}"
batch_size="${BATCH_SIZE:-96}"
eval_batch_size="${EVAL_BATCH_SIZE:-128}"
num_neighbors="${NUM_NEIGHBORS:-10}"
mem_dim="${MEM_DIM:-64}"
time_dim="${TIME_DIM:-32}"
emb_dim="${EMB_DIM:-64}"
lr="${LR:-1e-3}"
num_neg_train="${NUM_NEG_TRAIN:-10}"
num_neg_eval="${NUM_NEG_EVAL:-100}"
max_train_events="${MAX_TRAIN_EVENTS:-200000}"
max_val_events="${MAX_VAL_EVENTS:-20000}"
max_test_events="${MAX_TEST_EVENTS:-20000}"
adj="${ADJ:-val}"
device="${DEVICE:-cpu}"
parquet_batch_size="${PARQUET_BATCH_SIZE:-500000}"

run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
log_dir="${LOG_DIR:-$repo_root/logs/tgn_attn_exports_${run_tag}}"
mkdir -p "$log_dir"

common_args=(
  --exports_root "$exports_root"
  --adj "$adj"
  --epochs "$epochs"
  --batch_size "$batch_size"
  --eval_batch_size "$eval_batch_size"
  --num_neighbors "$num_neighbors"
  --mem_dim "$mem_dim"
  --time_dim "$time_dim"
  --emb_dim "$emb_dim"
  --lr "$lr"
  --num_neg_train "$num_neg_train"
  --num_neg_eval "$num_neg_eval"
  --max_train_events "$max_train_events"
  --max_val_events "$max_val_events"
  --max_test_events "$max_test_events"
  --parquet_batch_size "$parquet_batch_size"
  --device "$device"
)

datasets=("$@")
if [[ "${#datasets[@]}" -eq 0 ]]; then
  datasets=(
    tgbl-wiki
    tgbl-wiki-v2
    tgbl-review
    tgbl-review-v2
    tgbl-coin
    tgbl-comment
    tgbl-flight
  )
fi

echo "[run_tag] $run_tag"
echo "[log_dir] $log_dir"

for ds in "${datasets[@]}"; do
  echo "==== $ds ===="
  out="$log_dir/${ds}.txt"
  "$py" baselines/tgn_attention_linkpred_exports.py \
    --dataset "$ds" \
    "${common_args[@]}" 2>&1 | tee "$out"
done

echo "DONE: $log_dir"

