#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dataset="${1:-tgbn-genre}"
epochs="${2:-5}"

# Optional overrides (env vars):
# - BATCH_SIZE, FANOUTS, EMB_DIM, HIDDEN_DIM, NUM_HEADS
# - NUM_NEG_EVAL, MAX_TRAIN_EVENTS, MAX_EVAL_EVENTS, ADJ
batch_size="${BATCH_SIZE:-1024}"
fanouts="${FANOUTS:-10,5}"
emb_dim="${EMB_DIM:-64}"
hidden_dim="${HIDDEN_DIM:-64}"
num_heads="${NUM_HEADS:-4}"
num_neg_eval="${NUM_NEG_EVAL:-100}"
max_train_events="${MAX_TRAIN_EVENTS:-50000}"
max_eval_events="${MAX_EVAL_EVENTS:-5000}"
adj="${ADJ:-val}"
checkpoint_every="${CHECKPOINT_EVERY:-1}"
checkpoint_dir="${CHECKPOINT_DIR:-}"

extra_args=()
if [[ -n "$checkpoint_dir" ]]; then
  extra_args+=(--checkpoint_dir "$checkpoint_dir")
fi

# Budgeted but "real" (more than a 1-epoch smoke test).
# Requires: `relbench_exports/<dataset>` already exported with adjacency built.
PYTHONPATH=. .venv/bin/python baselines/graphsage_nodeprop.py \
  --dataset "$dataset" \
  --model gat \
  --epochs "$epochs" \
  --batch_size "$batch_size" \
  --fanouts "$fanouts" \
  --emb_dim "$emb_dim" \
  --hidden_dim "$hidden_dim" \
  --num_heads "$num_heads" \
  --num_neg_eval "$num_neg_eval" \
  --max_train_events "$max_train_events" \
  --max_eval_events "$max_eval_events" \
  --adj "$adj" \
  --checkpoint_every "$checkpoint_every" \
  "${extra_args[@]}"
