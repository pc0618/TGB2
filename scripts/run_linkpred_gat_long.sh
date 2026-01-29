#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

epochs="${1:-5}"
shift || true

# Longer TGN + GraphAttention link prediction runs (sampled-negative MRR).
# Notes:
# - `tgbl-*-v2` ids are not supported by this repo’s `PyGLinkPropPredDataset`; use `tgbl-wiki` / `tgbl-review`.
# - These datasets can be large; keep `max_*_events` caps for CPU runs.

common_args=(
  --eval_mode sampled
  --num_neg_eval "${NUM_NEG_EVAL:-100}"
  --num_epoch "$epochs"
  --checkpoint_every "${CHECKPOINT_EVERY:-1}"
  --patience "${PATIENCE:-1000}"
  --num_run 1
  --bs "${BATCH_SIZE:-96}"
)

checkpoint_dir="${CHECKPOINT_DIR:-}"
if [[ -n "$checkpoint_dir" ]]; then
  common_args+=(--checkpoint_dir "$checkpoint_dir")
fi

max_val_events="${MAX_VAL_EVENTS:-20000}"
max_test_events="${MAX_TEST_EVENTS:-20000}"
max_train_default="${MAX_TRAIN_EVENTS:-200000}"
max_train_wiki="${MAX_TRAIN_EVENTS_WIKI:-0}"

if [[ "$#" -gt 0 ]]; then
  datasets=("$@")
else
  datasets=(
    tgbl-wiki
    tgbl-review
    tgbl-coin
    tgbl-comment
    tgbl-flight
  )
fi

for ds in "${datasets[@]}"; do
  case "$ds" in
    tgbl-wiki)
      max_train="$max_train_wiki"
      ;;
    *)
      max_train="$max_train_default"
      ;;
  esac

  echo "==== Running $ds (num_epoch=$epochs, max_train_events=$max_train) ===="
  PYTHONPATH=. .venv/bin/python examples/linkproppred/thgl-forum/tgn.py \
    --data "$ds" \
    --max_train_events "$max_train" \
    --max_val_events "$max_val_events" \
    --max_test_events "$max_test_events" \
    "${common_args[@]}"
done
