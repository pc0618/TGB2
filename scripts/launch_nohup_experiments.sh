#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v nohup >/dev/null 2>&1; then
  echo "ERROR: nohup not found in PATH" >&2
  exit 1
fi

run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
epochs="${EPOCHS:-5}"

log_root="${LOG_ROOT:-$repo_root/logs/nohup_$run_tag}"
ckpt_root="${CKPT_ROOT:-$repo_root/saved_models/nohup_ckpts_$run_tag}"
mkdir -p "$log_root" "$ckpt_root"

python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: python not found/executable at $python_bin" >&2
  exit 1
fi

echo "run_tag=$run_tag"
echo "epochs=$epochs"
echo "log_root=$log_root"
echo "ckpt_root=$ckpt_root"
echo "linkpred_bs=${BATCH_SIZE:-96}"
echo "nodeprop_batch_size=${NODEPROP_BATCH_SIZE:-512}"
echo "wandb=${WANDB:-1}"
echo

run_nohup() {
  local name="$1"
  shift
  local log_file="$log_root/$name.out"
  local pid_file="$log_root/$name.pid"

  # shellcheck disable=SC2068
  nohup "$@" >"$log_file" 2>&1 &
  local pid=$!
  echo "$pid" >"$pid_file"
  echo "STARTED $name pid=$pid log=$log_file"
}

# -------------------------
# Link prediction (TGN + GraphAttention) on TGBL datasets
# -------------------------
if [[ "${RUN_LINKPRED:-1}" == "1" ]]; then
  linkpred_ckpt_root="$ckpt_root/linkpred"
  mkdir -p "$linkpred_ckpt_root"

  wandb_args_linkpred=()
  if [[ "${WANDB:-1}" == "1" ]]; then
    wandb_args_linkpred+=(--wandb)
    wandb_args_linkpred+=(--wandb_project "${WANDB_PROJECT_LINKPRED:-tgb2-linkpred}")
    wandb_args_linkpred+=(--wandb_group "${WANDB_GROUP:-$run_tag}")
  fi

  # Each command below is a unique dataset (one nohup per dataset).
  for ds in tgbl-wiki tgbl-review tgbl-coin tgbl-comment tgbl-flight; do
    max_train=200000
    if [[ "$ds" == "tgbl-wiki" ]]; then
      max_train=0
    fi

    name="linkpred_${ds}_e${epochs}"
    ds_ckpt_dir="$linkpred_ckpt_root/$ds"
    mkdir -p "$ds_ckpt_dir"

    run_nohup "$name" env PYTHONPATH=. PYTHONUNBUFFERED=1 \
      "$python_bin" examples/linkproppred/thgl-forum/tgn.py \
      --data "$ds" \
      --eval_mode sampled \
      --num_neg_eval "${NUM_NEG_EVAL:-100}" \
      --num_epoch "$epochs" \
      --num_run 1 \
      --bs "${BATCH_SIZE:-96}" \
      --max_train_events "$max_train" \
      --max_val_events "${MAX_VAL_EVENTS:-20000}" \
      --max_test_events "${MAX_TEST_EVENTS:-20000}" \
      --checkpoint_every "${CHECKPOINT_EVERY:-1}" \
      --checkpoint_dir "$ds_ckpt_dir" \
      --patience "${PATIENCE:-1000}" \
      "${wandb_args_linkpred[@]}"
  done
fi

# -------------------------
# Node prediction (GAT nodeprop) on TGBN datasets
# -------------------------
if [[ "${RUN_NODEPROP:-1}" == "1" ]]; then
  nodeprop_ckpt_root="$ckpt_root/nodeprop"
  mkdir -p "$nodeprop_ckpt_root"

  wandb_args_nodeprop=()
  if [[ "${WANDB:-1}" == "1" ]]; then
    wandb_args_nodeprop+=(--wandb)
    wandb_args_nodeprop+=(--wandb_mode online)
    wandb_args_nodeprop+=(--wandb_project "${WANDB_PROJECT_NODEPROP:-tgb2-nodeprop}")
    wandb_args_nodeprop+=(--wandb_name "${WANDB_NAME_PREFIX:-$run_tag}_nodeprop")
  fi

  # Each command below is a unique dataset (one nohup per dataset).
  for ds in tgbn-genre tgbn-trade tgbn-reddit tgbn-token; do
    name="nodeprop_${ds}_e${epochs}"
    ds_ckpt_dir="$nodeprop_ckpt_root/$ds"
    mkdir -p "$ds_ckpt_dir"

    run_nohup "$name" env PYTHONPATH=. PYTHONUNBUFFERED=1 \
      "$python_bin" baselines/graphsage_nodeprop.py \
      --dataset "$ds" \
      --model gat \
      --epochs "$epochs" \
      --checkpoint_every "${CHECKPOINT_EVERY:-1}" \
      --checkpoint_dir "$ds_ckpt_dir" \
      --batch_size "${NODEPROP_BATCH_SIZE:-512}" \
      --fanouts "${NODEPROP_FANOUTS:-10,5}" \
      --emb_dim "${NODEPROP_EMB_DIM:-64}" \
      --hidden_dim "${NODEPROP_HIDDEN_DIM:-64}" \
      --num_heads "${NODEPROP_NUM_HEADS:-4}" \
      --num_neg_eval "${NODEPROP_NUM_NEG_EVAL:-100}" \
      --max_train_events "${NODEPROP_MAX_TRAIN_EVENTS:-50000}" \
      --max_eval_events "${NODEPROP_MAX_EVAL_EVENTS:-5000}" \
      --adj "${NODEPROP_ADJ:-val}" \
      "${wandb_args_nodeprop[@]}"
  done
fi

echo
echo "Logs: $log_root"
echo "Checkpoints: $ckpt_root"
echo "Tail a log: tail -f \"$log_root/<name>.out\""
echo "PIDs:"
for f in "$log_root"/*.pid; do
  [[ -e "$f" ]] || continue
  echo "  $(basename "$f" .pid): $(cat "$f")"
done
