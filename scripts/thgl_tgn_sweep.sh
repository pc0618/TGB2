#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-~/relbench/.venv/bin/python}
DATASET=${DATASET:-thgl-software}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-256}
MEM_DIM=${MEM_DIM:-128}
TIME_DIM=${TIME_DIM:-128}
EMB_DIM=${EMB_DIM:-128}
LR_VALUES=${LR_VALUES:-"2e-4 5e-4"}
AGGRS=${AGGRS:-"sum mean max"}
PATIENCE=${PATIENCE:-2}
EDGE_EMB_DIM=${EDGE_EMB_DIM:-128}
WANDB_PROJECT=${WANDB_PROJECT:-tgb-thgl}
WANDB_ENTITY=${WANDB_ENTITY:-marin-community}
WANDB_GROUP=${WANDB_GROUP:-thgl-tgn}
RUN_PREFIX=${RUN_PREFIX:-tgn_thgl}
NUM_WORKERS=${NUM_WORKERS:-22}
NUM_NEG_SAMPLES=${NUM_NEG_SAMPLES:-1}
SCHEMA_VARIANT=${SCHEMA_VARIANT:-default18}
SCHEMA_CACHE_DIR=${SCHEMA_CACHE_DIR:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

extra_cache_args=()
if [[ -n "${SCHEMA_CACHE_DIR}" ]]; then
  extra_cache_args+=(--schema_cache_dir "${SCHEMA_CACHE_DIR}")
fi

for aggr in ${AGGRS}; do
  for lr in ${LR_VALUES}; do
    safe_lr=${lr//./p}
    safe_lr=${safe_lr//-/m}
    run_name="${RUN_PREFIX}_${aggr}_bs${BATCH_SIZE}_lr_${safe_lr}_mem${MEM_DIM}_time${TIME_DIM}_emb${EMB_DIM}_nw${NUM_WORKERS}_epochs${EPOCHS}_neg${NUM_NEG_SAMPLES}_schema${SCHEMA_VARIANT}"
    echo "[INFO] Launching run ${run_name} (aggr=${aggr}, lr=${lr})"
    PYTHONPATH=. "${PYTHON_BIN}" examples/linkproppred/thgl-forum/tgn.py \
      --data "${DATASET}" \
      --num_epoch "${EPOCHS}" \
      --num_run 1 \
      --bs "${BATCH_SIZE}" \
      --mem_dim "${MEM_DIM}" \
      --time_dim "${TIME_DIM}" \
      --emb_dim "${EMB_DIM}" \
      --lr "${lr}" \
      --patience "${PATIENCE}" \
      --split_frac 1.0 \
      --aggr "${aggr}" \
      --edge_emb_dim "${EDGE_EMB_DIM}" \
      --num_workers "${NUM_WORKERS}" \
      --num_neg_samples "${NUM_NEG_SAMPLES}" \
      --schema_variant "${SCHEMA_VARIANT}" \
      "${extra_cache_args[@]}" \
      --wandb \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --wandb_group "${WANDB_GROUP}" \
      --wandb_run_name "${run_name}" \
      ${EXTRA_ARGS}
  done
done
