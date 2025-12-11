# Agents Notes

## Dataset & Preprocessing
- **Source**: `thgl-software` from `tgb.linkproppred.dataset_pyg.PyGLinkPropPredDataset`.
- **TemporalData stats** (full dataset, before any `split_frac` truncation):
  - Total events: `1,489,806`
  - Train events: `1,042,866`
  - Validation events: `223,469`
  - Test events: `223,471`
- Each event carries `src`, `dst`, timestamp `t`, relation id `edge_type`, and message vector `msg`.
- We embed `edge_type` (128-dim) and concatenate it onto `msg`, so every TemporalData message includes relation context.
- Masks (`train_mask`, `val_mask`, `test_mask`) slice the TemporalData chronologically. Optional `--split_frac` down-samples each split uniformly without changing the chronological order.

## PyG TemporalData Usage
- `TemporalDataLoader` iterates chronologically, yielding batches of events (size = `BATCH_SIZE` events).
- Batch tensors are moved to device via `batch.to(device)`; each has aligned fields (`batch.src`, `batch.dst`, `batch.t`, `batch.msg`, `batch.edge_type`).
- We track a `LastNeighborLoader` (size 10) per run; given node ids (`n_id`) from the current batch, it returns the induced subgraph (edge_index + edge ids) from the running neighbor store.
- After each train/eval batch we call `model['memory'].update_state(src, dst, t, msg)` and `neighbor_loader.insert(src, dst)` so future batches see the newest history.
- During evaluation the negative sampler (`dataset.negative_sampler`) pairs each positive temporal edge with a set of same-time negative destinations; scores are fed into `Evaluator.eval` for MRR.

## Model Architecture
- **Memory**: `TGNMemory` with dimensions `(mem_dim, time_dim)` and `IdentityMessage`. Supports aggregator registry (`last`, `mean`, `sum`, `max`).
- **Temporal GNN**: `GraphAttentionEmbedding` wraps a single `TransformerConv` (2 heads, dropout 0.1). Inputs are the memory embeddings (`mem_dim`) and edge attributes `[time_encoding || msg]`. Output dimensionality = `emb_dim`. (So we currently have **one** GNN layer.)
- **Link Predictor**: `modules.decoder.LinkPredictor` scoring `(z_src, z_dst)` pairs with BCE loss. Negative samples drawn uniformly between `min_dst_idx` and `max_dst_idx`.
- **Hyperparameters** surfaced via CLI (`tgb/utils/utils.py`): lr, batch size, memories, aggregator, workers, `split_frac`, `log_every`, WandB flags, and checkpoint cadence.

## Training, Evaluation & Logging
- **Training loop**:
  1. Reset memory + neighbor loader.
  2. For each TemporalData batch: sample negatives, build local graph via neighbor loader, run `memory → GraphAttentionEmbedding → LinkPredictor`.
  3. BCE loss on positives vs negatives; `log_every` batches emit console + WandB metrics.
- **Validation/Test**:
  - Use the same loader but pull negatives via `dataset.negative_sampler`.
  - Compute one-vs-many scores (`y_pred_pos`, `y_pred_neg`) and pass to `Evaluator` for dataset metric (MRR here).
  - Memory and neighbor loader are updated with ground-truth edges after each batch, mirroring the TGN evaluation setting.
- **WandB**:
  - Auto-config includes dataset, aggregator, `bs`, `lr`, `mem/time/emb`, `edge_emb_dim`, `num_neighbors`, `num_workers`, `split_frac`, event counts, etc.
  - Run names now encode key hyperparameters (`model_data_aggr-bs_lr_mem_time_emb_neigh_layers_epochs`).
  - Local logs recorded at `examples/linkproppred/thgl-forum/saved_results/training_logs/<run>_metrics.log`.
- **Checkpoints**:
  - Early stopper saves best val-MRR checkpoint at `saved_models/TGN_thgl-software_<seed>_<run>.pth`.
  - Periodic checkpoints via `--checkpoint_every N` and `--checkpoint_dir <path>` capture full model + optimizer state every `N` epochs (disabled when `N=0`).

## Current “Hero” Config Template
- Dataset fraction: full (`--split_frac 1.0`).
- Epoch budget: 60.
- Batch sizes explored: 128 and 256.
- Learning rates under test: `1e-4`, `2e-4`.
- Aggregator: `mean`.
- Workers: 12 (limited by `/dev/shm` on the CPU node).
- Command pattern (sweep wrapper):
  ```
  cd ~/TGB2 && source ~/relbench/.venv/bin/activate && \
  EPOCHS=60 BATCH_SIZE=<128|256> LR_VALUES='1e-4 2e-4' AGGRS='mean' NUM_WORKERS=12 \
  RUN_PREFIX=hero_mean_full \
  EXTRA_ARGS="--split_frac 1.0 --log_every 200 --checkpoint_every 10 \
  --checkpoint_dir examples/linkproppred/thgl-forum/saved_checkpoints/hero_mean" \
  PYTHONPATH=. ./scripts/thgl_tgn_sweep.sh
  ```
- The best quick-run so far: `aggr=mean`, `lr=2e-4`, `bs=256`, `split_frac=0.02` → Val MRR 0.0316, Test MRR 0.0294. Those hyperparameters seeded the full-epoch hero runs.

