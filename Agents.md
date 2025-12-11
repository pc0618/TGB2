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
- Additional temporal features (`ageact_v1`) are concatenated to every edge message:
  - `src_age_days`, `dst_age_days` – days since each endpoint first appeared.
  - `src_hours_since_prev`, `dst_hours_since_prev` – hours since the node’s previous event (=-1 when none).
  - `src_events_7d`, `dst_events_7d` – number of events for the node in the trailing 7-day window (excluding the current one).
  These are derived directly from the chronological stream and cached inside `datasets/schema_cache_augmented/<schema>_ageact_v1.pt` for both the default 18-relation view and the aggregated 10-table view.
- Masks (`train_mask`, `val_mask`, `test_mask`) slice the TemporalData chronologically. Optional `--split_frac` down-samples each split uniformly without changing the chronological order.
- **10-table projection (agg10)**: When `--schema_variant agg10` is selected, we map the original 14 relation types onto the 6 aggregated relation tables below while appending the event-type id to the edge message. This keeps the entity vocabulary identical but collapses redundant edges (e.g., `U_SE_O_I`, `U_SE_C_I`, `U_SE_RO_I` all live in `user_issue_events` with `event_type ∈ {opened, closed, reopened}`). The schema+feature cache lives in `datasets/schema_cache_augmented/thgl-software_agg10_ageact_v1.pt`.
  - Aggregated relation ids: `user_issue_events`, `issue_repo_events`, `user_pr_events`, `pr_repo_events`, `user_repo_events`, `repo_repo_events`.
  - Event ids appended to `msg`: `opened=0`, `closed=1`, `reopened=2`, `added_collaborator=3`, `forked_from=4`.
  - Temporal dynamics: collapsing multiple edge labels into one stream means per-node histories become denser (e.g., a user’s open/close/reopen issue events are now distinguishable only via the attached event id). This makes the temporal aggregator rely more heavily on the appended event-type signal instead of separate relation embeddings, so we now capture “what happened” via features while “who interacted” stays identical. The memory/neighbor loader still sees the same timestamps, but attention has to disambiguate event semantics from the new feature slot instead of unique relation ids.

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
- Workers: 16 (the CPU node tolerates this after switching to `torch.multiprocessing.set_sharing_strategy("file_system")`; go lower if `/dev/shm` errors return).
- Training negatives: `--num_neg_samples 5` (averaged across the sampled destinations before BCE so the loss weight per positive stays stable).
- Command pattern (sweep wrapper):
  ```
  cd ~/TGB2 && source ~/relbench/.venv/bin/activate && \
  PYTHON_BIN=/home/pc0618/relational-benchmark/.venv/bin/python \
  EPOCHS=60 BATCH_SIZE=<128|256> LR_VALUES='1e-4 2e-4' AGGRS='mean' \
  NUM_WORKERS=16 NUM_NEG_SAMPLES=5 RUN_PREFIX=hero_mean_full_default18 \
  SCHEMA_VARIANT=default18 SCHEMA_CACHE_DIR=datasets/schema_cache_augmented \
  EXTRA_ARGS="--split_frac 1.0 --log_every 200 --checkpoint_every 10 \
  --checkpoint_dir examples/linkproppred/thgl-forum/saved_checkpoints/hero_mean_default18" \
  PYTHONPATH=. ./scripts/thgl_tgn_sweep.sh
  ```
- Aggregated schema uses the same command but set `SCHEMA_VARIANT=agg10`, `RUN_PREFIX=hero_mean_full_agg10`, and point the checkpoint dir at `.../hero_mean_agg10`.
- For the 10-table schema repeat the same command with `SCHEMA_VARIANT=agg10 SCHEMA_CACHE_DIR=datasets/schema_cache` and segregate checkpoints (e.g., `--checkpoint_dir .../hero_mean_agg10`). This uses the cached conversion discussed above and keeps wandb run names tagged with `schemaagg10`.
- The best quick-run so far: `aggr=mean`, `lr=2e-4`, `bs=256`, `split_frac=0.02` → Val MRR 0.0316, Test MRR 0.0294. Those hyperparameters seeded the full-epoch hero runs.
