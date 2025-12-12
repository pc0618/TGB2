# Temporal Relational Deep Learning on `thgl-software`

This repository contains the code and reproducibility assets we used to study **relational deep learning on temporal datasets** by bridging the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/) with [RelBench](https://relbench.stanford.edu). We focus on the `thgl-software` temporal heterogeneous graph (GitHub interactions in January 2024) and show how to:

1. Convert TGB’s `TemporalData` stream into **relational schemas** (both a fine-grained 18-table view and an aggregated 10-table view).
2. Attach vetted **temporal features** (`agegap_v1`) to every edge/event.
3. Train **Temporal Graph Networks (TGN)** with a single GraphAttention layer that leverage those schemas and features.
4. Compare against a more classical **GraphSAGE-style RelBench baseline** (`gnn_recommendation.py`).

The README documents the entire workflow so new contributors can rebuild the caches, launch training and sweeps, and re-run evaluation. All of the ad-hoc notes that used to live in `Agents.md` are folded into this file so you no longer need that notebook when onboarding.

---

## Repository layout

```
TGB2/
├── datasets/schema_cache_augmented/     # cached TemporalData tensors (schema + features)
├── examples/linkproppred/thgl-forum/    # main TGN runner (tgn.py) + saved_models/
├── schemas/                             # DBML schemas for 18-table & 10-table views
├── scripts/
│   ├── thgl_tgn_sweep.sh                # bash driver for sweeps on TGN
│   └── (relbench-compatible helpers live in ../relational-benchmark)
├── tgb/datasets/
│   ├── thgl_schema.py                   # schema conversion (default18 vs agg10)
│   └── thgl_features.py                 # agegap_v1 temporal features
├── README.md (this document)
└── ... (standard TGB source tree)
```

Companion repository: [`relational-benchmark`](https://github.com/snap-stanford/relbench). Our historical GraphSAGE-style baselines use `relbench/examples/gnn_recommendation.py`; you only need that repo if you want to reproduce the static baseline or run the RelBench loaders.

---

## Data recap: TGB + relational schemas

- **Dataset**: `thgl-software` (Temporal Heterogeneous Graph Link task). 681,927 nodes, 1,489,806 timestamped edges, 4 node types, 14 edge types, covering January 2024 GitHub events.
- **Goal**: given a source node + timestamp, predict which destination node it connects to next (measured with MRR under strict chronological splits: 70% train, 15% val, 15% test).
- **default18 schema** (`schemas/thgl_18_table_schema.dbml`): maps each TGB edge type to its own event table (e.g., `user_opened_issue`, `issue_closed_repo`, `repo_forked_repo`) plus four entity tables (`users`, `issues`, `pull_requests`, `repos`). This yields 18 relations (tables) total.
- **agg10 schema** (`schemas/thgl_10_table_schema.dbml`): one event table per entity pair (`user_issue_events`, `issue_repo_events`, …) with a categorical `event_type` column storing `opened`, `closed`, `reopened`, `added_collaborator`, or `forked_from`.
- **Temporal features** (`tgb/datasets/thgl_features.py`): `agegap_v1` appends four scalars to every edge message: days since the source/destination first appeared, and hours since their previous event (−1 if none). These are derived directly from the chronological stream and cached inside `datasets/schema_cache_augmented/`.

When you run `tgn.py` for the first time on a schema variant, it builds and caches the processed `TemporalData` tensor at `datasets/schema_cache_augmented/thgl-software_<schema>_agegap_v1.pt`. Subsequent runs simply load those files.

---

## Environment & dependencies

Tested on Python 3.10+ with PyTorch 2.1 and PyG 2.4. Install steps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .                # install TGB in editable mode
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.4.0 pandas==2.1.4 wandb==0.16.4
```

Optional (for RelBench baselines / recommendation experiments):

```bash
cd ../relational-benchmark
pip install -e .[example]
```

Set `PYTHONPATH` before running scripts so both repositories can be imported:

```bash
cd ~/TGB2
export PYTHONPATH=.
```

We log to [Weights & Biases](https://wandb.ai/). Run `wandb login` once per machine if you want metrics synced.

---

## Building schema caches

A quick smoke run builds the caches for a schema variant and validates that the loader works:

```bash
PYTHONPATH=. python examples/linkproppred/thgl-forum/tgn.py \
  --data thgl-software \
  --schema_variant default18 \
  --schema_cache_dir datasets/schema_cache_augmented \
  --split_frac 0.0005 \
  --num_epoch 1 --bs 32 --mem_dim 32 --time_dim 32 --emb_dim 32 \
  --lr 1e-3 --aggr mean --num_workers 0 --num_neg_samples 2 --log_every 10
```

Run the same command with `--schema_variant agg10` to build the aggregated cache. The first invocation prints `INFO: Building schema variant '...' with features (agegap_v1)` and writes the `.pt` file; later runs will report `INFO: Loading cached schema+feature data from ...`.

---

## Training & evaluation (TGN + GraphAttention)

We drive multi-run experiments via `scripts/thgl_tgn_sweep.sh`. The script reads environment variables, constructs descriptive WandB run names, and launches `tgn.py`. Example **hero run** (default18 schema, 60 epochs, 5 negatives, checkpoint every 5 epochs):

```bash
cd ~/TGB2
source .venv/bin/activate
PYTHONPATH=. PYTHON_BIN=$(which python) \
EPOCHS=60 BATCH_SIZE=128 LR_VALUES='1e-4 2e-4' AGGRS='mean' \
NUM_WORKERS=15 NUM_NEG_SAMPLES=5 RUN_PREFIX=hero_mean_default18 \
SCHEMA_VARIANT=default18 SCHEMA_CACHE_DIR=datasets/schema_cache_augmented \
FEATURE_TAG=feat_agegap_v1 \
EXTRA_ARGS="--split_frac 1.0 --log_every 200 --checkpoint_every 5 \
             --checkpoint_dir examples/linkproppred/thgl-forum/saved_checkpoints/hero_mean_default18" \
./scripts/thgl_tgn_sweep.sh
```

To run the aggregated schema or change workers, adjust `SCHEMA_VARIANT`, `RUN_PREFIX`, `NUM_WORKERS`, and `checkpoint_dir` accordingly. Every sweep run automatically:

1. Streams chronologically through the training mask with TGN memory + GraphAttention.
2. Logs batch losses and per-epoch validation MRR to stdout, WandB, and `saved_results/training_logs/<run>_metrics.log`.
3. Uses `EarlyStopMonitor` to save the best checkpoint in `examples/linkproppred/thgl-forum/saved_models/` (filenames encode feature version + message dimension, e.g., `TGN_thgl-software_agegap_v1_msg69_1_0.pth`).
4. Reloads that checkpoint and evaluates on the full validation and test masks at the end of training. Inference is therefore integrated into the training run—there is no separate “test-only” driver required.

**Single-run command (without the sweep script).** If you want to run one configuration manually, call `tgn.py` directly:

```bash
PYTHONPATH=. python examples/linkproppred/thgl-forum/tgn.py \
  --data thgl-software --schema_variant default18 \
  --schema_cache_dir datasets/schema_cache_augmented \
  --num_epoch 60 --bs 128 --mem_dim 128 --time_dim 128 --emb_dim 128 \
  --lr 2e-4 --aggr mean --edge_emb_dim 128 \
  --num_workers 12 --num_neg_samples 5 \
  --split_frac 1.0 --log_every 200 \
  --checkpoint_every 5 --checkpoint_dir examples/linkproppred/thgl-forum/saved_checkpoints/single_run \
  --wandb --wandb_project tgb-thgl --wandb_entity <team>
```

(Replace `--wandb_entity` with your organization or drop the flag to log locally.)

---

## Evaluation-only workflow

Training runs already reload the best checkpoint and report test MRR. If you need to re-evaluate an existing checkpoint without retraining, move it into `examples/linkproppred/thgl-forum/saved_models/` and rerun `tgn.py` with the same `--schema_variant` and `--feature_tag`, but set `--num_epoch 1` and `--patience 0`. On the first iteration the early stopper will immediately reload the supplied checkpoint and proceed to the evaluation phase:

```bash
PYTHONPATH=. python examples/linkproppred/thgl-forum/tgn.py \
  --data thgl-software --schema_variant agg10 \
  --schema_cache_dir datasets/schema_cache_augmented \
  --num_epoch 1 --bs 128 --lr 1e-4 --aggr mean --patience 0 \
  --num_workers 0 --num_neg_samples 5 --split_frac 1.0
```

Make sure the checkpoint filename matches the new naming scheme (`TGN_thgl-software_agegap_v1_msg69_<seed>_<run>.pth`).

---

## Smoke tests & debugging

- **Cache validation:** use the `split_frac 0.0005` command above. Expect ~17 train batches, validation MRR around 0.02, and test MRR around 0.01.
- **Resource limits:** our CPU nodes have 22 cores. More than ~15 dataloader workers may exhaust `/dev/shm` and produce `Bus error` messages; lower `NUM_WORKERS` if that happens.
- **WandB steps:** we maintain a global step counter inside `tgn.py`. If you fork the script, ensure you keep `wandb_step_counter` monotonic to avoid `wandb: WARNING Tried to log to step ...` warnings.
- **Checkpoint mismatches:** if you edit the message dimension (e.g., add features), delete old checkpoints so you do not accidentally reload incompatible tensors.

---

## RelBench GraphSAGE baseline (optional)

To reproduce the static GNN baseline we compared against:

```bash
cd ../relational-benchmark
source ../TGB2/.venv/bin/activate
python relbench/examples/gnn_recommendation.py \
  --dataset rel-thgl --task user-closed-pr \
  --lr 1e-4 --epochs 30 --batch_size 1024 \
  --channels 128 --aggr sum --num_layers 2 \
  --num_neighbors 96 48 --num_negatives 10 \
  --train_sample_ratio 1.0 --eval_sample_ratio 0.1 \
  --wandb --wandb_project relbench
```

This pipeline materializes a static heterogeneous graph with RelBench’s loaders, uses GraphSAGE-style layers, and relies on heavily engineered temporal features (rolling 7/30‑day stats, calendar context). We found that TGN + GraphAttention + `agegap_v1` matches or exceeds its performance with substantially less feature engineering while aligning fully with TGB’s temporal evaluation protocol.

---

## Releasing new experiments

1. Ensure caches exist for both schemas: check for `thgl-software_default18_agegap_v1.pt` and `thgl-software_agg10_agegap_v1.pt`.
2. Launch sweeps with descriptive `RUN_PREFIX` and `FEATURE_TAG` names.
3. Archive `examples/linkproppred/thgl-forum/saved_checkpoints/<run>` and downloaded WandB run folders if you want offline records.
4. Summarize key metrics (best val/test MRR) and hyperparameters in `saved_results/training_logs/`.

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `ValueError: edge (...) is not in the 'val' evaluation set` | Make sure `edge_type` is **not** overwritten when using the agg10 schema; always keep `edge_type` = original 14 ID tensor.
| `RuntimeError: unable to write to file </torch_*...>: No space left on device` | Reduce `NUM_WORKERS`, set `OMP_NUM_THREADS=1`, or increase `/dev/shm`.
| `wandb: WARNING Tried to log to step ...` | Use the built-in global step counter in `tgn.py` or keep your own strictly increasing step variable.
| `size mismatch for memory_updater.weight_ih` when loading checkpoints | Delete old checkpoints and rerun with the new feature version; current filenames encode `agegap_v1` and message dim to prevent clashes.

---

## Citation

Please cite the original TGB and RelBench papers if you use this repo:

```
@article{huang2023temporal,
  title={Temporal Graph Benchmark for Machine Learning on Temporal Graphs},
  journal={NeurIPS Datasets and Benchmarks},
  year={2023}
}

@article{robinson2024relbench,
  title={RelBench: A Benchmark for Deep Learning on Relational Databases},
  journal={arXiv preprint arXiv:2407.20060},
  year={2024}
}
```

Happy hacking!
