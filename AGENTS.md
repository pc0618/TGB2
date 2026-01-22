# Notes for agents: TGB → RelBench conversions (TGB2 repo)

This repo studies **relational deep learning on temporal datasets** by *bridging* the **Temporal Graph Benchmark (TGB)** API/data to **RelBench**-style relational schemas. The code here focuses on the TGB dataset `thgl-software`.

## Where the “translation” happens in this repo

There is no full “export to RelBench parquet/Database” pipeline in `TGB2/` itself; instead, TGB’s temporal edge stream (`TemporalData`) is **annotated** to *behave like* a relational schema during modeling, and the actual RelBench baseline is run from the separate `relbench` repo (see `README.md`).

Core files:
- `examples/linkproppred/thgl-forum/tgn.py`: builds/loads cached `TemporalData` tensors for a chosen schema variant and feature tag.
- `tgb/datasets/thgl_schema.py`: maps TGB edge types → “relation tables” (either fine-grained or aggregated) and appends an `event_type` column (encoded) when needed.
- `tgb/datasets/thgl_features.py`: adds temporal “age/gap” features (`agegap_v1`) to each event.
- `schemas/thgl_18_table_schema.dbml` / `schemas/thgl_10_table_schema.dbml`: DBML schema sketches of the *relational* view.

### `thgl-software`: schema variants used here

The repo uses two “schema variants” for the same underlying TGB dataset:

1) `default18` (fine-grained “one edge type = one event table”)
- Represented by the original TGB `edge_type` IDs (14 relation types).
- DBML reference: `schemas/thgl_18_table_schema.dbml` (4 entity tables + 14 event tables = 18 tables total).

2) `agg10` (aggregated “one entity-pair = one event table” + `event_type` column)
- Implemented in code by adding:
  - `TemporalData.agg_edge_type`: a smaller relation id space (6 aggregated relations in this repo).
  - an extra 1-dim categorical feature in `TemporalData.msg`: `event_type` encoded as an integer (opened/closed/…).
- DBML reference: `schemas/thgl_10_table_schema.dbml` (4 entity tables + 6 event tables = 10 tables total).

### Caching + feature attachment

`examples/linkproppred/thgl-forum/tgn.py` materializes and caches a processed `TemporalData` tensor per `(dataset, schema_variant, feature_version)`:
- cache path pattern: `datasets/schema_cache_augmented/{data}_{schema_variant}_{FEATURE_VERSION}.pt`
- on a cache miss:
  1. loads base TGB `TemporalData`
  2. applies schema conversion when `schema_variant != default18`
  3. appends `agegap_v1` features to every event

This is the “translation layer” you’ll extend if you add more schema variants or feature versions.

## How to convert *other* TGB datasets into RelBench (proposed recipe)

RelBench expects a **relational database** (a set of tables with primary keys and foreign keys) plus **temporal cutoffs** (`val_timestamp`, `test_timestamp`) and **tasks** exposing train/val/test tables.

A practical conversion recipe for any TGB dataset:

1) Load TGB data
- Use `tgb.linkproppred.dataset.LinkPropPredDataset` or `tgb.linkproppred.dataset_pyg.PyGLinkPropPredDataset` (or the nodeprop equivalents) to get:
  - edge list: `(src, dst, ts)` plus optional `edge_feat`/`msg`, weights, and optional `edge_type`.
  - node metadata when available (e.g., `thg` datasets include node types; some datasets include node features).

2) Choose temporal split cutoffs compatible with TGB
- TGB splits edges chronologically into 70%/15%/15%.
- For a RelBench-style dataset, define:
  - `val_timestamp`: timestamp at the 70% boundary
  - `test_timestamp`: timestamp at the 85% boundary
- Be careful with ties (many edges sharing the same timestamp): follow TGB’s edge-ordering and masks if you need exact parity.

3) Design the relational schema
- At minimum you need:
  - one or more **entity tables** (primary key = entity id)
  - one or more **event tables** (foreign keys to entity tables + timestamp column)
- For heterogeneous or multi-relation datasets:
  - either “one edge type = one event table” (like `default18`)
  - or “aggregate by entity-pair + `event_type` column” (like `agg10`)

4) Materialize to RelBench
- Emit each table as a dataframe-like artifact (e.g., parquet/csv) with:
  - explicit primary keys on entity tables
  - foreign key columns on event tables
  - a timestamp column on time-varying tables
- Implement a RelBench `Dataset` wrapper that returns `Database(tables=...)` and the two cutoffs.

5) Define tasks
- Link prediction in TGB is “predict the next tail given (head, relation, time)” evaluated with MRR (and negative sampling).
- In RelBench terms you typically define a task table with:
  - an input key (e.g., `src_id`), an as-of time, and optional context features
  - a target (e.g., the true `dst_id`, or a set/list of future `dst_id`s in a window)
- If you want *exact* TGB parity, you’ll likely implement a custom task/evaluator that reproduces TGB’s negative sampling + one-vs-many ranking.

## Dataset family conversion sketches (TGB 2.0)

Below are lightweight, schema-first conversion outlines for “other TGB datasets” beyond `thgl-software`.

### Dynamic Link Property Prediction (`tgbl-*`)

These are temporal edge streams (usually homogeneous or bipartite).

Typical relational schema:
- `nodes(node_id PK, …optional node features…)`
- `events(src_id FK->nodes, dst_id FK->nodes, ts, …edge features…, …weight…)`

Notes:
- Some datasets are bipartite in practice; you may prefer two entity tables (`left_nodes`, `right_nodes`) instead of a single `nodes`.
- `tgbl-flight` includes airport node features (`airport_node_feat.csv` in the TGB download); attach them to an `airports` table.

Task sketch:
- “next-destination” classification/ranking: each row is an observed event; inputs = `(src_id, ts[, features])`, target = `dst_id`.

### Temporal Heterogeneous Graph Link Prediction (`thgl-*`)

These include both **node types** and **edge types**.

Typical relational schema:
- one entity table per node type (e.g., `users`, `repos`, …)
- one event table per edge type **or** per entity-pair + `event_type`

Task sketch:
- “next-tail given (head, relation, time)” with MRR, optionally with relation-specific candidate sets (depending on node-type constraints).

### Temporal Knowledge Graph Link Prediction (`tkgl-*`)

These are temporal knowledge graph quadruples `(head, relation, tail, time)`; some also ship *static* triples.

Typical relational schema:
- `entities(entity_id PK)`
- `relations(relation_id PK, relation_name)`
- `facts(head_id FK->entities, relation_id FK->relations, tail_id FK->entities, ts)`
- optional `static_facts(head_id, relation_id, tail_id)` for `tkgl-wikidata` / `tkgl-smallpedia`

Task sketch:
- “future fact prediction”: given `(head, relation, time)` predict `tail`, and/or given `(relation, tail, time)` predict `head`.

### Dynamic Node Property Prediction (`tgbn-*`)

These are temporal interaction graphs with node labels evolving over time (often evaluated with NDCG@10).

Typical relational schema:
- `nodes(node_id PK)`
- `events(src_id, dst_id, ts, …edge features…)`
- `node_labels(node_id FK->nodes, ts, label)` or `node_labels(node_id, ts, label_vector)`

Task sketch:
- “predict the next label(s) for node at time t”: each row keyed by `(node_id, as_of_ts)` with target label(s) in a future window.

## Useful references

- TGB dataset overview and split protocol: https://tgb.complexdatalab.com/docs/dataset_overview/
- TGB link prediction datasets (`tgbl-*`): https://tgb.complexdatalab.com/docs/linkprop/
- TGB temporal heterogeneous datasets (`thgl-*`): https://tgb.complexdatalab.com/docs/thg/
- TGB temporal knowledge graph datasets (`tkgl-*`): https://tgb.complexdatalab.com/docs/tkg/
- TGB node prediction datasets (`tgbn-*`): https://tgb.complexdatalab.com/docs/nodeprop/
- RelBench quick start (Database/Task interfaces): https://relbench.stanford.edu/start/

## Online catalog: Temporal Graph Benchmark (TGB) datasets

Source: TGB official docs (accessed 2026-01-18). “Edges*” and “Steps” are reported as in the docs tables.

### Dynamic link property prediction (`tgbl-*`)

| Scale | Dataset | Package | Nodes | Edges* | Steps | Surprise | Metric | Notes (from docs) |
| --- | --- | --- | ---:| ---:| ---:| --- | --- | --- |
| `small` | `tgbl-wiki-v2` | `py-tgb==2.0.0` | 9,227 | 157,474 | 1,575 | `Yes` | `MRR` | Wikipedia co-editing over one month |
| `medium` | `tgbl-review-v2` | `py-tgb==2.0.0` | 3,529,440 | 19,128,226 | 1,280 | `Yes` | `MRR` | Amazon review ratings over 20 years |
| `large` | `tgbl-coin` | `py-tgb==2.0.0` | 638,486 | 22,809,580 | 1,295 | `Yes` | `MRR` | Cryptocurrency transactions over 3 years |
| `XL` | `tgbl-comment` | `py-tgb==2.0.0` | 994,790 | 44,660,625 | 2,294 | `Yes` | `MRR` | Reddit comments over 12 years |
| `XXL` | `tgbl-flight` | `py-tgb==2.0.0` | 181 | 67,261,804 | 1,027 | `Yes` | `MRR` | Flight network over 30 years |

### Dynamic node property prediction (`tgbn-*`)

| Scale | Dataset | Package | Nodes | Edges* | Steps | Surprise | Metric | Notes (from docs) |
| --- | --- | --- | ---:| ---:| ---:| --- | --- | --- |
| `small` | `tgbn-trade` | `py-tgb==2.0.0` | 255 | 507,625 | 31 | `Yes` | `NDCG@10` | UN trade networks over 31 years (annual) |
| `small` | `tgbn-genre` | `py-tgb==2.0.0` | 19,767 | 282,867 | 1,347 | `Yes` | `NDCG@10` | LastFM user-genre interactions over one month |
| `medium` | `tgbn-reddit` | `py-tgb==2.0.0` | 11,227 | 672,447 | 5,110 | `Yes` | `NDCG@10` | Reddit hyperlink network over 14 years |
| `large` | `tgbn-token` | `py-tgb==2.0.0` | 1,001 | 5,102,730 | 10,001 | `Yes` | `NDCG@10` | Blockchain user-token interactions |

### Temporal knowledge graph link prediction (`tkgl-*`)

| Scale | Dataset | Package | Nodes | Edges* | Steps | Time granularity | Metric | Notes (from docs) |
| --- | --- | --- | ---:| ---:| ---:| --- | --- | --- |
| `small` | `tkgl-smallpedia` | `py-tgb==2.0.0` | 94,274 | 565,375 | 2,421 | `year` | `MRR` | Wikipedia temporal relations (with static relations) |
| `medium` | `tkgl-polecat` | `py-tgb==2.0.0` | 18,306 | 5,810,421 | 1826 | `day` | `MRR` | GitHub temporal relations (2018-01 to 2022-12) |
| `large` | `tkgl-icews` | `py-tgb==2.0.0` | 32,996 | 6,728,276 | 12 | `month` | `MRR` | ICEWS temporal relations (1995-01 to 2022-12) |
| `XL` | `tkgl-wikidata` | `py-tgb==2.0.0` | 12,324,201 | 121,438,407 | 2,305 | `year` | `MRR` | Wikidata temporal relations (with static relations) |

### Temporal heterogeneous graph link prediction (`thgl-*`)

| Scale | Dataset | Package | Nodes | Edges* | Steps | Metric | Notes (from docs) |
| --- | --- | --- | ---:| ---:| ---:| --- | --- |
| `small` | `thgl-software` | `py-tgb==2.0.0` | 681,927 | 1,489,806 | 744 | `MRR` | GitHub interactions in January 2024 |
| `medium` | `thgl-forum` | `py-tgb==2.0.0` | 1,578,304 | 32,213,501 | 744 | `MRR` | Reddit interactions in January 2014 |
| `large` | `thgl-github` | `py-tgb==2.0.0` | 1,331,643 | 12,245,744 | 744 | `MRR` | GitHub interactions in March 2024 |
| `XL` | `thgl-myket` | `py-tgb==2.0.0` | 1,987,232 | 46,824,394 | 225 | `MRR` | Myket app interactions (2020-06 to 2021-01) |

## Notes: `weight` vs `split` (schema design)

TGB loaders expose (at least) two edge-level scalars across many datasets:
- `w` / `weight`: an interaction weight/value from the raw dataset (often `1.0`, but can be meaningful, e.g. transaction amount, rating, score).
- train/val/test split: produced by TGB’s chronological split logic (`train_mask`, `val_mask`, `test_mask`) and/or by a chosen cutoff timestamp.

For a *relational schema* (and for RelBench), it’s usually better to:
- **Keep** `weight` as an attribute column *only if it’s semantically meaningful* (or if you want the option to use it as a feature).
- **Avoid** storing `split` as a physical column in the database schema. Instead, store timestamps and define dataset cutoffs:
  - RelBench uses `Dataset.val_timestamp` and `Dataset.test_timestamp`, and tasks derive train/val/test windows from these timestamps.
  - This mirrors what we do for `thgl-software`: splits are applied “post hoc” in the loader/evaluator; they are not part of the relational schema.

In other words: `split` is *training protocol metadata*, while `weight` is *data* (sometimes feature, sometimes constant noise).

## RelBench translation (local pipeline)

This repo now includes a local “TGB → RelBench” exporter that materializes a RelBench `Database` (parquet tables + table metadata) without embedding `split` columns in the schema.

### Export command

From `TGB2/`:

```bash
.venv/bin/python scripts/export_to_relbench.py --dataset <DATASET_NAME> --root datasets --out_dir relbench_exports
```

Notes:
- `tkgl-*` is intentionally rejected (knowledge graphs are out of scope here).
- Some TGB docs dataset ids include `-v2`; the exporter maps these to the library ids where needed (e.g., `tgbl-wiki-v2` → `tgbl-wiki`).
- Output layout: `relbench_exports/<DATASET_NAME>/`
  - `db/*.parquet` (RelBench `Database.save()` format)
  - `metadata.json` (includes `val_timestamp` / `test_timestamp`)
  - `schema.dbml` (for dbdiagram.io visualization)

### Loading as a RelBench Dataset

Use `relbench_tgb.get_exported_dataset()`:

```python
from relbench_tgb import get_exported_dataset

ds = get_exported_dataset("tgbl-wiki-v2", exports_root="relbench_exports")
db_full = ds.get_db(upto_test_timestamp=False)
```

`Dataset.get_db()` defaults to `upto_test_timestamp=True` to prevent test leakage; set `False` when you need the full database (e.g., to compute labels after the test cutoff).

### Upstream RelBench PR prep (local artifacts + hashes)

We prepared a PR against `snap-stanford/relbench` to add the converted TGB datasets + tasks (all non-KG families: `tgbl-*`, `tgbn-*`, `thgl-*`).

RelBench repo state:
- Repo: `/home/pc0618/relbench`
- Branch: `add-tgb-datasets`
- Commits:
  - `fb07391`: add `relbench/relbench/datasets/tgb.py`, `relbench/relbench/tasks/tgb.py`, and register all `rel-tgb-*` datasets/tasks
  - `0917ac0`: fix `thgl-github` task registrations to match actual type-pairs; switch tasks to “next” semantics; add SHA256 entries to `relbench/relbench/datasets/hashes.json` + `relbench/relbench/tasks/hashes.json`

Local zip artifacts (hosting TBD):
- Dataset zips: `/home/pc0618/relbench_tgb_artifacts/rel-tgb-*/db.zip`
- Task zips: `/home/pc0618/relbench_tgb_artifacts/rel-tgb-*/tasks/*.zip`
- Hash manifests (used to populate RelBench hashes.json files):
  - `/home/pc0618/relbench_tgb_artifacts/hashes_datasets.json`
  - `/home/pc0618/relbench_tgb_artifacts/hashes_tasks.json`

Remaining step for `relbench/CONTRIBUTING.md`:
- Decide where to host `db.zip` and `<task>.zip` files and include the hosted links in the PR description (hashes are already computed/added).

### Status: Dynamic Link Property Prediction exports (`tgbl-*`)

Exports were generated under `TGB2/relbench_exports/` using `scripts/export_to_relbench.py`.

Common schema (homogeneous case):
- `nodes(node_id PK)`
- `events(event_id PK, src_id FK->nodes, dst_id FK->nodes, event_ts (time_col), weight)`

Special case (wiki-style bipartite):
- `src_nodes(src_id PK)`
- `dst_nodes(dst_id PK)`
- `events(event_id PK, src_id FK->src_nodes, dst_id FK->dst_nodes, event_ts (time_col), weight)`

Generated (row counts from parquet metadata):
- `tgbl-wiki-v2` (`tgb_internal_name=tgbl-wiki`): `src_nodes=8,227`, `dst_nodes=1,000`, `events=157,474`
- `tgbl-review-v2` (`tgb_internal_name=tgbl-review`): `nodes=352,637`, `events=4,873,540`
- `tgbl-coin` (`tgb_internal_name=tgbl-coin`): `nodes=638,486`, `events=22,809,486`
- `tgbl-comment` (`tgb_internal_name=tgbl-comment`): `nodes=994,790`, `events=44,314,507`
- `tgbl-flight` (`tgb_internal_name=tgbl-flight`): `nodes=18,143`, `events=67,169,570`

Notes / caveats:
- No `split` column is stored; the exporter stores `val_timestamp_s` / `test_timestamp_s` in `metadata.json` and RelBench tasks should derive splits from timestamps.
- `weight` is whatever TGB exposes as `w`:
  - For some datasets it is meaningful (e.g., transaction value / score); for others it is constant `1.0`.
  - In TGB’s own preprocessing, some datasets also fold `w` into `edge_feat` (`msg`). In our RelBench schema we keep a single `weight` column and do not duplicate it.
- Edge feature vectors (`edge_feat` / `msg`) are **not** exported as relational columns for these tgbl datasets yet (to avoid the “msg_000…” explosion). This means:
  - `tgbl-wiki-v2`: 172-dim edge features exist in TGB; currently omitted from the RelBench database (only `weight=1.0` kept).
  - `tgbl-flight`: 16-dim encoded features exist in TGB (callsign/typecode encoding); currently omitted (only `weight=1.0` kept).
  - `tgbl-comment`: has additional edge features in TGB (and also appends score as part of the message); currently omitted beyond `weight`.
- Large datasets (`tgbl-comment`, `tgbl-flight`) can be expensive to load with RelBench’s current in-memory `Database.load()` since it materializes full tables into pandas.

## GCN baseline (static, link prediction)

Implemented a simple static GCN baseline that trains on edges with `event_ts <= val_timestamp` and evaluates MRR on future edges using sampled negatives:
- Script: `baselines/gcn_linkpred.py`
- Input: RelBench-exported parquet databases under `relbench_exports/<dataset>/db/`
- Supports bipartite exports (wiki) by internally offsetting dst node ids.
- For very large datasets, use sampling flags to keep memory/compute bounded:
  - `--adj_max_edges` to subsample edges used for message passing
  - `--max_train_edges` / `--max_val_edges` / `--max_test_edges` to subsample supervision/eval edges

Smoke run on `tgbl-wiki-v2` (CPU, undirected message passing, 1 epoch) completed successfully.

## GraphSAGE baseline (sampled, scalable link prediction)

For large `tgbl-*` datasets where full-batch GCN is not practical, use a sampled 2-layer GraphSAGE baseline built on a **disk-backed CSR adjacency** (no giant `edge_index` in RAM).

Components:
- CSR builder: `scripts/build_csr_adj.py`
- Sampled GraphSAGE trainer: `baselines/graphsage_linkpred.py`

Workflow (example: `tgbl-wiki-v2`):
1) Build adjacency up to the training cutoff (RelBench-style: use edges with `event_ts <= val_timestamp`):
   - `scripts/build_csr_adj.py --dataset tgbl-wiki-v2 --exports_root relbench_exports --upto val --undirected`
2) Train/eval with sampled neighborhoods:
   - `baselines/graphsage_linkpred.py --dataset tgbl-wiki-v2 --exports_root relbench_exports --adj val --undirected --fanouts 15,10`

Scaling knobs for `tgbl-coin`, `tgbl-comment`, `tgbl-flight`:
- Build CSR with a larger `--batch_size` to speed up scanning parquet.
- Use `--max_train_edges/--max_val_edges/--max_test_edges` to subsample supervision/eval sets.
- Reduce `--fanouts` and/or `--batch_size` to fit memory/compute.

### tgbl-coin (debugged run)

Built undirected CSR adjacency up to the training cutoff:
- `scripts/build_csr_adj.py --dataset tgbl-coin --exports_root relbench_exports --upto val --undirected --batch_size 1000000`

Trained sampled GraphSAGE (CPU) end-to-end and wrote a checkpoint:
- `baselines/graphsage_linkpred.py --dataset tgbl-coin --exports_root relbench_exports --adj val --undirected --epochs 3 --batch_size 2048 --fanouts 10,5 --num_neg_eval 50 --max_train_edges 200000 --max_val_edges 50000 --max_test_edges 50000 --save_dir saved_models/graphsage`
- Output: `saved_models/graphsage/graphsage_tgbl-coin.pt`

### Remaining tgbl GraphSAGE checkpoints (sampled training)

All runs used `--wandb --wandb_mode offline` (creates local runs under `TGB2/wandb/` without needing credentials).

Checkpoints written to `saved_models/graphsage/`:
- `graphsage_tgbl-wiki-v2.pt` (and `.json`)
- `graphsage_tgbl-review-v2.pt` (and `.json`)
- `graphsage_tgbl-coin.pt` (and `.json`)
- `graphsage_tgbl-comment.pt` (and `.json`)
- `graphsage_tgbl-flight.pt` (and `.json`)

Note: These are “end-to-end trained” under a *sampling budget* (`--max_train_edges`, etc.). Increase those limits and/or epochs to approach full-data training, especially for `tgbl-comment` and `tgbl-flight`.

## Relational baseline (PK/FK graph over the exported schema)

### Why this baseline exists (environment constraints)
- `relbench==2.0.1` includes a modeling stack under `relbench.modeling.*` that depends on a *different* `torch_frame` package than the one available via PyPI here.
- The available PyPI `torch-frame==1.7.5` imports vision utilities (`torchvision`) and fails to import cleanly in this CPU-only environment due to missing compiled ops.
- Git installs are blocked here, so we cannot pull the correct PyG/RelBench `torch-frame` implementation.

Result: we implement a **schema-faithful relational GNN baseline** in-repo, operating directly on the exported tables and PK/FK relations, without `torch-frame`.

### Relational representation (event-as-node)
For `tgbl-*` exports, the schema is:
- homogeneous: `nodes(node_id)` + `events(event_id, src_id, dst_id, event_ts, weight)`
- bipartite (`tgbl-wiki-v2`): `src_nodes(src_id)` + `dst_nodes(dst_id)` + `events(event_id, src_id, dst_id, event_ts, weight)`

We build a *relational* message passing graph by treating each row in `events` as an **event node** and connecting it via foreign keys:
- `src_node → event_node` using `events.src_id`
- `dst_node → event_node` using `events.dst_id`

Node embeddings are ID embeddings. Event embeddings are computed from:
- `log1p(weight)` and normalized `event_ts` (small MLP), plus
- the incident src/dst node embeddings.

### Files
- Build relational CSR adjacencies + per-event arrays:
  - `scripts/build_rel_event_csr.py`
- Train relational baseline (EventSAGE-style):
  - `baselines/relational_eventsage_linkpred.py`

### Commands
Build relational adjacencies up to cutoffs (recommended):
- `scripts/build_rel_event_csr.py --dataset <name> --exports_root relbench_exports --upto val`
- `scripts/build_rel_event_csr.py --dataset <name> --exports_root relbench_exports --upto test`

Train/eval (sampled; W&B offline by default):
- `baselines/relational_eventsage_linkpred.py --dataset <name> --exports_root relbench_exports --epochs 3 --batch_size 2048 --fanout 10 --max_train_edges 200000 --num_neg_eval 100 --max_eval_edges 20000 --wandb --wandb_mode offline --save_dir saved_models/releventsage`

### Results (MRR, sampled negatives; 3 epochs; `max_train_edges=200k`, `max_eval_edges=20k`, `num_neg_eval=100`)
- `tgbl-coin`: val `0.6195`, test `0.5361` → `saved_models/releventsage/releventsage_tgbl-coin.pt`
- `tgbl-flight`: val `0.6137`, test `0.6057` → `saved_models/releventsage/releventsage_tgbl-flight.pt`
- `tgbl-review-v2`: val `0.2471`, test `0.2288` → `saved_models/releventsage/releventsage_tgbl-review-v2.pt`
- `tgbl-wiki-v2`: val `0.2793`, test `0.2399` → `saved_models/releventsage/releventsage_tgbl-wiki-v2.pt`
- `tgbl-comment`: val `0.2729`, test `0.2271` → `saved_models/releventsage/releventsage_tgbl-comment.pt`

### Comparison vs GraphSAGE (apples-to-apples checkpoint eval)

The numbers above were printed during training runs and may differ across scripts because:
- `graphsage_linkpred.py` evaluates using the training adjacency (`adj=val`) only.
- `relational_eventsage_linkpred.py` evaluates **test** using an adjacency that can include more history (`adj=test` by default).

To compare representations fairly, we re-evaluated the saved checkpoints under the **same protocol**:
- Message passing graph: `adj=val` for both methods (no test leakage).
- Evaluation: sampled-negative MRR with `num_neg_eval=100`, `max_eval_edges=20000`.

Results (val / test):
- `tgbl-wiki-v2`: GraphSAGE `0.4203 / 0.3782` vs RelEventSAGE `0.2757 / 0.2517` (GraphSAGE better)
- `tgbl-review-v2`: GraphSAGE `0.0932 / 0.0852` vs RelEventSAGE `0.2596 / 0.2317` (RelEventSAGE better)
- `tgbl-coin`: GraphSAGE `0.4541 / 0.3932` vs RelEventSAGE `0.6064 / 0.5554` (RelEventSAGE better)
- `tgbl-comment`: GraphSAGE `0.2089 / 0.1536` vs RelEventSAGE `0.2896 / 0.2305` (RelEventSAGE better)
- `tgbl-flight`: GraphSAGE `0.7082 / 0.6737` vs RelEventSAGE `0.6357 / 0.5915` (GraphSAGE better)

Interpretation (under this simplified protocol):
- RelEventSAGE tends to win on datasets where `weight` is informative/heavy-tailed and event records matter (`review/coin/comment`).
- GraphSAGE tends to win where the task is closer to pure structural proximity under our current schema (and we do not use high-dim `msg_*` features), notably `wiki` and `flight`.

### Caveats
- Metrics are from our *sampled-negative MRR* evaluation (same as the GNN baselines here), not the official TGB evaluator.
- We intentionally keep `msg_*` vectors out of the schema; this baseline uses only `(event_ts, weight)` plus graph structure.

## THGL: RelBench translation + relational baseline

We extended the same “multi-table, schema-first” strategy used for `thgl-software` to the remaining Temporal Heterogeneous Graph Link (THGL) datasets:
- Each **node type** becomes its own entity table: `nodes_type_<t>(node_type_<t>_id)`.
- Each **edge type** becomes its own event table: `events_edge_type_<e>(event_id, src_id FK->nodes_type_src, dst_id FK->nodes_type_dst, event_ts, weight)`.
- No `split` columns are stored; splits are derived from `metadata.json` cutoffs as before.

### Export (streaming support for large THGL)
- Exporter: `scripts/export_to_relbench.py`
- For `thgl-*` datasets, event tables are now written in a **streaming** manner (parquet chunk writer), to avoid building huge pandas DataFrames per edge type.

Exports created under `relbench_exports/`:
- `thgl-software` (already existed)
- `thgl-forum`
- `thgl-github`
- `thgl-myket`

### Relational adjacency / event arrays (THGL)
To train a relational baseline without loading full tables into pandas, we build node→event CSR adjacencies per node type and unified per-event arrays:
- Builder: `scripts/build_rel_event_csr_thgl.py`
- Commands:
  - `scripts/build_rel_event_csr_thgl.py --dataset <thgl-name> --exports_root relbench_exports --upto val`
  - `scripts/build_rel_event_csr_thgl.py --dataset <thgl-name> --exports_root relbench_exports --upto test`

### Baseline + results (MRR)
Baseline: `baselines/relational_eventsage_linkpred_thgl.py` (event-as-node PK/FK graph; type-correct negative sampling; sampled-negative MRR).

Runs completed (W&B `offline`):
- `thgl-software` (3 epochs, `max_train_edges=200k`, `max_eval_edges=20k`, `num_neg_eval=100`):
  - val `0.1388`, test `0.1206` → `saved_models/releventsage/releventsage_thgl-software.pt`
- `thgl-forum` (3 epochs, `max_train_edges=200k`, `max_eval_edges=20k`, `num_neg_eval=100`):
  - val `0.4635`, test `0.4401` → `saved_models/releventsage/releventsage_thgl-forum.pt`
- `thgl-myket` (3 epochs, `max_train_edges=200k`, `max_eval_edges=20k`, `num_neg_eval=100`):
  - val `0.7264`, test `0.7084` → `saved_models/releventsage/releventsage_thgl-myket.pt`
- `thgl-github` (reduced budget due to runtime; 1 epoch, `max_train_edges=50k`, `max_eval_edges=5k`, `num_neg_eval=50`):
  - val `0.1441`, test `0.1159` → `saved_models/releventsage/releventsage_thgl-github.pt`

Notes:
- `thgl-github` is the slowest of the set under this baseline because it has 4 node types and 14 edge types, and evaluation requires scanning all edge-type parquet tables to sample val/test edges.

## `thgl-software`: TGN + GraphAttention (budgeted + sampled-negative eval)

This repo also includes the “TGN + GraphAttention” runner (used historically for `thgl-software`) at:
- `examples/linkproppred/thgl-forum/tgn.py` (run with `--data thgl-software`)

To make it comparable to the RelBench baselines, we added:
- `--eval_mode sampled --num_neg_eval K`: sampled-negative MRR with type-filtered negatives (by `node_type`), instead of TGB’s official one-vs-many negatives.
- `--max_train_events / --max_val_events / --max_test_events`: caps split sizes to match the RelBench budget style (`max_train_edges`, `max_eval_edges`).

Run (default18, CPU):
- `--max_train_events 200000 --max_val_events 20000 --max_test_events 20000 --eval_mode sampled --num_neg_eval 100 --num_epoch 3`

Result (sampled-negative MRR):
- val MRR by epoch: `0.1121`, `0.1332`, `0.1423`
- test MRR (best val checkpoint): `0.1384`

Scaled-up run (same eval; full train split, CPU):
- `--max_train_events 0 --max_val_events 20000 --max_test_events 20000 --eval_mode sampled --num_neg_eval 100 --num_epoch 3`

Result (sampled-negative MRR):
- val MRR by epoch: `0.2494`, `0.2676`, `0.2876`
- test MRR (best val checkpoint): `0.1960`

Note: these are now directly comparable (same “sampled-negative MRR” style) to the RelEventSAGE THGL baseline above, though the models differ substantially (TGN temporal memory vs event-as-node relational GNN).

## TGN + GraphAttention: additional runs (THGL + TGBL; sampled-negative MRR)

We extended `examples/linkproppred/thgl-forum/tgn.py` to also run on `tgbl-*` datasets:
- THGL (`thgl-*`): uses cached `default18` + `agegap_v1` (`datasets/schema_cache_augmented/<data>_default18_agegap_v1.pt`) and evaluates with **type-filtered** sampled negatives (by destination `node_type`).
- TGBL (`tgbl-*`): runs directly on the raw `TemporalData` stream and evaluates with **uniform** sampled negatives from the destination id range (`[min(dst), max(dst)]`).

Important compatibility note (this repo’s `tgb` snapshot):
- `tgbl-wiki-v2` / `tgbl-review-v2` are **not supported** by `PyGLinkPropPredDataset`; use `tgbl-wiki` / `tgbl-review` instead.

All runs below use: `--eval_mode sampled --num_neg_eval 100 --num_epoch 3 --num_run 1 --bs 200`.

### THGL (budgeted train; `max_train_events=200000`, `max_val_events=20000`, `max_test_events=20000`)

- Dataset metadata observed via `PyGLinkPropPredDataset` in this environment:
  - `thgl-software` / `thgl-github`: 4 node types, 14 relations
  - `thgl-forum` / `thgl-myket`: 2 node types, 2 relations

- `thgl-github`: val MRR by epoch `0.1028, 0.0973, 0.0928`; test MRR `0.0956`
- `thgl-forum`: val MRR by epoch `0.2508, 0.4087, 0.3397`; test MRR `0.3218`
- `thgl-myket`: val MRR by epoch `0.2118, 0.1213, 0.1063`; test MRR `0.1533`

### TGBL (mixed: `wiki` full-train; others budgeted)

- `tgbl-wiki` (full train split; `max_train_events=0`, `max_val_events=20000`, `max_test_events=20000`): val MRR by epoch `0.3700, 0.5493, 0.5996`; test MRR `0.5023`
- `tgbl-review` (`max_train_events=200000`, `max_val_events=20000`, `max_test_events=20000`): val MRR by epoch `0.0882, 0.0797, 0.0754`; test MRR `0.1013`
- `tgbl-coin` (`max_train_events=200000`, `max_val_events=20000`, `max_test_events=20000`): val MRR by epoch `0.3228, 0.3831, 0.4423`; test MRR `0.4947`
- `tgbl-comment` (`max_train_events=200000`, `max_val_events=20000`, `max_test_events=20000`): val MRR by epoch `0.4377, 0.3928, 0.3830`; test MRR `0.3754`
- `tgbl-flight` (`max_train_events=200000`, `max_val_events=20000`, `max_test_events=20000`): val MRR by epoch `0.4237, 0.5498, 0.5926`; test MRR `0.5563`

### New: RelEventSAGE on non-v2 exports (`tgbl-wiki`, `tgbl-review`)

We exported the *non-v2* datasets into `relbench_exports/` and trained the same RelEventSAGE baseline, so results can be compared directly to the TGN+GraphAttention runs above.

Export:
- `PYTHONPATH=. .venv/bin/python scripts/export_to_relbench.py --dataset tgbl-wiki --out_dir relbench_exports`
- `PYTHONPATH=. .venv/bin/python scripts/export_to_relbench.py --dataset tgbl-review --out_dir relbench_exports`

Adjacency build (cutoff = `val`):
- `PYTHONPATH=. .venv/bin/python scripts/build_rel_event_csr.py --dataset tgbl-wiki --exports_root relbench_exports --upto val`
- `PYTHONPATH=. .venv/bin/python scripts/build_rel_event_csr.py --dataset tgbl-review --exports_root relbench_exports --upto val`

Training (sampled-negative MRR; `K=100`; `epochs=3`; `train_adj=val`; `eval_adj_test=val`):
- `tgbl-wiki` (full train split): val MRR `0.3072`, test MRR `0.2663` → `saved_models/releventsage/releventsage_tgbl-wiki.pt`
- `tgbl-review` (`max_train_edges=200000`): val MRR `0.2679`, test MRR `0.2436` → `saved_models/releventsage/releventsage_tgbl-review.pt`

Artifacts:
- Results JSON: `examples/linkproppred/thgl-forum/saved_results/TGN_<dataset>_results.json`
- Model checkpoints: `examples/linkproppred/thgl-forum/saved_models/TGN_<dataset>_<feature_tag>_msg<d>_<seed>_<run>.pth`

## NodeProp (`tgbn-*`): memory-safe export + baseline (MRR + NDCG@10)

Problem encountered:
- The official TGB nodeprop loader can materialize a **dense** `node_label_dict` with per-(ts,node) dense vectors; for `tgbn-token` this can be prohibitively memory-heavy.

Mitigation / guardrails added:
- `scripts/export_to_relbench.py` now loads nodeprop **edges without labels** from the processed `ml_<dataset>.pkl` + `ml_<dataset>_edge.pkl` files, to avoid constructing dense label dicts.
- For `tgbn-token` and `tgbn-reddit`, label tables are stream-written directly from `<dataset>_node_labels.csv` using mapping pickles, emitting only non-zero label weights.
- `baselines/graphsage_nodeprop.py` now has an RSS guardrail (`--max_rss_gb`, default 50 GB) and uses disk-backed CSR adjacency/label CSR arrays.

Schema notes:
- Entity table: `nodes(node_id)`
- Label universe table: `labels(label_id)` (label ids are in `[0, num_classes)` and correspond to the “label nodes” used by these datasets)
- Interaction/event table: `events(event_id, src_id, dst_id, event_ts, weight)` (stored weight = first edge feature; for `tgbn-token`, first feature is log-normalized as in TGB preprocessing)
- Label events (targets) as 2 tables:
  - `label_events(label_event_id, src_id, label_ts)`
  - `label_event_items(item_id, label_event_id, label_id, label_weight)`
- No `split` column stored; train/val/test are derived from `metadata.json` cutoffs.

Commands (cache build):
- `scripts/build_csr_adj.py --dataset <tgbn-*> --upto val`
- `scripts/build_label_event_csr.py --dataset <tgbn-*>`

Baseline runs (small/guardrailed):
- Model: `baselines/graphsage_nodeprop.py` (Sampled GraphSAGE encoder; BPR loss; sampled-negative evaluation)
- Params: `epochs=2`, `batch_size=256`, `fanouts=5,2`, `emb_dim=32`, `hidden_dim=32`, `num_neg_eval=50`, `max_train_events=50k`, `max_eval_events=5k`, adjacency cutoff=`val`, directed CSR.

Results (val / test):
- `tgbn-trade`:
  - epoch 2: val MRR `0.9522`, val NDCG@10 `0.3769`; test MRR `0.9393`, test NDCG@10 `0.3765`
- `tgbn-genre`:
  - epoch 2: val MRR `0.8682`, val NDCG@10 `0.6169`; test MRR `0.8591`, test NDCG@10 `0.6054`
- `tgbn-reddit`:
  - epoch 2: val MRR `0.7804`, val NDCG@10 `0.6474`; test MRR `0.7555`, test NDCG@10 `0.6146`
- `tgbn-token`:
  - epoch 2: val MRR `0.4043`, val NDCG@10 `0.3763`; test MRR `0.3405`, test NDCG@10 `0.3098`

Notes:
- Official `tgbn-*` metric is NDCG@10; we also report sampled-negative MRR for consistency with our other baselines.

### New: Relational GAT-style encoder (nodeprop)

`baselines/graphsage_nodeprop.py` now supports `--model gat`, which swaps the neighbor-mean aggregation for a **sampled multi-head dot-product attention** aggregator (2 layers; attention over sampled neighbors). This keeps the **same** RelBench export, CSR adjacency, and MRR + NDCG@10 evaluation protocol as the GraphSAGE baseline.

Smoke test (to validate correctness; small budget):
- `PYTHONPATH=. .venv/bin/python baselines/graphsage_nodeprop.py --dataset tgbn-genre --model gat --epochs 1 --batch_size 512 --fanouts 5,2 --emb_dim 32 --hidden_dim 32 --num_heads 4 --num_neg_eval 50 --max_train_events 5000 --max_eval_events 1000 --adj val`
- Result: epoch 1 val MRR `0.8248`, val NDCG@10 `0.5358`; test MRR `0.8167`, test NDCG@10 `0.5425`
