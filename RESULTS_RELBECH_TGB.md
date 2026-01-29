# Results: RelBench baselines on (1) existing RelBench datasets and (2) TGB→RelBench translated exports

This repo has two evaluation “worlds”:
1) **Existing RelBench datasets** (official tasks/metrics via `relbench/examples/*`).
2) **Translated TGB datasets exported to a RelBench-style relational format** under `TGB2/relbench_exports/<dataset>/db/*.parquet` (evaluated with our apples-to-apples sampled-negative MRR protocol).

We report results below in two task categories:
- **Entity prediction** (node/entity property prediction; e.g. `tgbn-*`).
- **Link prediction** (edge prediction / recommendation; e.g. `tgbl-*`, `thgl-*`).

## A) Entity prediction

### A1) Translated TGB → RelBench exports: Dynamic Node Property Prediction (`tgbn-*`)

Model/script:
- GraphSAGE nodeprop baseline: `TGB2/baselines/graphsage_nodeprop.py`

Metrics:
- Official **NDCG@10** (plus sampled-negative **MRR** for consistency).

| Dataset | Val MRR | Val NDCG@10 | Test MRR | Test NDCG@10 |
|---|---:|---:|---:|---:|
| `tgbn-trade` | 0.9522 | 0.3769 | 0.9393 | 0.3765 |
| `tgbn-genre` | 0.8682 | 0.6169 | 0.8591 | 0.6054 |
| `tgbn-reddit` | 0.7804 | 0.6474 | 0.7555 | 0.6146 |
| `tgbn-token` | 0.4043 | 0.3763 | 0.3405 | 0.3098 |

Optional variant (same pipeline):
- `TGB2/baselines/graphsage_nodeprop.py --model gat` (relational GAT-style neighbor attention)

Real test (CPU; best epoch by val NDCG@10; all runs used `adj=val`, `fanouts=10,5`, `emb_dim=64`, `hidden_dim=64`, `num_heads=4`, `dropout=0.1`, `attn_dropout=0.1`, `num_neg_eval=100`, `max_train_events=200000`, `max_eval_events=20000`, `epochs=5`):

| Dataset | Val MRR | Val NDCG@10 | Test MRR | Test NDCG@10 |
|---|---:|---:|---:|---:|
| `tgbn-trade` | 0.9102 | 0.3849 | 0.8635 | 0.3401 |
| `tgbn-genre` | 0.6664 | 0.4263 | 0.6552 | 0.4139 |
| `tgbn-reddit` | 0.4326 | 0.3413 | 0.4067 | 0.3098 |
| `tgbn-token` | 0.2451 | 0.2303 | 0.2101 | 0.1935 |

Logs:
- `TGB2/logs/nodeprop_gat_real_20260129_063249/`

### A2) Existing RelBench datasets: entity prediction

Smoke results (existing RelBench datasets; official metrics):

| Dataset | Task | Model | Val (selected metric) | Test (selected metric) | Logs |
|---|---|---|---:|---:|---|
| `rel-event` | `user-repeat` | GAT | AP `0.5247` | AP `0.5017` | `relbench_runs/logs/rel-event_user-repeat_gnn_gat_smoke.txt` |

## B) Link prediction

### B0) Scripts/models used (existing RelBench vs translated TGB exports)

Existing RelBench (official datasets/tasks, MAP@10-style recommendation metrics):
- GraphSAGE baseline: `relbench/examples/gnn_recommendation.py` (with `--gnn sage`)
- TGN + GraphAttention (TransformerConv): `relbench/examples/tgn_attention_recommendation.py`

Translated TGB → RelBench exports (apples-to-apples sampled-negative MRR@100 on `TGB2/relbench_exports/*`):
- GraphSAGE-on-exports (projected edge graph): `TGB2/baselines/graphsage_linkpred.py`
- GraphSAGE (event-as-node, PK/FK relational graph) on exports:
  - `TGB2/baselines/relational_eventsage_linkpred.py` for `tgbl-*`
  - `TGB2/baselines/relational_eventsage_linkpred_thgl.py` for `thgl-*`
- TGN + GraphAttention (TransformerConv) on exports (streaming parquet, sampled-negative MRR@100):
  - `TGB2/baselines/tgn_attention_linkpred_exports.py`
  - Runner: `TGB2/scripts/run_tgn_attention_exports_mrr.sh`

### B1) Existing RelBench datasets (official recommendation evaluation; MAP@10)

| Dataset | Task | GraphSAGE val MAP@10 | TGN+Attn val MAP@10 | Notes |
|---|---|---:|---:|---|
| `rel-f1` | `driver-race-compete` | 0.06048 | 0.27821 | some evals may be skipped if a split is empty; see logs |
| `rel-hm` | `user-item-purchase` | 0.0006649 | 0.0009053 |  |
| `rel-stack` | `post-post-related` | 0.0024797 | 0.00017857 |  |

Logs:
- `relbench_runs/rel-f1_gnn_sage_epoch1.txt`
- `relbench_runs/rel-f1/tgn_attention_nocap_20260129_011052.txt`
- `relbench_runs/logs/smoke_existing_20260128_215438_rel-hm_user-item-purchase_gnn_sage_smoke.txt`
- `relbench_runs/logs/smoke_existing_20260128_215438_rel-hm_user-item-purchase_tgn_attention_smoke.txt`
- `relbench_runs/logs/smoke_existing_20260129_045608_rel-stack_post-post-related_gnn_sage_smoke.txt`
- `relbench_runs/logs/smoke_existing_20260129_045546_rel-stack_post-post-related_tgn_attention_smoke.txt`

### B2) Translated TGB → RelBench exports: Dynamic Link Property Prediction (`tgbl-*`) (sampled-negative MRR@100)

#### GraphSAGE variants on exports (projected edges vs event-as-node PK/FK)

Re-evaluated under the same protocol (`adj=val`, sampled negatives `K=100`, `max_eval=20000`):

| Dataset | GraphSAGE (projected edges) val | GraphSAGE (projected edges) test | GraphSAGE (event-as-node) val | GraphSAGE (event-as-node) test |
|---|---:|---:|---:|---:|
| `tgbl-wiki-v2` | 0.4203 | 0.3782 | 0.2757 | 0.2517 |
| `tgbl-review-v2` | 0.0932 | 0.0852 | 0.2596 | 0.2317 |
| `tgbl-coin` | 0.4541 | 0.3932 | 0.6064 | 0.5554 |
| `tgbl-comment` | 0.2089 | 0.1536 | 0.2896 | 0.2305 |
| `tgbl-flight` | 0.7082 | 0.6737 | 0.6357 | 0.5915 |

Notes:
- The **GraphSAGE (event-as-node)** column is the “exported RelBench schema” **event-as-node** architecture: each `events` row is treated as a node connected to its FK endpoints (`src_id`, `dst_id`) via PK/FK links, and message passing runs over that relational graph.
- These numbers come from the exports-based scripts listed above (`TGB2/baselines/relational_eventsage_linkpred*.py`).

#### TGN + GraphAttention (TransformerConv) on exports

Budget/config used:
- Epochs: 5
- Train cap: `max_train_events=50k`
- Eval cap: `max_val_events=20k`, `max_test_events=20k`
- Batch size: 96
- Embedding dims: `mem_dim=32`, `time_dim=16`, `emb_dim=32`

| Dataset | TGN+Attn exports val (best epoch) | TGN+Attn exports test |
|---|---:|---:|
| `tgbl-wiki-v2` | 0.3998 | 0.3384 |
| `tgbl-review-v2` | 0.2528 | 0.2457 |
| `tgbl-coin` | 0.5604 | 0.5067 |
| `tgbl-comment` | 0.2960 | 0.2098 |
| `tgbl-flight` | 0.4838 | 0.4566 |
| `tgbl-wiki` | 0.4139 | 0.3591 |
| `tgbl-review` | 0.2647 | 0.2465 |

Logs:
- `TGB2/logs/exports_tgn_attn_mrr_5ep_rerun_20260129_011938/`

### B3) Translated TGB → RelBench exports: Temporal Heterogeneous Graph Link Prediction (`thgl-*`) (sampled-negative MRR@100)

Baseline 1: **GraphSAGE (event-as-node)** on exports (hetero PK/FK row-as-node graph).  
Baseline 2: **TGN + GraphAttention** on exports (combined `events_edge_type_*` stream; negatives sampled per destination table).

| Dataset | GraphSAGE (event-as-node) val | GraphSAGE (event-as-node) test | TGN+Attn exports val (best epoch) | TGN+Attn exports test |
|---|---:|---:|---:|---:|
| `thgl-software` | 0.1388 | 0.1206 | 0.1367 | 0.1290 |
| `thgl-forum` | 0.4635 | 0.4401 | 0.3452 | 0.3527 |
| `thgl-myket` | 0.7264 | 0.7084 | 0.6614 | 0.6648 |
| `thgl-github` | 0.1441* | 0.1159* | 0.0782 | 0.0767 |

\* `thgl-github` GraphSAGE (event-as-node) was run with a reduced budget (1 epoch, `max_train_edges=50k`, `max_eval_edges=5k`, `K=50`), so it is not strictly comparable to the other rows.

Logs:
- `TGB2/logs/exports_tgn_attn_mrr_5ep_thgl_20260129_012117/`
