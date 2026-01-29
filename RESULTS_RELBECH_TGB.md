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
- Smoke test (budgeted): `tgbn-genre` epoch 1 → val MRR `0.8248`, val NDCG@10 `0.5358`; test MRR `0.8167`, test NDCG@10 `0.5425`

### A2) Existing RelBench datasets: entity prediction

Not summarized here yet (this repo’s current “existing RelBench” smoke focus was on link-prediction recommendation tasks like `rel-f1`/`rel-hm`).

## B) Link prediction

### B0) Scripts/models used (existing RelBench vs translated TGB exports)

Existing RelBench (official datasets/tasks, MAP@10-style recommendation metrics):
- GraphSAGE baseline: `relbench/examples/gnn_recommendation.py` (with `--gnn sage`)
- TGN + GraphAttention (TransformerConv): `relbench/examples/tgn_attention_recommendation.py`

Translated TGB → RelBench exports (apples-to-apples sampled-negative MRR@100 on `TGB2/relbench_exports/*`):
- GraphSAGE-on-exports (projected edge graph): `TGB2/baselines/graphsage_linkpred.py`
- RelEventSAGE-on-exports (PK/FK row-as-node relational graph):
  - `TGB2/baselines/relational_eventsage_linkpred.py` for `tgbl-*`
  - `TGB2/baselines/relational_eventsage_linkpred_thgl.py` for `thgl-*`
- TGN + GraphAttention (TransformerConv) on exports (streaming parquet, sampled-negative MRR@100):
  - `TGB2/baselines/tgn_attention_linkpred_exports.py`
  - Runner: `TGB2/scripts/run_tgn_attention_exports_mrr.sh`

### B1) Existing RelBench datasets (official recommendation evaluation; MAP@10)

| Dataset | Task | GraphSAGE val MAP@10 | TGN+Attn val MAP@10 | Test |
|---|---|---:|---:|---|
| `rel-f1` | `driver-race-compete` | 0.06048 | 0.27821 | `<empty test split>` |
| `rel-hm` | `user-item-purchase` | 0.0006649 | 0.0009053 | non-empty (see logs) |

Logs:
- `relbench_runs/rel-f1_gnn_sage_epoch1.txt`
- `relbench_runs/rel-f1/tgn_attention_nocap_20260129_011052.txt`
- `relbench_runs/logs/smoke_existing_20260128_215438_rel-hm_user-item-purchase_gnn_sage_smoke.txt`
- `relbench_runs/logs/smoke_existing_20260128_215438_rel-hm_user-item-purchase_tgn_attention_smoke.txt`

### B2) Translated TGB → RelBench exports: Dynamic Link Property Prediction (`tgbl-*`) (sampled-negative MRR@100)

#### GraphSAGE vs RelEventSAGE (exports)

Re-evaluated under the same protocol (`adj=val`, sampled negatives `K=100`, `max_eval=20000`):

| Dataset | GraphSAGE val | GraphSAGE test | RelEventSAGE val | RelEventSAGE test |
|---|---:|---:|---:|---:|
| `tgbl-wiki-v2` | 0.4203 | 0.3782 | 0.2757 | 0.2517 |
| `tgbl-review-v2` | 0.0932 | 0.0852 | 0.2596 | 0.2317 |
| `tgbl-coin` | 0.4541 | 0.3932 | 0.6064 | 0.5554 |
| `tgbl-comment` | 0.2089 | 0.1536 | 0.2896 | 0.2305 |
| `tgbl-flight` | 0.7082 | 0.6737 | 0.6357 | 0.5915 |

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

#### Reference: TGN + GraphAttention on original TGB TemporalData stream (not exports)

These runs use sampled-negative evaluation (K=100) but operate on the TGB TemporalData stream, not the relational exports:

| Dataset | TGN+Attn val (best epoch) | TGN+Attn test |
|---|---:|---:|
| `tgbl-wiki` | 0.5996 | 0.5023 |
| `tgbl-review` | 0.0882 | 0.1013 |
| `tgbl-coin` | 0.4423 | 0.4947 |
| `tgbl-comment` | 0.4377 | 0.3754 |
| `tgbl-flight` | 0.5926 | 0.5563 |

### B3) Translated TGB → RelBench exports: Temporal Heterogeneous Graph Link Prediction (`thgl-*`) (sampled-negative MRR@100)

Baseline 1: **RelEventSAGE** on exports (hetero PK/FK row-as-node graph).  
Baseline 2: **TGN + GraphAttention** on original TGB TemporalData stream (type-filtered negatives by destination node type).  
Baseline 3: **TGN + GraphAttention** on exports (combined `events_edge_type_*` stream; negatives sampled per destination table).

| Dataset | RelEventSAGE val | RelEventSAGE test | TGN+Attn (TGB stream) val | TGN+Attn (TGB stream) test |
|---|---:|---:|---:|---:|
| `thgl-software` | 0.1388 | 0.1206 | 0.2876 | 0.1960 |
| `thgl-forum` | 0.4635 | 0.4401 | 0.4087 | 0.3218 |
| `thgl-myket` | 0.7264 | 0.7084 | 0.2118 | 0.1533 |
| `thgl-github` | 0.1441* | 0.1159* | 0.1028 | 0.0956 |

\* `thgl-github` RelEventSAGE was run with a reduced budget (1 epoch, `max_train_edges=50k`, `max_eval_edges=5k`, `K=50`), so it is not strictly comparable to the other rows.

TGN + GraphAttention (TransformerConv) on exports (same budget/config as `tgbl-*` exports table above):

| Dataset | TGN+Attn exports val (best epoch) | TGN+Attn exports test |
|---|---:|---:|
| `thgl-software` | 0.1367 | 0.1290 |
| `thgl-forum` | 0.3452 | 0.3527 |
| `thgl-github` | 0.0782 | 0.0767 |
| `thgl-myket` | 0.6614 | 0.6648 |

Logs:
- `TGB2/logs/exports_tgn_attn_mrr_5ep_thgl_20260129_012117/`
