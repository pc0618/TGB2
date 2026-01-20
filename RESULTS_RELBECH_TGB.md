# Results: TGB → RelBench experiments (3 task families)

This file summarizes the experiments run in this repo after exporting TGB datasets into a RelBench-style relational format (`relbench_exports/<dataset>/db/*.parquet`) and evaluating baselines.

Unless otherwise noted:
- Link prediction metrics are **sampled-negative MRR** with a fixed `K=100` negatives per positive and an evaluation cap (typically `max_eval_edges/events=20000`).
- Node property prediction reports the official **NDCG@10** (and we additionally report sampled-negative MRR for consistency).

## 1) Dynamic Link Property Prediction (`tgbl-*`)

### A) GraphSAGE vs RelEventSAGE (apples-to-apples checkpoint eval)

Re-evaluated under the same protocol (`adj=val`, sampled negatives `K=100`, `max_eval=20000`):

| Dataset | GraphSAGE val | GraphSAGE test | RelEventSAGE val | RelEventSAGE test |
|---|---:|---:|---:|---:|
| `tgbl-wiki-v2` | 0.4203 | 0.3782 | 0.2757 | 0.2517 |
| `tgbl-review-v2` | 0.0932 | 0.0852 | 0.2596 | 0.2317 |
| `tgbl-coin` | 0.4541 | 0.3932 | 0.6064 | 0.5554 |
| `tgbl-comment` | 0.2089 | 0.1536 | 0.2896 | 0.2305 |
| `tgbl-flight` | 0.7082 | 0.6737 | 0.6357 | 0.5915 |

### B) TGN + GraphAttention (TransformerConv) on `tgbl-*`

These runs use the shared sampled-negative evaluation (K=100) but run directly on the TGB TemporalData stream (not the relational exports).

| Dataset | TGN+Attn val (best epoch) | TGN+Attn test |
|---|---:|---:|
| `tgbl-wiki` | 0.5996 | 0.5023 |
| `tgbl-review` | 0.0882 | 0.1013 |
| `tgbl-coin` | 0.4423 | 0.4947 |
| `tgbl-comment` | 0.4377 | 0.3754 |
| `tgbl-flight` | 0.5926 | 0.5563 |

### C) RelEventSAGE on non-v2 exports (`tgbl-wiki`, `tgbl-review`)

To compare like-for-like with the TGN dataset ids above, we exported `tgbl-wiki` and `tgbl-review` and trained RelEventSAGE on those exports.

| Dataset | RelEventSAGE val | RelEventSAGE test |
|---|---:|---:|
| `tgbl-wiki` | 0.3072 | 0.2663 |
| `tgbl-review` | 0.2679 | 0.2436 |

## 2) Temporal Heterogeneous Graph Link Prediction (`thgl-*`)

Baseline 1: **RelEventSAGE (hetero, relational event-as-node PK/FK graph)** on RelBench exports.  
Baseline 2: **TGN + GraphAttention** on TGB TemporalData with a comparable sampled-negative evaluator (type-filtered negatives by destination `node_type`).

| Dataset | RelEventSAGE val | RelEventSAGE test | TGN+Attn val (best epoch) | TGN+Attn test |
|---|---:|---:|---:|---:|
| `thgl-software` | 0.1388 | 0.1206 | 0.2876 | 0.1960 |
| `thgl-forum` | 0.4635 | 0.4401 | 0.4087 | 0.3218 |
| `thgl-myket` | 0.7264 | 0.7084 | 0.2118 | 0.1533 |
| `thgl-github` | 0.1441* | 0.1159* | 0.1028 | 0.0956 |

\* `thgl-github` RelEventSAGE was run with a reduced budget (1 epoch, `max_train_edges=50k`, `max_eval_edges=5k`, `K=50`), so it is not strictly comparable to the other rows.

## 3) Dynamic Node Property Prediction (`tgbn-*`)

Model: `baselines/graphsage_nodeprop.py` (GraphSAGE encoder; sampled-negative evaluation for MRR; NDCG@10 official).

| Dataset | Val MRR | Val NDCG@10 | Test MRR | Test NDCG@10 |
|---|---:|---:|---:|---:|
| `tgbn-trade` | 0.9522 | 0.3769 | 0.9393 | 0.3765 |
| `tgbn-genre` | 0.8682 | 0.6169 | 0.8591 | 0.6054 |
| `tgbn-reddit` | 0.7804 | 0.6474 | 0.7555 | 0.6146 |
| `tgbn-token` | 0.4043 | 0.3763 | 0.3405 | 0.3098 |

### NodeProp: relational GAT-style encoder (added)

`baselines/graphsage_nodeprop.py` also supports `--model gat` (sampled multi-head attention over sampled neighbors) under the same pipeline.

Smoke test (small budget):
- Dataset: `tgbn-genre`
- Epoch 1: val MRR `0.8248`, val NDCG@10 `0.5358`; test MRR `0.8167`, test NDCG@10 `0.5425`

