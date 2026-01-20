#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from baselines.graphsage_linkpred import CSRAdjacency, SampledGraphSAGE, _load_meta


class _GATLayer(nn.Module):
    def __init__(
        self,
        *,
        self_in_dim: int,
        neigh_in_dim: int,
        out_dim: int,
        num_heads: int,
        attn_dropout: float,
    ):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"out_dim={out_dim} must be divisible by num_heads={num_heads}")
        self.out_dim = int(out_dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(out_dim // num_heads)
        self.scale = float(self.head_dim) ** -0.5

        self.lin_q = nn.Linear(self_in_dim, out_dim, bias=False)
        self.lin_k = nn.Linear(neigh_in_dim, out_dim, bias=False)
        self.lin_v = nn.Linear(neigh_in_dim, out_dim, bias=False)
        self.lin_self = nn.Linear(self_in_dim, out_dim, bias=False)
        self.drop_attn = nn.Dropout(float(attn_dropout))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in (self.lin_q, self.lin_k, self.lin_v, self.lin_self):
            nn.init.xavier_uniform_(m.weight)

    def forward(self, h_self: torch.Tensor, h_neigh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_self: [N, D_self]
            h_neigh: [N, fanout, D_neigh]
        Returns:
            out: [N, out_dim]
        """
        n = h_self.size(0)
        fanout = h_neigh.size(1)

        q = self.lin_q(h_self).view(n, self.num_heads, self.head_dim)  # [N,Hd,K]
        k = self.lin_k(h_neigh).view(n, fanout, self.num_heads, self.head_dim)  # [N,F,Hd,K]
        v = self.lin_v(h_neigh).view(n, fanout, self.num_heads, self.head_dim)

        scores = (k * q.unsqueeze(1)).sum(dim=-1) * self.scale  # [N,F,Hd]
        attn = torch.softmax(scores, dim=1)  # [N,F,Hd]
        attn = self.drop_attn(attn)

        agg = (attn.unsqueeze(-1) * v).sum(dim=1)  # [N,Hd,K]
        agg = agg.reshape(n, self.out_dim)  # [N, out_dim]
        out = self.lin_self(h_self) + agg
        return out


class SampledGAT(nn.Module):
    def __init__(
        self,
        *,
        num_nodes: int,
        emb_dim: int,
        hidden_dim: int,
        fanouts: tuple[int, int],
        dropout: float,
        num_heads: int,
        attn_dropout: float,
    ):
        super().__init__()
        self.emb = nn.Embedding(int(num_nodes), int(emb_dim))
        self.fanouts = fanouts
        self.gat1 = _GATLayer(
            self_in_dim=int(emb_dim),
            neigh_in_dim=int(emb_dim),
            out_dim=int(hidden_dim),
            num_heads=int(num_heads),
            attn_dropout=float(attn_dropout),
        )
        self.gat2 = _GATLayer(
            self_in_dim=int(emb_dim),
            neigh_in_dim=int(hidden_dim),
            out_dim=int(hidden_dim),
            num_heads=int(num_heads),
            attn_dropout=float(attn_dropout),
        )
        self.act = nn.ReLU()
        self.drop = nn.Dropout(float(dropout))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.emb.weight, std=0.02)
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

    def encode(self, seeds: torch.Tensor, adj: CSRAdjacency, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        fan1, fan2 = self.fanouts
        seeds_np = seeds.detach().cpu().numpy().astype(np.int64, copy=False)

        nbr1 = adj.sample_neighbors(seeds_np, fan1, rng=rng)  # [B, fan1]
        nbr1_flat = nbr1.reshape(-1)
        nbr2 = adj.sample_neighbors(nbr1_flat, fan2, rng=rng)  # [B*fan1, fan2]

        nbr2_t = torch.from_numpy(nbr2).to(device=device, dtype=torch.long)
        nbr1_t = torch.from_numpy(nbr1_flat).to(device=device, dtype=torch.long)

        h2 = self.emb(nbr2_t)  # [B*fan1, fan2, D]
        h1_self = self.emb(nbr1_t)  # [B*fan1, D]
        h1 = self.act(self.gat1(h1_self, h2))  # [B*fan1, H]
        h1 = self.drop(h1)

        h1 = h1.view(seeds.shape[0], fan1, -1)  # [B, fan1, H]
        h0 = self.emb(seeds.to(device))  # [B, D]
        out = self.act(self.gat2(h0, h1))  # [B, H]
        out = self.drop(out)
        return out


@dataclass(frozen=True)
class LabelCSR:
    indptr: np.ndarray
    label_ids: np.ndarray
    label_w: np.ndarray
    src_ids: np.ndarray
    ts_s: np.ndarray


def _load_label_csr(exports_root: Path, dataset: str) -> LabelCSR:
    adj_dir = exports_root / dataset / "adj"
    indptr = np.load(adj_dir / "label_event_indptr.npy", mmap_mode="r")
    label_ids = np.load(adj_dir / "label_event_label_ids.npy", mmap_mode="r")
    label_w = np.load(adj_dir / "label_event_label_weights.npy", mmap_mode="r")
    src_ids = np.load(adj_dir / "label_event_src_ids.npy", mmap_mode="r")
    ts_s = np.load(adj_dir / "label_event_ts_s.npy", mmap_mode="r")
    return LabelCSR(indptr=indptr, label_ids=label_ids, label_w=label_w, src_ids=src_ids, ts_s=ts_s)


def _load_num_nodes(exports_root: Path, dataset: str) -> int:
    db_dir = exports_root / dataset / "db"
    return int(pq.ParquetFile(db_dir / "nodes.parquet").metadata.num_rows)


def _load_num_classes_from_metadata(exports_root: Path, dataset: str) -> Optional[int]:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    extra = meta.get("extra", {})
    v = extra.get("num_classes")
    return int(v) if v is not None else None


def _sample_split_ids(ts_s: np.ndarray, *, start_excl: Optional[int], end_incl: Optional[int]) -> np.ndarray:
    mask = np.ones_like(ts_s, dtype=bool)
    if start_excl is not None:
        mask &= ts_s > int(start_excl)
    if end_incl is not None:
        mask &= ts_s <= int(end_incl)
    return np.flatnonzero(mask)


def _subsample_ids(ids: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    if max_count <= 0 or ids.size <= max_count:
        return ids
    return rng.choice(ids, size=max_count, replace=False)


def _label_candidates_for_event(csr: LabelCSR, event_id: int) -> tuple[np.ndarray, np.ndarray]:
    start = int(csr.indptr[event_id])
    end = int(csr.indptr[event_id + 1])
    return (
        np.asarray(csr.label_ids[start:end], dtype=np.int64),
        np.asarray(csr.label_w[start:end], dtype=np.float32),
    )


def _sample_positive(label_ids: np.ndarray, label_w: np.ndarray, rng: np.random.Generator) -> int:
    if label_ids.size == 0:
        return -1
    w = np.maximum(label_w, 0.0)
    s = float(w.sum())
    if s <= 0:
        return int(rng.choice(label_ids))
    p = w.astype(np.float64, copy=False)
    p /= float(p.sum())
    # Avoid rare numpy strictness on p.sum() != 1 due to floating-point rounding.
    if p.size:
        tail = float(p[:-1].sum()) if p.size > 1 else 0.0
        p[-1] = max(0.0, 1.0 - tail)
        # Renormalize if we clipped.
        ps = float(p.sum())
        if ps > 0 and abs(ps - 1.0) > 1e-12:
            p /= ps
    return int(rng.choice(label_ids, p=p))


def _sample_negatives(
    *,
    rng: np.random.Generator,
    num_neg: int,
    label_universe: int,
    positives: np.ndarray,
) -> np.ndarray:
    if num_neg <= 0:
        return np.zeros((0,), dtype=np.int64)
    if positives.size == 0:
        return rng.integers(0, label_universe, size=num_neg, dtype=np.int64)
    pos_set = set(int(x) for x in positives.tolist())
    out: list[int] = []
    while len(out) < num_neg:
        cand = int(rng.integers(0, label_universe))
        if cand not in pos_set:
            out.append(cand)
    return np.asarray(out, dtype=np.int64)

def _rss_gb() -> Optional[float]:
    # Lightweight RSS read without extra deps.
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Example: "VmRSS:\t  123456 kB"
                    kb = int(line.split()[1])
                    return float(kb) / 1024.0 / 1024.0
    except Exception:
        return None
    return None


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    adj: CSRAdjacency,
    csr: LabelCSR,
    event_ids: np.ndarray,
    *,
    label_universe: int,
    num_neg: int,
    ndcg_k: int,
    device: torch.device,
    seed: int,
    max_events: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if event_ids.size == 0:
        return float("nan"), float("nan")
    if max_events > 0 and event_ids.size > max_events:
        event_ids = rng.choice(event_ids, size=max_events, replace=False)

    mrrs: list[float] = []
    ndcgs: list[float] = []
    for ev in event_ids.tolist():
        src_id = int(csr.src_ids[ev])
        pos_ids, pos_w = _label_candidates_for_event(csr, int(ev))
        if pos_ids.size == 0:
            continue
        # Candidate set = positives + sampled negatives.
        neg_ids = _sample_negatives(rng=rng, num_neg=num_neg, label_universe=label_universe, positives=pos_ids)
        cand = np.concatenate([pos_ids, neg_ids], axis=0)
        rel = np.concatenate([pos_w, np.zeros_like(neg_ids, dtype=np.float32)], axis=0)

        src = torch.tensor([src_id], device=device, dtype=torch.long)
        cand_t = torch.from_numpy(cand).to(device=device, dtype=torch.long)
        z_src = model.encode(src, adj, rng=rng, device=device)  # [1,H]
        z_c = model.encode(cand_t, adj, rng=rng, device=device)  # [C,H]
        score = (z_c * z_src.expand_as(z_c)).sum(dim=1).detach().cpu().numpy()

        order = np.argsort(-score, kind="mergesort")
        ranked_rel = rel[order]
        ranked_is_pos = order < pos_ids.size  # True for positives in the concatenated list

        # MRR: reciprocal rank of best positive.
        pos_ranks = np.flatnonzero(ranked_is_pos) + 1
        mrrs.append(float(1.0 / float(pos_ranks.min())))

        # NDCG@k with gain = relevance weight (simplified).
        k = int(ndcg_k)
        k = min(k, ranked_rel.size)
        denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
        dcg = float((ranked_rel[:k].astype(np.float64) / denom).sum())
        ideal = np.sort(pos_w.astype(np.float64))[::-1]
        ideal_k = min(k, ideal.size)
        idcg = float((ideal[:ideal_k] / np.log2(np.arange(2, ideal_k + 2, dtype=np.float64))).sum())
        ndcgs.append(float(dcg / idcg) if idcg > 0 else 0.0)

    if not mrrs:
        return float("nan"), float("nan")
    return float(np.mean(mrrs)), float(np.mean(ndcgs))


def _maybe_init_wandb(args, config: dict):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception:
        print("WARNING: wandb not available; disabling.")
        return None
    if args.wandb_mode == "offline":
        import os

        os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=config)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampled GNN baseline for tgbn-* node property prediction exports (MRR + NDCG@K).")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--adj", default="val", help="CSR adjacency cutoff: val | test | all | <unix_seconds>")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="sage", choices=["sage", "gat"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fanouts", default="10,5")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4, help="GAT heads (requires hidden_dim divisible by num_heads).")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Dropout on attention weights (GAT only).")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neg_train", type=int, default=1)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--ndcg_k", type=int, default=10)
    parser.add_argument("--max_train_events", type=int, default=200000)
    parser.add_argument("--max_val_events", type=int, default=20000)
    parser.add_argument("--max_test_events", type=int, default=20000)
    parser.add_argument("--max_eval_events", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--max_rss_gb", type=float, default=50.0, help="Abort early if process RSS exceeds this many GB (best-effort guardrail).")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--wandb_project", default="tgb2-nodeprop")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    num_nodes = _load_num_nodes(exports_root, args.dataset)
    num_classes = _load_num_classes_from_metadata(exports_root, args.dataset)
    label_universe = int(num_classes) if num_classes is not None else num_nodes

    # Load adjacency built externally via scripts/build_csr_adj.py
    adj_dir = exports_root / args.dataset / "adj"
    if args.adj == "val":
        upto = meta.val_timestamp_s
    elif args.adj == "test":
        upto = meta.test_timestamp_s
    elif args.adj == "all":
        upto = None
    else:
        upto = int(args.adj)

    suffix = f"upto_{upto}" if upto is not None else "all"
    suffix += "_undirected" if args.undirected else "_directed"
    adj = CSRAdjacency.load(adj_dir / f"csr_indptr_{suffix}.npy", adj_dir / f"csr_indices_{suffix}.npy")

    csr = _load_label_csr(exports_root, args.dataset)

    rng = np.random.default_rng(int(args.seed))
    train_ids = _sample_split_ids(csr.ts_s, start_excl=None, end_incl=meta.val_timestamp_s)
    val_ids = _sample_split_ids(csr.ts_s, start_excl=meta.val_timestamp_s, end_incl=meta.test_timestamp_s)
    test_ids = _sample_split_ids(csr.ts_s, start_excl=meta.test_timestamp_s, end_incl=None)

    train_ids = _subsample_ids(train_ids, int(args.max_train_events), rng=rng)
    val_ids = _subsample_ids(val_ids, int(args.max_val_events), rng=rng)
    test_ids = _subsample_ids(test_ids, int(args.max_test_events), rng=rng)

    fan1, fan2 = (int(x) for x in args.fanouts.split(","))
    device = torch.device(args.device)

    if args.model == "sage":
        model = SampledGraphSAGE(
            num_nodes=int(num_nodes),
            emb_dim=int(args.emb_dim),
            hidden_dim=int(args.hidden_dim),
            fanouts=(fan1, fan2),
            dropout=float(args.dropout),
        ).to(device)
    else:
        model = SampledGAT(
            num_nodes=int(num_nodes),
            emb_dim=int(args.emb_dim),
            hidden_dim=int(args.hidden_dim),
            fanouts=(fan1, fan2),
            dropout=float(args.dropout),
            num_heads=int(args.num_heads),
            attn_dropout=float(args.attn_dropout),
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    wb = _maybe_init_wandb(
        args,
        config={
            "dataset": args.dataset,
            "adj": args.adj,
            "undirected": bool(args.undirected),
            "model": args.model,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "fanouts": [fan1, fan2],
            "num_heads": int(args.num_heads) if args.model == "gat" else None,
            "attn_dropout": float(args.attn_dropout) if args.model == "gat" else None,
            "num_neg_train": int(args.num_neg_train),
            "num_neg_eval": int(args.num_neg_eval),
            "ndcg_k": int(args.ndcg_k),
            "max_train_events": int(args.max_train_events),
            "max_eval_events": int(args.max_eval_events),
            "label_universe": label_universe,
        },
    )

    steps_per_epoch = max(1, math.ceil(train_ids.size / int(args.batch_size)))
    print(
        f"Dataset={args.dataset} model={args.model} nodes={num_nodes} label_events(train/val/test)={train_ids.size}/{val_ids.size}/{test_ids.size} "
        f"label_universe={label_universe} adj_suffix={suffix}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        if args.max_rss_gb and args.max_rss_gb > 0:
            rss = _rss_gb()
            if rss is not None and rss > float(args.max_rss_gb):
                raise SystemExit(f"RSS guardrail tripped before epoch {epoch}: {rss:.2f} GB > {float(args.max_rss_gb):.2f} GB")
        model.train()
        perm = rng.permutation(train_ids.size)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            if args.max_rss_gb and args.max_rss_gb > 0 and (step % 50 == 0):
                rss = _rss_gb()
                if rss is not None and rss > float(args.max_rss_gb):
                    raise SystemExit(
                        f"RSS guardrail tripped at epoch {epoch} step {step}: {rss:.2f} GB > {float(args.max_rss_gb):.2f} GB"
                    )
            batch_ev = train_ids[perm[step * int(args.batch_size) : (step + 1) * int(args.batch_size)]]
            if batch_ev.size == 0:
                continue

            src_ids = csr.src_ids[batch_ev].astype(np.int64, copy=False)

            pos_labels: list[int] = []
            for ev in batch_ev.tolist():
                lab_ids, lab_w = _label_candidates_for_event(csr, int(ev))
                pos_labels.append(_sample_positive(lab_ids, lab_w, rng=rng))
            pos_labels_np = np.asarray(pos_labels, dtype=np.int64)

            # Sample negatives per row.
            neg_labels = np.empty((batch_ev.size, int(args.num_neg_train)), dtype=np.int64)
            for i, ev in enumerate(batch_ev.tolist()):
                lab_ids, _ = _label_candidates_for_event(csr, int(ev))
                neg_labels[i] = _sample_negatives(
                    rng=rng, num_neg=int(args.num_neg_train), label_universe=label_universe, positives=lab_ids
                )
            neg_flat = neg_labels.reshape(-1)

            src_t = torch.from_numpy(src_ids).to(device=device, dtype=torch.long)
            pos_t = torch.from_numpy(pos_labels_np).to(device=device, dtype=torch.long)
            neg_t = torch.from_numpy(neg_flat).to(device=device, dtype=torch.long)

            batch_rng = np.random.default_rng(int(args.seed) * 1_000_000 + epoch * 10_000 + step)
            z_src = model.encode(src_t, adj, rng=batch_rng, device=device)
            z_pos = model.encode(pos_t, adj, rng=batch_rng, device=device)
            pos_score = (z_src * z_pos).sum(dim=1, keepdim=True)
            z_neg = model.encode(neg_t, adj, rng=batch_rng, device=device).view(batch_ev.size, int(args.num_neg_train), -1)
            neg_score = torch.einsum("bd,bnd->bn", z_src, z_neg)
            loss = F.softplus(neg_score - pos_score).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / float(steps_per_epoch)
        model.eval()
        val_mrr, val_ndcg = _evaluate(
            model,
            adj,
            csr,
            val_ids,
            label_universe=label_universe,
            num_neg=int(args.num_neg_eval),
            ndcg_k=int(args.ndcg_k),
            device=device,
            seed=int(args.seed) + 13 + epoch,
            max_events=int(args.max_eval_events),
        )
        test_mrr, test_ndcg = _evaluate(
            model,
            adj,
            csr,
            test_ids,
            label_universe=label_universe,
            num_neg=int(args.num_neg_eval),
            ndcg_k=int(args.ndcg_k),
            device=device,
            seed=int(args.seed) + 17 + epoch,
            max_events=int(args.max_eval_events),
        )
        print(
            f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} val_ndcg@{int(args.ndcg_k)}={val_ndcg:.4f} "
            f"test_mrr={test_mrr:.4f} test_ndcg@{int(args.ndcg_k)}={test_ndcg:.4f}"
        )
        if wb is not None:
            wb.log(
                {
                    "epoch": epoch,
                    "loss": avg_loss,
                    "val_mrr": val_mrr,
                    f"val_ndcg@{int(args.ndcg_k)}": val_ndcg,
                    "test_mrr": test_mrr,
                    f"test_ndcg@{int(args.ndcg_k)}": test_ndcg,
                }
            )

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset": args.dataset,
            "exports_root": str(exports_root),
            "adj": args.adj,
            "undirected": bool(args.undirected),
            "meta": {"val_timestamp_s": meta.val_timestamp_s, "test_timestamp_s": meta.test_timestamp_s},
            "model": {
                "arch": args.model,
                "num_nodes": int(num_nodes),
                "emb_dim": int(args.emb_dim),
                "hidden_dim": int(args.hidden_dim),
                "fanouts": [fan1, fan2],
                "dropout": float(args.dropout),
                "num_heads": int(args.num_heads) if args.model == "gat" else None,
                "attn_dropout": float(args.attn_dropout) if args.model == "gat" else None,
            },
            "state_dict": model.state_dict(),
        }
        ckpt_path = save_dir / f"{args.model}_nodeprop_{args.dataset}.pt"
        torch.save(out, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
