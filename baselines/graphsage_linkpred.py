#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn


@dataclass(frozen=True)
class ExportMeta:
    dataset: str
    tgb_internal_name: str
    val_timestamp_s: int
    test_timestamp_s: int


def _load_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    return ExportMeta(
        dataset=meta["dataset"],
        tgb_internal_name=meta["tgb_internal_name"],
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
    )


def _detect_bipartite(db_dir: Path) -> tuple[bool, int, int]:
    src_path = db_dir / "src_nodes.parquet"
    dst_path = db_dir / "dst_nodes.parquet"
    if src_path.exists() and dst_path.exists():
        n_src = pq.ParquetFile(src_path).metadata.num_rows
        n_dst = pq.ParquetFile(dst_path).metadata.num_rows
        return True, int(n_src), int(n_dst)
    n = pq.ParquetFile(db_dir / "nodes.parquet").metadata.num_rows
    return False, int(n), 0


def _iter_events(events_path: Path, batch_size: int):
    pf = pq.ParquetFile(events_path)
    yield from pf.iter_batches(batch_size=batch_size, columns=["src_id", "dst_id", "event_ts"])


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _load_edges_by_time(
    exports_root: Path,
    dataset: str,
    *,
    start_s_exclusive: Optional[int],
    end_s_inclusive: Optional[int],
    parquet_batch_size: int,
    max_edges: Optional[int],
    seed: int,
) -> tuple[np.ndarray, bool, int, int]:
    db_dir = exports_root / dataset / "db"
    events_path = db_dir / "events.parquet"
    is_bipartite, n_src, n_dst = _detect_bipartite(db_dir)

    total = pq.ParquetFile(events_path).metadata.num_rows
    p = 1.0
    if max_edges is not None and max_edges > 0:
        p = min(1.0, float(max_edges) / float(total))
    rng = np.random.default_rng(seed)

    edges: list[np.ndarray] = []
    for batch in _iter_events(events_path, parquet_batch_size):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))

        mask = np.ones_like(ts_s, dtype=bool)
        if start_s_exclusive is not None:
            mask &= ts_s > int(start_s_exclusive)
        if end_s_inclusive is not None:
            mask &= ts_s <= int(end_s_inclusive)
        if p < 1.0 and mask.any():
            keep = rng.random(mask.shape[0]) < p
            mask &= keep
        if not mask.any():
            continue
        edges.append(np.stack([src[mask], dst[mask]], axis=1))

    out = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), dtype=np.int64)
    return out, is_bipartite, n_src, n_dst


def _load_official_task_edges(
    *,
    relbench_cache_root: Path,
    dataset: str,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    task_dir = relbench_cache_root / f"rel-tgb-{dataset}" / "tasks" / "src-dst-mrr"
    df = pd.read_parquet(task_dir / f"{split}.parquet", columns=["src_id", "dst_id", "event_ts"])
    ts_s = (pd.to_datetime(df["event_ts"], utc=True).astype("int64").to_numpy(copy=False) // 1_000_000_000).astype(
        np.int64, copy=False
    )
    edges = df[["src_id", "dst_id"]].to_numpy(dtype=np.int64, copy=False)
    return edges, ts_s


def _load_official_negatives(
    *,
    relbench_cache_root: Path,
    dataset: str,
    split: str,
) -> dict:
    path = relbench_cache_root / f"rel-tgb-{dataset}" / "negatives" / f"{split}_ns.pkl"
    with path.open("rb") as f:
        return pickle.load(f)


def _subsample_edges_and_ts(edges: np.ndarray, ts_s: np.ndarray, *, max_edges: Optional[int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    if max_edges is None or int(max_edges) <= 0 or edges.shape[0] <= int(max_edges):
        return edges, ts_s
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(edges.shape[0], size=int(max_edges), replace=False)
    return edges[idx], ts_s[idx]


class CSRAdjacency:
    def __init__(self, indptr: np.ndarray, indices: np.ndarray):
        self.indptr = indptr
        self.indices = indices
        self.num_nodes = int(indptr.shape[0] - 1)

    @classmethod
    def load(cls, indptr_path: Path, indices_path: Path) -> "CSRAdjacency":
        indptr = np.load(indptr_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
        return cls(indptr=indptr, indices=indices)

    def sample_neighbors(self, nodes: np.ndarray, fanout: int, rng: np.random.Generator) -> np.ndarray:
        nodes = nodes.astype(np.int64, copy=False)
        start = self.indptr[nodes].astype(np.int64, copy=False)
        end = self.indptr[nodes + 1].astype(np.int64, copy=False)
        deg = (end - start).astype(np.int64, copy=False)

        # Sample with replacement; for deg==0, return self-loops.
        out = np.empty((nodes.shape[0], fanout), dtype=np.int64)
        zero_deg = deg == 0
        if zero_deg.any():
            out[zero_deg, :] = nodes[zero_deg, None]

        mask = ~zero_deg
        if mask.any():
            deg_m = deg[mask]
            start_m = start[mask]
            rand = rng.random((start_m.shape[0], fanout), dtype=np.float64)
            offs = (rand * deg_m[:, None]).astype(np.int64, copy=False)
            idx = start_m[:, None] + offs
            out[mask, :] = self.indices[idx]

        return out


class SampledGraphSAGE(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, hidden_dim: int, fanouts: tuple[int, int], dropout: float):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        self.fanouts = fanouts
        self.lin1_self = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.lin1_neigh = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.lin2_self = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.lin2_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        for m in [self.lin1_self, self.lin1_neigh, self.lin2_self, self.lin2_neigh]:
            nn.init.xavier_uniform_(m.weight)

    def encode(self, seeds: torch.Tensor, adj: CSRAdjacency, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        # 2-layer GraphSAGE with neighbor sampling.
        fan1, fan2 = self.fanouts
        seeds_np = seeds.detach().cpu().numpy().astype(np.int64, copy=False)

        nbr1 = adj.sample_neighbors(seeds_np, fan1, rng=rng)  # [B, fan1]
        nbr1_flat = nbr1.reshape(-1)
        nbr2 = adj.sample_neighbors(nbr1_flat, fan2, rng=rng)  # [B*fan1, fan2]

        nbr2_t = torch.from_numpy(nbr2).to(device=device, dtype=torch.long)
        nbr1_t = torch.from_numpy(nbr1_flat).to(device=device, dtype=torch.long)

        h2 = self.emb(nbr2_t)  # [B*fan1, fan2, D]
        h2_mean = h2.mean(dim=1)  # [B*fan1, D]
        h1_self = self.emb(nbr1_t)  # [B*fan1, D]
        h1 = self.act(self.lin1_self(h1_self) + self.lin1_neigh(h2_mean))  # [B*fan1, H]
        h1 = self.drop(h1)
        h1 = h1.view(seeds.shape[0], fan1, -1)  # [B, fan1, H]
        h1_mean = h1.mean(dim=1)  # [B, H]

        h0 = self.emb(seeds.to(device))  # [B, D]
        out = self.act(self.lin2_self(h0) + self.lin2_neigh(h1_mean))  # [B, H]
        out = self.drop(out)
        return out


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

        q = self.lin_q(h_self).view(n, self.num_heads, self.head_dim)  # [N,H,K]
        k = self.lin_k(h_neigh).view(n, fanout, self.num_heads, self.head_dim)  # [N,F,H,K]
        v = self.lin_v(h_neigh).view(n, fanout, self.num_heads, self.head_dim)

        scores = (k * q.unsqueeze(1)).sum(dim=-1) * self.scale  # [N,F,H]
        attn = torch.softmax(scores, dim=1)  # [N,F,H]
        attn = self.drop_attn(attn)

        agg = (attn.unsqueeze(-1) * v).sum(dim=1)  # [N,H,K]
        agg = agg.reshape(n, self.out_dim)
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


def _maybe_init_wandb(args, config: dict):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    try:
        import wandb
    except Exception:
        print("WARNING: wandb not available; disabling.")
        return None
    if args.wandb_mode == "offline":
        # Avoid requiring credentials in automated runs.
        import os

        os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=config)


@torch.no_grad()
def _evaluate_mrr(
    model: nn.Module,
    adj: CSRAdjacency,
    edges: np.ndarray,
    *,
    num_neg: int,
    num_dst: int,
    dst_offset: int,
    device: torch.device,
    max_edges: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    if edges.shape[0] == 0:
        return float("nan")
    if max_edges > 0 and edges.shape[0] > max_edges:
        idx = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[idx]

    src = torch.from_numpy(edges[:, 0]).to(device=device, dtype=torch.long)
    pos_dst_local = torch.from_numpy(edges[:, 1]).to(device=device, dtype=torch.long)
    pos_dst = pos_dst_local + dst_offset if dst_offset else pos_dst_local

    z_src = model.encode(src, adj, rng=rng, device=device)
    z_pos = model.encode(pos_dst, adj, rng=rng, device=device)
    pos_score = (z_src * z_pos).sum(dim=1, keepdim=True)

    neg_dst_local = torch.randint(0, num_dst, (src.shape[0], num_neg), device=device, dtype=torch.long)
    neg_dst = neg_dst_local + dst_offset if dst_offset else neg_dst_local
    neg_flat = neg_dst.reshape(-1)
    z_neg = model.encode(neg_flat, adj, rng=rng, device=device).view(src.shape[0], num_neg, -1)
    neg_score = torch.einsum("bd,bnd->bn", z_src, z_neg)

    rank = 1 + (neg_score >= pos_score).sum(dim=1)
    return float((1.0 / rank.float()).mean().item())


@torch.no_grad()
def _evaluate_tgb_official(
    model: nn.Module,
    adj: CSRAdjacency,
    edges: np.ndarray,
    ts_s: np.ndarray,
    neg_dict: dict,
    *,
    dst_offset: int,
    device: torch.device,
    max_edges: int,
    seed: int,
    k_value: int = 10,
) -> tuple[float, float]:
    """Official TGB one-vs-many MRR/Hits@k using provided negatives."""
    rng = np.random.default_rng(seed)
    if edges.shape[0] == 0:
        return float("nan"), float("nan")
    if max_edges > 0 and edges.shape[0] > max_edges:
        idx = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[idx]
        ts_s = ts_s[idx]

    src = edges[:, 0].astype(np.int64, copy=False)
    dst_local = edges[:, 1].astype(np.int64, copy=False)
    dst_global = dst_local + int(dst_offset) if dst_offset else dst_local

    src_t = torch.from_numpy(src).to(device=device, dtype=torch.long)
    pos_t = torch.from_numpy(dst_global).to(device=device, dtype=torch.long)

    # Some datasets can have per-example negative lists with slightly varying
    # lengths, so we avoid assuming a fixed K.
    neg_lists: list[np.ndarray] = []
    lens = np.empty((edges.shape[0],), dtype=np.int64)
    for i in range(edges.shape[0]):
        key = (int(src[i]), int(dst_global[i]), int(ts_s[i]))
        neg_g = np.asarray(neg_dict[key], dtype=np.int64)
        neg_g = neg_g.astype(np.int64, copy=False)
        neg_lists.append(neg_g)
        lens[i] = int(neg_g.shape[0])
    neg_ptr = np.zeros((edges.shape[0] + 1,), dtype=np.int64)
    np.cumsum(lens, out=neg_ptr[1:])
    neg_flat = np.concatenate(neg_lists, axis=0) if neg_lists else np.zeros((0,), dtype=np.int64)
    neg_row = np.repeat(np.arange(edges.shape[0], dtype=np.int64), lens.astype(np.int64, copy=False))
    neg_flat_t = torch.from_numpy(neg_flat).to(device=device, dtype=torch.long)
    neg_row_t = torch.from_numpy(neg_row).to(device=device, dtype=torch.long)

    z_src = model.encode(src_t, adj, rng=rng, device=device)
    z_pos = model.encode(pos_t, adj, rng=rng, device=device)
    pos_score = (z_src * z_pos).sum(dim=1)  # [B]

    z_neg_flat = model.encode(neg_flat_t, adj, rng=rng, device=device)  # [sumK,H]
    neg_score_flat = (z_src[neg_row_t] * z_neg_flat).sum(dim=1)  # [sumK]
    optimistic = torch.zeros((src_t.shape[0],), device=device, dtype=torch.float32)
    pessimistic = torch.zeros((src_t.shape[0],), device=device, dtype=torch.float32)
    for i in range(src_t.shape[0]):
        start = int(neg_ptr[i])
        end = int(neg_ptr[i + 1])
        if end <= start:
            continue
        s = neg_score_flat[start:end]
        p = pos_score[i]
        optimistic[i] = (s > p).sum()
        pessimistic[i] = (s >= p).sum()

    rank = 0.5 * (optimistic + pessimistic) + 1.0

    mrr = float((1.0 / rank).mean().item())
    hits = float((rank <= float(int(k_value))).float().mean().item())
    return mrr, hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampled GraphSAGE baseline for RelBench-exported TGB link datasets.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--adj", default="val", help="Which CSR adjacency to use: val | test | all | <unix_seconds>")
    parser.add_argument("--undirected", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", default="sage", choices=["sage", "gat"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fanouts", default="15,10", help="fanout1,fanout2")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_heads", type=int, default=4, help="GAT heads (requires hidden_dim divisible by num_heads).")
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Dropout on attention weights (GAT only).")
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--num_neg_train", type=int, default=1)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--max_eval_edges", type=int, default=20000)
    parser.add_argument(
        "--eval_mode",
        default="sampled",
        choices=["sampled", "tgb"],
        help="Evaluation protocol: sampled negatives vs official TGB one-vs-many (requires relbench_cache_root).",
    )
    parser.add_argument(
        "--relbench_cache_root",
        default="/home/pc0618/tmp_relbench_cache_official",
        help="Root containing prepared rel-tgb-* cache dirs (for eval_mode=tgb).",
    )
    parser.add_argument("--parquet_batch_size", type=int, default=500000)
    parser.add_argument("--max_train_edges", type=int, default=None)
    parser.add_argument("--max_val_edges", type=int, default=None)
    parser.add_argument("--max_test_edges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", default=None, help="Optional directory to save model checkpoint + config.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tgb-relbench-graphsage")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    db_dir = exports_root / args.dataset / "db"
    is_bipartite, n_src, n_dst = _detect_bipartite(db_dir)
    num_nodes = n_src + n_dst if is_bipartite else n_src
    dst_offset = n_src if is_bipartite else 0
    num_dst = n_dst if is_bipartite else num_nodes

    if args.adj == "val":
        upto_s = meta.val_timestamp_s
    elif args.adj == "test":
        upto_s = meta.test_timestamp_s
    elif args.adj == "all":
        upto_s = None
    else:
        upto_s = int(args.adj)

    suffix = f"upto_{upto_s}" if upto_s is not None else "all"
    suffix += "_undirected" if args.undirected else "_directed"
    adj_dir = exports_root / args.dataset / "adj"
    indptr_path = adj_dir / f"csr_indptr_{suffix}.npy"
    indices_path = adj_dir / f"csr_indices_{suffix}.npy"
    adj: Optional[CSRAdjacency] = None

    fan1, fan2 = (int(x) for x in args.fanouts.split(","))
    device = torch.device(args.device)
    if args.model == "sage":
        model = SampledGraphSAGE(
            num_nodes=num_nodes,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            fanouts=(fan1, fan2),
            dropout=args.dropout,
        ).to(device)
    else:
        model = SampledGAT(
            num_nodes=num_nodes,
            emb_dim=args.emb_dim,
            hidden_dim=args.hidden_dim,
            fanouts=(fan1, fan2),
            dropout=args.dropout,
            num_heads=int(args.num_heads),
            attn_dropout=float(args.attn_dropout),
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    run = _maybe_init_wandb(
        args,
        config={
            "dataset": args.dataset,
            "bipartite": is_bipartite,
            "num_nodes": num_nodes,
            "fanouts": [fan1, fan2],
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "wd": args.weight_decay,
            "batch_size": args.batch_size,
            "neg_train": args.num_neg_train,
            "neg_eval": args.num_neg_eval,
            "eval_mode": args.eval_mode,
            "adj": args.adj,
            "undirected": args.undirected,
            "model": args.model,
            "num_heads": int(args.num_heads) if args.model == "gat" else None,
            "attn_dropout": float(args.attn_dropout) if args.model == "gat" else None,
        },
    )

    if args.eval_mode == "tgb":
        relbench_cache_root = Path(args.relbench_cache_root)
        train_edges, train_ts_s = _load_official_task_edges(
            relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="train"
        )
        val_edges, val_ts_s = _load_official_task_edges(
            relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="val"
        )
        test_edges, test_ts_s = _load_official_task_edges(
            relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="test"
        )
        train_edges, train_ts_s = _subsample_edges_and_ts(
            train_edges, train_ts_s, max_edges=args.max_train_edges, seed=args.seed + 10
        )
        val_edges, val_ts_s = _subsample_edges_and_ts(val_edges, val_ts_s, max_edges=args.max_val_edges, seed=args.seed + 20)
        test_edges, test_ts_s = _subsample_edges_and_ts(
            test_edges, test_ts_s, max_edges=args.max_test_edges, seed=args.seed + 30
        )
        val_neg = _load_official_negatives(relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="val")
        test_neg = _load_official_negatives(
            relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="test"
        )
    else:
        train_edges, *_ = _load_edges_by_time(
            exports_root,
            args.dataset,
            start_s_exclusive=None,
            end_s_inclusive=meta.val_timestamp_s,
            parquet_batch_size=args.parquet_batch_size,
            max_edges=args.max_train_edges,
            seed=args.seed + 10,
        )
        val_edges, *_ = _load_edges_by_time(
            exports_root,
            args.dataset,
            start_s_exclusive=meta.val_timestamp_s,
            end_s_inclusive=meta.test_timestamp_s,
            parquet_batch_size=args.parquet_batch_size,
            max_edges=args.max_val_edges,
            seed=args.seed + 20,
        )
        test_edges, *_ = _load_edges_by_time(
            exports_root,
            args.dataset,
            start_s_exclusive=meta.test_timestamp_s,
            end_s_inclusive=None,
            parquet_batch_size=args.parquet_batch_size,
            max_edges=args.max_test_edges,
            seed=args.seed + 30,
        )
        train_ts_s = val_ts_s = test_ts_s = None
        val_neg = test_neg = None

    if adj is None:
        if indptr_path.exists() and indices_path.exists():
            adj = CSRAdjacency.load(indptr_path, indices_path)
        else:
            # Build a small in-memory adjacency from the (possibly capped) training edges.
            # This keeps the script runnable without precomputing CSR files.
            edges_for_adj = train_edges.astype(np.int64, copy=False)
            src_ids = edges_for_adj[:, 0]
            dst_ids = edges_for_adj[:, 1] + int(dst_offset) if dst_offset else edges_for_adj[:, 1]
            if args.undirected:
                src_all = np.concatenate([src_ids, dst_ids], axis=0)
                dst_all = np.concatenate([dst_ids, src_ids], axis=0)
            else:
                src_all = src_ids
                dst_all = dst_ids

            deg = np.bincount(src_all, minlength=int(num_nodes)).astype(np.int64, copy=False)
            indptr = np.empty((int(num_nodes) + 1,), dtype=np.int64)
            indptr[0] = 0
            np.cumsum(deg, out=indptr[1:])
            indices = np.empty((int(indptr[-1]),), dtype=np.int64)
            cur = indptr[:-1].copy()
            for s, d in zip(src_all.tolist(), dst_all.tolist()):
                j = cur[s]
                indices[j] = int(d)
                cur[s] += 1
            adj = CSRAdjacency(indptr=indptr, indices=indices)

    if train_edges.shape[0] == 0:
        raise RuntimeError("No training edges found. Check cutoffs.")
    perm = np.random.permutation(train_edges.shape[0])
    train_edges = train_edges[perm]
    steps_per_epoch = max(1, math.ceil(train_edges.shape[0] / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng = np.random.default_rng(args.seed + 1000 + epoch)
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = train_edges[step * args.batch_size : (step + 1) * args.batch_size]
            src = torch.from_numpy(batch[:, 0]).to(device=device, dtype=torch.long)
            dst_local = torch.from_numpy(batch[:, 1]).to(device=device, dtype=torch.long)
            dst = dst_local + dst_offset if dst_offset else dst_local

            z_src = model.encode(src, adj, rng=rng, device=device)
            z_dst = model.encode(dst, adj, rng=rng, device=device)
            pos_logit = (z_src * z_dst).sum(dim=1)

            neg_dst_local = torch.randint(0, num_dst, (src.shape[0] * args.num_neg_train,), device=device, dtype=torch.long)
            neg_dst = neg_dst_local + dst_offset if dst_offset else neg_dst_local
            src_rep = src.repeat_interleave(args.num_neg_train)
            z_src_n = model.encode(src_rep, adj, rng=rng, device=device)
            z_neg = model.encode(neg_dst, adj, rng=rng, device=device)
            neg_logit = (z_src_n * z_neg).sum(dim=1)

            loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(neg_logit, torch.zeros_like(neg_logit))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())

        model.eval()
        with torch.no_grad():
            if args.eval_mode == "tgb":
                val_mrr, val_hits = _evaluate_tgb_official(
                    model,
                    adj,
                    val_edges,
                    val_ts_s,
                    val_neg,
                    dst_offset=dst_offset,
                    device=device,
                    max_edges=args.max_eval_edges,
                    seed=args.seed + 2000 + epoch,
                    k_value=10,
                )
                test_mrr, test_hits = _evaluate_tgb_official(
                    model,
                    adj,
                    test_edges,
                    test_ts_s,
                    test_neg,
                    dst_offset=dst_offset,
                    device=device,
                    max_edges=args.max_eval_edges,
                    seed=args.seed + 3000 + epoch,
                    k_value=10,
                )
            else:
                val_mrr = _evaluate_mrr(
                    model,
                    adj,
                    val_edges,
                    num_neg=args.num_neg_eval,
                    num_dst=num_dst,
                    dst_offset=dst_offset,
                    device=device,
                    max_edges=args.max_eval_edges,
                    seed=args.seed + 2000 + epoch,
                )
                test_mrr = _evaluate_mrr(
                    model,
                    adj,
                    test_edges,
                    num_neg=args.num_neg_eval,
                    num_dst=num_dst,
                    dst_offset=dst_offset,
                    device=device,
                    max_edges=args.max_eval_edges,
                    seed=args.seed + 3000 + epoch,
                )

        avg_loss = epoch_loss / steps_per_epoch
        if args.eval_mode == "tgb":
            print(
                f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} val_hits@10={val_hits:.4f} "
                f"test_mrr={test_mrr:.4f} test_hits@10={test_hits:.4f}"
            )
            if run is not None:
                run.log(
                    {
                        "epoch": epoch,
                        "loss": avg_loss,
                        "val_mrr": val_mrr,
                        "val_hits@10": val_hits,
                        "test_mrr": test_mrr,
                        "test_hits@10": test_hits,
                    }
                )
        else:
            print(
                f"epoch={epoch} loss={avg_loss:.4f} val_mrr@{args.num_neg_eval}={val_mrr:.4f} "
                f"test_mrr@{args.num_neg_eval}={test_mrr:.4f}"
            )
            if run is not None:
                run.log({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr, "test_mrr": test_mrr})

    if run is not None:
        run.finish()

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "dataset": args.dataset,
            "exports_root": str(args.exports_root),
            "adj": args.adj,
            "undirected": bool(args.undirected),
            "meta": {
                "val_timestamp_s": meta.val_timestamp_s,
                "test_timestamp_s": meta.test_timestamp_s,
            },
            "model": {
                "num_nodes": num_nodes,
                "emb_dim": args.emb_dim,
                "hidden_dim": args.hidden_dim,
                "fanouts": [fan1, fan2],
                "dropout": args.dropout,
            },
            "state_dict": model.state_dict(),
        }
        out_path = save_dir / f"graphsage_{args.dataset}.pt"
        torch.save(ckpt, out_path)
        (save_dir / f"graphsage_{args.dataset}.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "exports_root": str(args.exports_root),
                    "adj": args.adj,
                    "undirected": bool(args.undirected),
                    "epochs": args.epochs,
                    "emb_dim": args.emb_dim,
                    "hidden_dim": args.hidden_dim,
                    "fanouts": [fan1, fan2],
                    "dropout": args.dropout,
                    "batch_size": args.batch_size,
                    "num_neg_train": args.num_neg_train,
                    "num_neg_eval": args.num_neg_eval,
                    "max_train_edges": args.max_train_edges,
                    "max_val_edges": args.max_val_edges,
                    "max_test_edges": args.max_test_edges,
                    "max_eval_edges": args.max_eval_edges,
                    "seed": args.seed,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
