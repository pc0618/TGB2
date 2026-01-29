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
from torch.nn import functional as F


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

        out = np.empty((nodes.shape[0], fanout), dtype=np.int64)
        zero_deg = deg == 0
        if zero_deg.any():
            out[zero_deg, :] = -1

        mask = ~zero_deg
        if mask.any():
            deg_m = deg[mask]
            start_m = start[mask]
            rand = rng.random((start_m.shape[0], fanout), dtype=np.float64)
            offs = (rand * deg_m[:, None]).astype(np.int64, copy=False)
            idx = start_m[:, None] + offs
            out[mask, :] = self.indices[idx]

        return out


class RelEventSAGE(nn.Module):
    def __init__(
        self,
        *,
        is_bipartite: bool,
        num_src_nodes: int,
        num_dst_nodes: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.is_bipartite = is_bipartite
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        if is_bipartite:
            self.src_emb = nn.Embedding(num_src_nodes, emb_dim)
            self.dst_emb = nn.Embedding(num_dst_nodes, emb_dim)
        else:
            self.node_emb = nn.Embedding(num_src_nodes, emb_dim)

        # Encode (time, weight) into hidden_dim.
        self.event_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_src_lin = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.event_dst_lin = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.src_lin_self = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.src_lin_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.dst_lin_self = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.dst_lin_neigh = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        if self.is_bipartite:
            nn.init.normal_(self.src_emb.weight, std=0.02)
            nn.init.normal_(self.dst_emb.weight, std=0.02)
        else:
            nn.init.normal_(self.node_emb.weight, std=0.02)

    def _node_embed_src(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.src_emb(node_ids) if self.is_bipartite else self.node_emb(node_ids)

    def _node_embed_dst(self, node_ids: torch.Tensor) -> torch.Tensor:
        return self.dst_emb(node_ids) if self.is_bipartite else self.node_emb(node_ids)

    def _event_repr(
        self,
        event_ids_flat: torch.Tensor,
        *,
        event_src: np.ndarray,
        event_dst: np.ndarray,
        event_ts_s: np.ndarray,
        event_w: np.ndarray,
        ts_min_s: int,
        ts_range_s: int,
        device: torch.device,
    ) -> torch.Tensor:
        # event_ids_flat may contain -1 for padding.
        valid = event_ids_flat >= 0
        out = torch.zeros((event_ids_flat.numel(), self.hidden_dim), device=device)
        if not valid.any():
            return out

        ev = event_ids_flat[valid].detach().cpu().numpy().astype(np.int64, copy=False)
        src = torch.from_numpy(event_src[ev]).to(device=device, dtype=torch.long)
        dst = torch.from_numpy(event_dst[ev]).to(device=device, dtype=torch.long)

        src_e = self._node_embed_src(src)
        dst_e = self._node_embed_dst(dst)

        ts = torch.from_numpy(event_ts_s[ev]).to(device=device, dtype=torch.float32)
        w = torch.from_numpy(event_w[ev]).to(device=device, dtype=torch.float32)
        # Some datasets have heavy-tailed weights (e.g., up to thousands); compress
        # to avoid overflow in small MLPs.
        w = torch.log1p(w)
        ts_norm = (ts - float(ts_min_s)) / float(ts_range_s) if ts_range_s > 0 else ts * 0.0
        feat = torch.stack([ts_norm, w], dim=1)

        ev_h = self.event_mlp(feat) + self.event_src_lin(src_e) + self.event_dst_lin(dst_e)
        ev_h = self.act(ev_h)
        ev_h = self.drop(ev_h)

        out[valid] = ev_h
        return out

    def encode_src(
        self,
        seeds: torch.Tensor,
        *,
        adj_src_to_events: CSRAdjacency,
        event_src: np.ndarray,
        event_dst: np.ndarray,
        event_ts_s: np.ndarray,
        event_w: np.ndarray,
        ts_min_s: int,
        ts_range_s: int,
        fanout: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        seeds_np = seeds.detach().cpu().numpy().astype(np.int64, copy=False)
        nbr_ev = adj_src_to_events.sample_neighbors(seeds_np, fanout, rng=rng)  # [B, F]
        nbr_ev_t = torch.from_numpy(nbr_ev.reshape(-1)).to(device=device, dtype=torch.long)
        ev_h = self._event_repr(
            nbr_ev_t,
            event_src=event_src,
            event_dst=event_dst,
            event_ts_s=event_ts_s,
            event_w=event_w,
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            device=device,
        ).view(seeds.shape[0], fanout, -1)
        neigh = ev_h.mean(dim=1)

        self_h = self._node_embed_src(seeds.to(device))
        out = self.act(self.src_lin_self(self_h) + self.src_lin_neigh(neigh))
        out = self.drop(out)
        return out

    def encode_dst(
        self,
        seeds: torch.Tensor,
        *,
        adj_dst_to_events: CSRAdjacency,
        event_src: np.ndarray,
        event_dst: np.ndarray,
        event_ts_s: np.ndarray,
        event_w: np.ndarray,
        ts_min_s: int,
        ts_range_s: int,
        fanout: int,
        rng: np.random.Generator,
        device: torch.device,
    ) -> torch.Tensor:
        seeds_np = seeds.detach().cpu().numpy().astype(np.int64, copy=False)
        nbr_ev = adj_dst_to_events.sample_neighbors(seeds_np, fanout, rng=rng)  # [B, F]
        nbr_ev_t = torch.from_numpy(nbr_ev.reshape(-1)).to(device=device, dtype=torch.long)
        ev_h = self._event_repr(
            nbr_ev_t,
            event_src=event_src,
            event_dst=event_dst,
            event_ts_s=event_ts_s,
            event_w=event_w,
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            device=device,
        ).view(seeds.shape[0], fanout, -1)
        neigh = ev_h.mean(dim=1)

        self_h = self._node_embed_dst(seeds.to(device))
        out = self.act(self.dst_lin_self(self_h) + self.dst_lin_neigh(neigh))
        out = self.drop(out)
        return out


def _suffix_from_adj_arg(meta: ExportMeta, adj: str) -> str:
    if adj == "val":
        return f"upto_{meta.val_timestamp_s}"
    if adj == "test":
        return f"upto_{meta.test_timestamp_s}"
    if adj == "all":
        return "all"
    return f"upto_{int(adj)}"


def _load_rel_event_arrays(exports_root: Path, dataset: str, *, suffix: str):
    adj_dir = exports_root / dataset / "adj"
    src_indptr = adj_dir / f"csr_events_by_src_id_indptr_{suffix}.npy"
    src_indices = adj_dir / f"csr_events_by_src_id_indices_{suffix}.npy"
    dst_indptr = adj_dir / f"csr_events_by_dst_id_indptr_{suffix}.npy"
    dst_indices = adj_dir / f"csr_events_by_dst_id_indices_{suffix}.npy"

    event_src = adj_dir / f"events_src_id_{suffix}.npy"
    event_dst = adj_dir / f"events_dst_id_{suffix}.npy"
    event_ts = adj_dir / f"events_ts_s_{suffix}.npy"
    event_w = adj_dir / f"events_weight_{suffix}.npy"

    missing = [p for p in [src_indptr, src_indices, dst_indptr, dst_indices, event_src, event_dst, event_ts, event_w] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing relational event CSR arrays. Build them via "
            f"`scripts/build_rel_event_csr.py --dataset {dataset} --upto {suffix.replace('upto_', '')}`. "
            f"Missing: {', '.join(str(p) for p in missing)}"
        )

    return (
        CSRAdjacency.load(src_indptr, src_indices),
        CSRAdjacency.load(dst_indptr, dst_indices),
        np.load(event_src, mmap_mode="r"),
        np.load(event_dst, mmap_mode="r"),
        np.load(event_ts, mmap_mode="r"),
        np.load(event_w, mmap_mode="r"),
    )


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


@torch.no_grad()
def _evaluate_mrr(
    model: RelEventSAGE,
    adj_src_to_events: CSRAdjacency,
    adj_dst_to_events: CSRAdjacency,
    event_src: np.ndarray,
    event_dst: np.ndarray,
    event_ts_s: np.ndarray,
    event_w: np.ndarray,
    *,
    edges: np.ndarray,
    ts_min_s: int,
    ts_range_s: int,
    num_neg: int,
    num_dst: int,
    device: torch.device,
    fanout: int,
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
    pos_dst = torch.from_numpy(edges[:, 1]).to(device=device, dtype=torch.long)

    z_src = model.encode_src(
        src,
        adj_src_to_events=adj_src_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    )
    z_pos = model.encode_dst(
        pos_dst,
        adj_dst_to_events=adj_dst_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    )
    pos_score = (z_src * z_pos).sum(dim=1, keepdim=True)

    neg_dst = torch.randint(0, num_dst, (src.shape[0], num_neg), device=device, dtype=torch.long)
    neg_flat = neg_dst.reshape(-1)
    z_neg = model.encode_dst(
        neg_flat,
        adj_dst_to_events=adj_dst_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    ).view(src.shape[0], num_neg, -1)
    neg_score = torch.einsum("bd,bnd->bn", z_src, z_neg)

    rank = 1 + (neg_score >= pos_score).sum(dim=1)
    return float((1.0 / rank.float()).mean().item())


@torch.no_grad()
def _evaluate_tgb_official(
    model: RelEventSAGE,
    adj_src_to_events: CSRAdjacency,
    adj_dst_to_events: CSRAdjacency,
    event_src: np.ndarray,
    event_dst: np.ndarray,
    event_ts_s: np.ndarray,
    event_w: np.ndarray,
    *,
    edges: np.ndarray,
    ts_s: np.ndarray,
    neg_dict: dict,
    ts_min_s: int,
    ts_range_s: int,
    is_bipartite: bool,
    dst_offset: int,
    device: torch.device,
    fanout: int,
    max_edges: int,
    seed: int,
    k_value: int = 10,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    if edges.shape[0] == 0:
        return float("nan"), float("nan")
    if max_edges > 0 and edges.shape[0] > max_edges:
        idx = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[idx]
        ts_s = ts_s[idx]

    src = edges[:, 0].astype(np.int64, copy=False)
    dst_local = edges[:, 1].astype(np.int64, copy=False)
    dst_global = dst_local + int(dst_offset) if is_bipartite else dst_local

    # Some datasets (notably tgbl-wiki-v2) can have per-example negative lists
    # with slightly varying lengths, so we avoid assuming a fixed K.
    neg_lists: list[np.ndarray] = []
    lens = np.empty((edges.shape[0],), dtype=np.int64)
    for i in range(edges.shape[0]):
        key = (int(src[i]), int(dst_global[i]), int(ts_s[i]))
        neg_g = np.asarray(neg_dict[key], dtype=np.int64)
        if is_bipartite:
            neg_l = neg_g - int(dst_offset)
        else:
            neg_l = neg_g
        neg_l = neg_l.astype(np.int64, copy=False)
        neg_lists.append(neg_l)
        lens[i] = int(neg_l.shape[0])
    neg_ptr = np.zeros((edges.shape[0] + 1,), dtype=np.int64)
    np.cumsum(lens, out=neg_ptr[1:])
    neg_flat = np.concatenate(neg_lists, axis=0) if neg_lists else np.zeros((0,), dtype=np.int64)
    neg_row = np.repeat(np.arange(edges.shape[0], dtype=np.int64), lens.astype(np.int64, copy=False))

    src_t = torch.from_numpy(src).to(device=device, dtype=torch.long)
    pos_dst_t = torch.from_numpy(dst_local).to(device=device, dtype=torch.long)
    neg_flat_t = torch.from_numpy(neg_flat).to(device=device, dtype=torch.long)
    neg_row_t = torch.from_numpy(neg_row).to(device=device, dtype=torch.long)

    z_src = model.encode_src(
        src_t,
        adj_src_to_events=adj_src_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    )
    z_pos = model.encode_dst(
        pos_dst_t,
        adj_dst_to_events=adj_dst_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    )
    pos_score = (z_src * z_pos).sum(dim=1)  # [N]

    z_neg_flat = model.encode_dst(
        neg_flat_t,
        adj_dst_to_events=adj_dst_to_events,
        event_src=event_src,
        event_dst=event_dst,
        event_ts_s=event_ts_s,
        event_w=event_w,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        fanout=fanout,
        rng=rng,
        device=device,
    )  # [sumK, H]

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
    parser = argparse.ArgumentParser(
        description="Relational baseline (event-as-node PK/FK schema) for RelBench-exported TGB link datasets."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--train_adj", default="val", help="Adjacency cutoff for training graph: val | test | all | <unix_seconds>")
    parser.add_argument("--eval_adj_test", default="test", help="Adjacency cutoff for test eval graph: test | all | <unix_seconds>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fanout", type=int, default=10)
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", default=None, help="Optional directory to save model checkpoint + config.")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--wandb_project", default="tgb2-relational")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    db_dir = exports_root / args.dataset / "db"
    is_bipartite, n_src, n_dst = _detect_bipartite(db_dir)
    num_dst_nodes = n_dst if is_bipartite else n_src

    train_suffix = _suffix_from_adj_arg(meta, args.train_adj)
    (adj_src_train, adj_dst_train, event_src_train, event_dst_train, event_ts_train, event_w_train) = _load_rel_event_arrays(
        exports_root, args.dataset, suffix=train_suffix
    )
    ts_min_s = int(np.min(event_ts_train)) if event_ts_train.size else 0
    ts_max_s = int(np.max(event_ts_train)) if event_ts_train.size else 0
    ts_range_s = max(1, ts_max_s - ts_min_s)

    # Training edges: use the train adjacency's event arrays (events <= val cutoff).
    train_src_all = np.asarray(event_src_train, dtype=np.int64)
    train_dst_all = np.asarray(event_dst_train, dtype=np.int64)
    num_train = int(train_src_all.shape[0])
    rng = np.random.default_rng(int(args.seed))
    if args.max_train_edges is not None and args.max_train_edges > 0 and num_train > int(args.max_train_edges):
        chosen = rng.choice(num_train, size=int(args.max_train_edges), replace=False)
        train_src_all = train_src_all[chosen]
        train_dst_all = train_dst_all[chosen]
        num_train = int(train_src_all.shape[0])

    config = {
        "dataset": args.dataset,
        "train_adj": args.train_adj,
        "eval_mode": args.eval_mode,
        "fanout": args.fanout,
        "batch_size": args.batch_size,
        "emb_dim": args.emb_dim,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "num_neg_train": args.num_neg_train,
        "num_neg_eval": args.num_neg_eval,
    }
    wb = _maybe_init_wandb(args, config=config)

    device = torch.device(args.device)
    model = RelEventSAGE(
        is_bipartite=is_bipartite,
        num_src_nodes=n_src,
        num_dst_nodes=num_dst_nodes,
        emb_dim=int(args.emb_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    steps_per_epoch = max(1, math.ceil(num_train / int(args.batch_size)))
    print(
        f"Dataset={args.dataset} bipartite={is_bipartite} train_edges={num_train} "
        f"src_nodes={n_src} dst_nodes={num_dst_nodes} train_adj_suffix={train_suffix}"
    )

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = rng.permutation(num_train)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            idx = perm[step * int(args.batch_size) : (step + 1) * int(args.batch_size)]
            if idx.size == 0:
                continue
            src = torch.from_numpy(train_src_all[idx]).to(device=device, dtype=torch.long)
            pos_dst = torch.from_numpy(train_dst_all[idx]).to(device=device, dtype=torch.long)
            neg_dst = torch.randint(0, num_dst_nodes, (src.shape[0], int(args.num_neg_train)), device=device, dtype=torch.long)

            batch_rng = np.random.default_rng(int(args.seed) * 1_000_000 + epoch * 10_000 + step)

            z_src = model.encode_src(
                src,
                adj_src_to_events=adj_src_train,
                event_src=np.asarray(event_src_train),
                event_dst=np.asarray(event_dst_train),
                event_ts_s=np.asarray(event_ts_train),
                event_w=np.asarray(event_w_train),
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                fanout=int(args.fanout),
                rng=batch_rng,
                device=device,
            )
            z_pos = model.encode_dst(
                pos_dst,
                adj_dst_to_events=adj_dst_train,
                event_src=np.asarray(event_src_train),
                event_dst=np.asarray(event_dst_train),
                event_ts_s=np.asarray(event_ts_train),
                event_w=np.asarray(event_w_train),
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                fanout=int(args.fanout),
                rng=batch_rng,
                device=device,
            )
            pos_score = (z_src * z_pos).sum(dim=1, keepdim=True)  # [B,1]

            neg_flat = neg_dst.reshape(-1)
            z_neg = model.encode_dst(
                neg_flat,
                adj_dst_to_events=adj_dst_train,
                event_src=np.asarray(event_src_train),
                event_dst=np.asarray(event_dst_train),
                event_ts_s=np.asarray(event_ts_train),
                event_w=np.asarray(event_w_train),
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                fanout=int(args.fanout),
                rng=batch_rng,
                device=device,
            ).view(src.shape[0], int(args.num_neg_train), -1)
            neg_score = torch.einsum("bd,bnd->bn", z_src, z_neg)  # [B, num_neg]

            # Stable BPR loss: -log(sigmoid(pos - neg)) == softplus(neg - pos)
            loss = F.softplus(neg_score - pos_score).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / float(steps_per_epoch)

        # Eval val/test.
        model.eval()
        if args.eval_mode == "tgb":
            relbench_cache_root = Path(args.relbench_cache_root)
            val_edges, val_ts_s = _load_official_task_edges(relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="val")
            test_edges, test_ts_s = _load_official_task_edges(relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="test")
            val_neg = _load_official_negatives(relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="val")
            test_neg = _load_official_negatives(relbench_cache_root=relbench_cache_root, dataset=args.dataset, split="test")

            val_mrr, val_hits = _evaluate_tgb_official(
                model,
                adj_src_train,
                adj_dst_train,
                np.asarray(event_src_train),
                np.asarray(event_dst_train),
                np.asarray(event_ts_train),
                np.asarray(event_w_train),
                edges=val_edges,
                ts_s=val_ts_s,
                neg_dict=val_neg,
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                is_bipartite=is_bipartite,
                dst_offset=n_src,
                device=device,
                fanout=int(args.fanout),
                max_edges=int(args.max_eval_edges),
                seed=int(args.seed) + 13 + epoch,
                k_value=10,
            )
        else:
            val_edges, _, _, _ = _load_edges_by_time(
                exports_root,
                args.dataset,
                start_s_exclusive=meta.val_timestamp_s,
                end_s_inclusive=meta.test_timestamp_s,
                parquet_batch_size=int(args.parquet_batch_size),
                max_edges=int(args.max_eval_edges),
                seed=int(args.seed) + 7,
            )
            val_mrr = _evaluate_mrr(
                model,
                adj_src_train,
                adj_dst_train,
                np.asarray(event_src_train),
                np.asarray(event_dst_train),
                np.asarray(event_ts_train),
                np.asarray(event_w_train),
                edges=val_edges,
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                num_neg=int(args.num_neg_eval),
                num_dst=num_dst_nodes,
                device=device,
                fanout=int(args.fanout),
                max_edges=int(args.max_eval_edges),
                seed=int(args.seed) + 13 + epoch,
            )

        test_mrr = float("nan")
        test_hits = float("nan")
        try:
            test_suffix = _suffix_from_adj_arg(meta, args.eval_adj_test)
            (adj_src_test, adj_dst_test, event_src_test, event_dst_test, event_ts_test, event_w_test) = _load_rel_event_arrays(
                exports_root, args.dataset, suffix=test_suffix
            )
            ts_min_t = int(np.min(event_ts_test)) if event_ts_test.size else ts_min_s
            ts_max_t = int(np.max(event_ts_test)) if event_ts_test.size else ts_max_s
            ts_range_t = max(1, ts_max_t - ts_min_t)
            if args.eval_mode == "tgb":
                test_mrr, test_hits = _evaluate_tgb_official(
                    model,
                    adj_src_test,
                    adj_dst_test,
                    np.asarray(event_src_test),
                    np.asarray(event_dst_test),
                    np.asarray(event_ts_test),
                    np.asarray(event_w_test),
                    edges=test_edges,
                    ts_s=test_ts_s,
                    neg_dict=test_neg,
                    ts_min_s=ts_min_t,
                    ts_range_s=ts_range_t,
                    is_bipartite=is_bipartite,
                    dst_offset=n_src,
                    device=device,
                    fanout=int(args.fanout),
                    max_edges=int(args.max_eval_edges),
                    seed=int(args.seed) + 17 + epoch,
                    k_value=10,
                )
            else:
                test_edges2, _, _, _ = _load_edges_by_time(
                    exports_root,
                    args.dataset,
                    start_s_exclusive=meta.test_timestamp_s,
                    end_s_inclusive=None,
                    parquet_batch_size=int(args.parquet_batch_size),
                    max_edges=int(args.max_eval_edges),
                    seed=int(args.seed) + 9,
                )
                test_mrr = _evaluate_mrr(
                    model,
                    adj_src_test,
                    adj_dst_test,
                    np.asarray(event_src_test),
                    np.asarray(event_dst_test),
                    np.asarray(event_ts_test),
                    np.asarray(event_w_test),
                    edges=test_edges2,
                    ts_min_s=ts_min_t,
                    ts_range_s=ts_range_t,
                    num_neg=int(args.num_neg_eval),
                    num_dst=num_dst_nodes,
                    device=device,
                    fanout=int(args.fanout),
                    max_edges=int(args.max_eval_edges),
                    seed=int(args.seed) + 17 + epoch,
                )
        except FileNotFoundError:
            pass

        if args.eval_mode == "tgb":
            print(
                f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} val_hits@10={val_hits:.4f} "
                f"test_mrr={test_mrr:.4f} test_hits@10={test_hits:.4f}"
            )
            if wb is not None:
                wb.log(
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
            print(f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} test_mrr={test_mrr:.4f}")
            if wb is not None:
                wb.log({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr, "test_mrr": test_mrr})

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "dataset": args.dataset,
            "meta": {"val_timestamp_s": meta.val_timestamp_s, "test_timestamp_s": meta.test_timestamp_s},
            "config": config,
            "state_dict": model.state_dict(),
        }
        ckpt_path = save_dir / f"releventsage_{args.dataset}.pt"
        torch.save(out, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
