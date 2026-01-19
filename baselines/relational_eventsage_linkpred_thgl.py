#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
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
    node_type_count: int
    edge_type_count: int


def _load_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    extra = meta.get("extra", {})
    return ExportMeta(
        dataset=meta["dataset"],
        tgb_internal_name=meta["tgb_internal_name"],
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
        node_type_count=int(extra.get("node_type_count", 0)),
        edge_type_count=int(extra.get("edge_type_count", 0)),
    )


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _parse_node_type_from_table_name(table: str) -> int:
    m = re.fullmatch(r"nodes_type_(\d+)", table)
    if not m:
        raise ValueError(f"Unexpected node table name: {table}")
    return int(m.group(1))


def _parse_edge_type_from_file(path: Path) -> int:
    m = re.fullmatch(r"events_edge_type_(\d+)\.parquet", path.name)
    if not m:
        raise ValueError(f"Unexpected event table filename: {path.name}")
    return int(m.group(1))


def _read_relbench_parquet_metadata(path: Path) -> dict[str, object]:
    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}
    out: dict[str, object] = {}
    for key in (b"fkey_col_to_pkey_table", b"pkey_col", b"time_col"):
        if key in md:
            out[key.decode("utf-8")] = json.loads(md[key].decode("utf-8"))
    return out


def _iter_event_batches(events_path: Path, batch_size: int):
    pf = pq.ParquetFile(events_path)
    yield from pf.iter_batches(batch_size=batch_size, columns=["src_id", "dst_id", "event_ts"])


def _load_edges_by_time_thgl(
    exports_root: Path,
    dataset: str,
    *,
    start_s_exclusive: Optional[int],
    end_s_inclusive: Optional[int],
    parquet_batch_size: int,
    max_edges: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    db_dir = exports_root / dataset / "db"
    event_paths = sorted(db_dir.glob("events_edge_type_*.parquet"), key=lambda p: p.name)
    if not event_paths:
        raise FileNotFoundError(f"No thgl event tables found under {db_dir}")

    total_rows = sum(int(pq.ParquetFile(p).metadata.num_rows) for p in event_paths)
    p_keep = 1.0
    if max_edges > 0 and total_rows > 0:
        p_keep = min(1.0, float(max_edges) / float(total_rows))
    rng = np.random.default_rng(seed)

    src_types: list[np.ndarray] = []
    dst_types: list[np.ndarray] = []
    edge_types: list[np.ndarray] = []
    src_ids: list[np.ndarray] = []
    dst_ids: list[np.ndarray] = []

    for path in event_paths:
        et = _parse_edge_type_from_file(path)
        md = _read_relbench_parquet_metadata(path)
        fkeys = md.get("fkey_col_to_pkey_table")
        if not isinstance(fkeys, dict):
            raise RuntimeError(f"Missing RelBench metadata on {path}")
        src_tbl = fkeys.get("src_id")
        dst_tbl = fkeys.get("dst_id")
        if not isinstance(src_tbl, str) or not isinstance(dst_tbl, str):
            raise RuntimeError(f"Malformed fkey metadata on {path}: {fkeys}")
        src_t = _parse_node_type_from_table_name(src_tbl)
        dst_t = _parse_node_type_from_table_name(dst_tbl)

        for batch in _iter_event_batches(path, parquet_batch_size):
            src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))

            mask = np.ones_like(ts_s, dtype=bool)
            if start_s_exclusive is not None:
                mask &= ts_s > int(start_s_exclusive)
            if end_s_inclusive is not None:
                mask &= ts_s <= int(end_s_inclusive)
            if p_keep < 1.0 and mask.any():
                keep = rng.random(mask.shape[0]) < p_keep
                mask &= keep
            if not mask.any():
                continue

            src_m = src[mask]
            dst_m = dst[mask]
            k = int(src_m.shape[0])
            src_types.append(np.full(k, src_t, dtype=np.int16))
            dst_types.append(np.full(k, dst_t, dtype=np.int16))
            edge_types.append(np.full(k, et, dtype=np.int16))
            src_ids.append(src_m.astype(np.int64, copy=False))
            dst_ids.append(dst_m.astype(np.int64, copy=False))

    if not src_ids:
        return (
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,), dtype=np.int16),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    src_t_all = np.concatenate(src_types, axis=0)
    dst_t_all = np.concatenate(dst_types, axis=0)
    et_all = np.concatenate(edge_types, axis=0)
    src_all = np.concatenate(src_ids, axis=0)
    dst_all = np.concatenate(dst_ids, axis=0)

    if max_edges > 0 and src_all.shape[0] > max_edges:
        idx = rng.choice(src_all.shape[0], size=max_edges, replace=False)
        src_t_all = src_t_all[idx]
        dst_t_all = dst_t_all[idx]
        et_all = et_all[idx]
        src_all = src_all[idx]
        dst_all = dst_all[idx]

    return src_t_all, dst_t_all, et_all, src_all, dst_all


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


def _suffix_from_adj_arg(meta: ExportMeta, adj: str) -> str:
    if adj == "val":
        return f"upto_{meta.val_timestamp_s}"
    if adj == "test":
        return f"upto_{meta.test_timestamp_s}"
    if adj == "all":
        return "all"
    return f"upto_{int(adj)}"


@dataclass(frozen=True)
class ThglAdj:
    node_type_to_n: dict[int, int]
    src_adj: dict[int, CSRAdjacency]
    dst_adj: dict[int, CSRAdjacency]
    ev_src_type: np.ndarray
    ev_dst_type: np.ndarray
    ev_edge_type: np.ndarray
    ev_src_id: np.ndarray
    ev_dst_id: np.ndarray
    ev_ts_s: np.ndarray
    ev_w: np.ndarray


def _load_thgl_rel_arrays(exports_root: Path, dataset: str, *, suffix: str) -> ThglAdj:
    db_dir = exports_root / dataset / "db"
    node_paths = sorted(db_dir.glob("nodes_type_*.parquet"), key=lambda p: p.name)
    node_type_to_n: dict[int, int] = {}
    for p in node_paths:
        t = _parse_node_type_from_table_name(p.stem)
        node_type_to_n[t] = int(pq.ParquetFile(p).metadata.num_rows)

    adj_dir = exports_root / dataset / "adj"
    src_adj: dict[int, CSRAdjacency] = {}
    dst_adj: dict[int, CSRAdjacency] = {}
    for t in node_type_to_n.keys():
        src_indptr = adj_dir / f"csr_src_events_indptr_nodes_type_{t}_{suffix}.npy"
        src_indices = adj_dir / f"csr_src_events_indices_nodes_type_{t}_{suffix}.npy"
        dst_indptr = adj_dir / f"csr_dst_events_indptr_nodes_type_{t}_{suffix}.npy"
        dst_indices = adj_dir / f"csr_dst_events_indices_nodes_type_{t}_{suffix}.npy"
        if not (src_indptr.exists() and src_indices.exists() and dst_indptr.exists() and dst_indices.exists()):
            raise FileNotFoundError(
                f"Missing thgl CSR arrays for nodes_type_{t} ({suffix}). "
                f"Build via `scripts/build_rel_event_csr_thgl.py --dataset {dataset} --upto {suffix.replace('upto_', '')}`."
            )
        src_adj[t] = CSRAdjacency.load(src_indptr, src_indices)
        dst_adj[t] = CSRAdjacency.load(dst_indptr, dst_indices)

    # Global event arrays:
    ev_src_type = np.load(adj_dir / f"events_src_type_{suffix}.npy", mmap_mode="r")
    ev_dst_type = np.load(adj_dir / f"events_dst_type_{suffix}.npy", mmap_mode="r")
    ev_edge_type = np.load(adj_dir / f"events_edge_type_{suffix}.npy", mmap_mode="r")
    ev_src_id = np.load(adj_dir / f"events_src_id_{suffix}.npy", mmap_mode="r")
    ev_dst_id = np.load(adj_dir / f"events_dst_id_{suffix}.npy", mmap_mode="r")
    ev_ts_s = np.load(adj_dir / f"events_ts_s_{suffix}.npy", mmap_mode="r")
    ev_w = np.load(adj_dir / f"events_weight_{suffix}.npy", mmap_mode="r")

    return ThglAdj(
        node_type_to_n=node_type_to_n,
        src_adj=src_adj,
        dst_adj=dst_adj,
        ev_src_type=ev_src_type,
        ev_dst_type=ev_dst_type,
        ev_edge_type=ev_edge_type,
        ev_src_id=ev_src_id,
        ev_dst_id=ev_dst_id,
        ev_ts_s=ev_ts_s,
        ev_w=ev_w,
    )


class HeteroRelEventSAGE(nn.Module):
    def __init__(
        self,
        *,
        node_type_to_n: dict[int, int],
        num_edge_types: int,
        emb_dim: int,
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.node_types = sorted(node_type_to_n.keys())
        self.node_emb = nn.ModuleDict(
            {str(t): nn.Embedding(int(node_type_to_n[t]), emb_dim) for t in self.node_types}
        )

        self.edge_emb = nn.Embedding(int(num_edge_types), emb_dim)
        self.edge_lin = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.event_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.event_src_lin = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.event_dst_lin = nn.Linear(emb_dim, hidden_dim, bias=False)

        self.src_self = nn.ModuleDict({str(t): nn.Linear(emb_dim, hidden_dim, bias=False) for t in self.node_types})
        self.src_neigh = nn.ModuleDict({str(t): nn.Linear(hidden_dim, hidden_dim, bias=False) for t in self.node_types})
        self.dst_self = nn.ModuleDict({str(t): nn.Linear(emb_dim, hidden_dim, bias=False) for t in self.node_types})
        self.dst_neigh = nn.ModuleDict({str(t): nn.Linear(hidden_dim, hidden_dim, bias=False) for t in self.node_types})

        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for emb in self.node_emb.values():
            nn.init.normal_(emb.weight, std=0.02)
        nn.init.normal_(self.edge_emb.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _embed_nodes(self, node_type: torch.Tensor, node_id: torch.Tensor) -> torch.Tensor:
        # node_type, node_id shape [N]
        device = node_id.device
        out = torch.empty((node_id.numel(), self.emb_dim), device=device)
        for t in torch.unique(node_type).tolist():
            t_int = int(t)
            mask = node_type == t_int
            out[mask] = self.node_emb[str(t_int)](node_id[mask])
        return out

    def _event_repr(
        self,
        event_ids: torch.Tensor,
        *,
        ev_src_type: np.ndarray,
        ev_dst_type: np.ndarray,
        ev_edge_type: np.ndarray,
        ev_src_id: np.ndarray,
        ev_dst_id: np.ndarray,
        ev_ts_s: np.ndarray,
        ev_w: np.ndarray,
        ts_min_s: int,
        ts_range_s: int,
        device: torch.device,
    ) -> torch.Tensor:
        valid = event_ids >= 0
        out = torch.zeros((event_ids.numel(), self.hidden_dim), device=device)
        if not valid.any():
            return out

        ev = event_ids[valid].detach().cpu().numpy().astype(np.int64, copy=False)
        src_t = torch.from_numpy(ev_src_type[ev].astype(np.int64, copy=False)).to(device=device)
        dst_t = torch.from_numpy(ev_dst_type[ev].astype(np.int64, copy=False)).to(device=device)
        src_id = torch.from_numpy(ev_src_id[ev].astype(np.int64, copy=False)).to(device=device)
        dst_id = torch.from_numpy(ev_dst_id[ev].astype(np.int64, copy=False)).to(device=device)
        rel = torch.from_numpy(ev_edge_type[ev].astype(np.int64, copy=False)).to(device=device)

        src_e = self._embed_nodes(src_t, src_id)
        dst_e = self._embed_nodes(dst_t, dst_id)
        rel_e = self.edge_emb(rel)

        ts = torch.from_numpy(ev_ts_s[ev].astype(np.int64, copy=False)).to(device=device, dtype=torch.float32)
        w = torch.from_numpy(ev_w[ev].astype(np.float32, copy=False)).to(device=device, dtype=torch.float32)
        w = torch.log1p(w)
        ts_norm = (ts - float(ts_min_s)) / float(ts_range_s) if ts_range_s > 0 else ts * 0.0
        feat = torch.stack([ts_norm, w], dim=1)

        ev_h = self.event_mlp(feat) + self.event_src_lin(src_e) + self.event_dst_lin(dst_e) + self.edge_lin(rel_e)
        ev_h = self.act(ev_h)
        ev_h = self.drop(ev_h)
        out[valid] = ev_h
        return out

    def encode_nodes_of_type(
        self,
        t: int,
        node_ids: torch.Tensor,
        *,
        role: str,
        adj: CSRAdjacency,
        fanout: int,
        rng: np.random.Generator,
        ev_arrays: ThglAdj,
        ts_min_s: int,
        ts_range_s: int,
        device: torch.device,
    ) -> torch.Tensor:
        node_ids = node_ids.to(device=device, dtype=torch.long)
        seeds_np = node_ids.detach().cpu().numpy().astype(np.int64, copy=False)
        nbr_ev = adj.sample_neighbors(seeds_np, fanout, rng=rng)
        nbr_ev_t = torch.from_numpy(nbr_ev.reshape(-1)).to(device=device, dtype=torch.long)
        ev_h = self._event_repr(
            nbr_ev_t,
            ev_src_type=ev_arrays.ev_src_type,
            ev_dst_type=ev_arrays.ev_dst_type,
            ev_edge_type=ev_arrays.ev_edge_type,
            ev_src_id=ev_arrays.ev_src_id,
            ev_dst_id=ev_arrays.ev_dst_id,
            ev_ts_s=ev_arrays.ev_ts_s,
            ev_w=ev_arrays.ev_w,
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            device=device,
        ).view(node_ids.shape[0], fanout, -1)
        neigh = ev_h.mean(dim=1)

        self_e = self.node_emb[str(t)](node_ids)
        if role == "src":
            out = self.act(self.src_self[str(t)](self_e) + self.src_neigh[str(t)](neigh))
        elif role == "dst":
            out = self.act(self.dst_self[str(t)](self_e) + self.dst_neigh[str(t)](neigh))
        else:
            raise ValueError(f"Unknown role: {role}")
        out = self.drop(out)
        return out


def _encode_mixed(
    model: HeteroRelEventSAGE,
    node_types: torch.Tensor,
    node_ids: torch.Tensor,
    *,
    role: str,
    fanout: int,
    rng: np.random.Generator,
    ev_arrays: ThglAdj,
    ts_min_s: int,
    ts_range_s: int,
    device: torch.device,
) -> torch.Tensor:
    out = torch.empty((node_ids.numel(), model.hidden_dim), device=device)
    for t in torch.unique(node_types).tolist():
        t_int = int(t)
        mask = node_types == t_int
        adj = ev_arrays.src_adj[t_int] if role == "src" else ev_arrays.dst_adj[t_int]
        out[mask] = model.encode_nodes_of_type(
            t_int,
            node_ids[mask],
            role=role,
            adj=adj,
            fanout=fanout,
            rng=rng,
            ev_arrays=ev_arrays,
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            device=device,
        )
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
        import os

        os.environ.setdefault("WANDB_MODE", "offline")
    return wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=config)


@torch.no_grad()
def _evaluate_mrr(
    model: HeteroRelEventSAGE,
    ev_arrays: ThglAdj,
    *,
    src_type: np.ndarray,
    dst_type: np.ndarray,
    edge_type: np.ndarray,
    src_id: np.ndarray,
    dst_id: np.ndarray,
    ts_min_s: int,
    ts_range_s: int,
    num_neg: int,
    fanout: int,
    max_edges: int,
    seed: int,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(seed)
    n = int(src_id.shape[0])
    if n == 0:
        return float("nan")
    if max_edges > 0 and n > max_edges:
        idx = rng.choice(n, size=max_edges, replace=False)
        src_type = src_type[idx]
        dst_type = dst_type[idx]
        edge_type = edge_type[idx]
        src_id = src_id[idx]
        dst_id = dst_id[idx]
        n = int(src_id.shape[0])

    src_t = torch.from_numpy(src_type.astype(np.int64, copy=False)).to(device=device)
    dst_t = torch.from_numpy(dst_type.astype(np.int64, copy=False)).to(device=device)
    rel_t = torch.from_numpy(edge_type.astype(np.int64, copy=False)).to(device=device)
    src = torch.from_numpy(src_id.astype(np.int64, copy=False)).to(device=device)
    pos_dst = torch.from_numpy(dst_id.astype(np.int64, copy=False)).to(device=device)

    z_src = _encode_mixed(
        model,
        src_t,
        src,
        role="src",
        fanout=fanout,
        rng=rng,
        ev_arrays=ev_arrays,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        device=device,
    )
    z_pos = _encode_mixed(
        model,
        dst_t,
        pos_dst,
        role="dst",
        fanout=fanout,
        rng=rng,
        ev_arrays=ev_arrays,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        device=device,
    )
    rel_h = model.edge_lin(model.edge_emb(rel_t))
    pos_score = ((z_src + rel_h) * z_pos).sum(dim=1, keepdim=True)

    # Sample type-correct negatives.
    neg_ids = torch.empty((n, num_neg), device=device, dtype=torch.long)
    for t in torch.unique(dst_t).tolist():
        t_int = int(t)
        mask = (dst_t == t_int).nonzero(as_tuple=False).view(-1)
        if mask.numel() == 0:
            continue
        max_id = int(ev_arrays.node_type_to_n[t_int])
        neg_ids[mask] = torch.randint(0, max_id, (mask.numel(), num_neg), device=device, dtype=torch.long)

    neg_flat = neg_ids.reshape(-1)
    neg_types = dst_t.repeat_interleave(num_neg)
    z_neg = _encode_mixed(
        model,
        neg_types,
        neg_flat,
        role="dst",
        fanout=fanout,
        rng=rng,
        ev_arrays=ev_arrays,
        ts_min_s=ts_min_s,
        ts_range_s=ts_range_s,
        device=device,
    ).view(n, num_neg, -1)
    neg_score = torch.einsum("bd,bnd->bn", z_src + rel_h, z_neg)

    rank = 1 + (neg_score >= pos_score).sum(dim=1)
    return float((1.0 / rank.float()).mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Relational event-as-node baseline for thgl-* link prediction exports.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--train_adj", default="val", help="Adjacency cutoff for training graph: val | test | all | <unix_seconds>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neg_train", type=int, default=1)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--max_train_edges", type=int, default=200000)
    parser.add_argument("--max_eval_edges", type=int, default=20000)
    parser.add_argument("--parquet_batch_size", type=int, default=500000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save_dir", default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    parser.add_argument("--wandb_project", default="tgb2-relational")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    args = parser.parse_args()

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    suffix = _suffix_from_adj_arg(meta, args.train_adj)
    ev_arrays = _load_thgl_rel_arrays(exports_root, args.dataset, suffix=suffix)

    ts_min_s = int(np.min(ev_arrays.ev_ts_s)) if ev_arrays.ev_ts_s.size else 0
    ts_max_s = int(np.max(ev_arrays.ev_ts_s)) if ev_arrays.ev_ts_s.size else 0
    ts_range_s = max(1, ts_max_s - ts_min_s)

    num_events = int(ev_arrays.ev_src_id.shape[0])
    rng = np.random.default_rng(int(args.seed))
    if args.max_train_edges > 0 and num_events > int(args.max_train_edges):
        train_idx = rng.choice(num_events, size=int(args.max_train_edges), replace=False)
    else:
        train_idx = np.arange(num_events, dtype=np.int64)

    config = {
        "dataset": args.dataset,
        "train_adj": args.train_adj,
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
    model = HeteroRelEventSAGE(
        node_type_to_n=ev_arrays.node_type_to_n,
        num_edge_types=max(1, int(meta.edge_type_count)),
        emb_dim=int(args.emb_dim),
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    steps_per_epoch = max(1, math.ceil(train_idx.shape[0] / int(args.batch_size)))
    print(f"Dataset={args.dataset} train_events={train_idx.shape[0]} node_types={len(ev_arrays.node_type_to_n)} edge_types={meta.edge_type_count} suffix={suffix}")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        perm = rng.permutation(train_idx.shape[0])
        total_loss = 0.0
        for step in range(steps_per_epoch):
            sel = perm[step * int(args.batch_size) : (step + 1) * int(args.batch_size)]
            if sel.size == 0:
                continue
            ev_id = train_idx[sel]

            src_t = torch.from_numpy(ev_arrays.ev_src_type[ev_id].astype(np.int64, copy=False)).to(device=device)
            dst_t = torch.from_numpy(ev_arrays.ev_dst_type[ev_id].astype(np.int64, copy=False)).to(device=device)
            rel_t = torch.from_numpy(ev_arrays.ev_edge_type[ev_id].astype(np.int64, copy=False)).to(device=device)
            src = torch.from_numpy(ev_arrays.ev_src_id[ev_id].astype(np.int64, copy=False)).to(device=device)
            pos_dst = torch.from_numpy(ev_arrays.ev_dst_id[ev_id].astype(np.int64, copy=False)).to(device=device)

            # Negatives (type-correct).
            num_neg = int(args.num_neg_train)
            neg_ids = torch.empty((src.shape[0], num_neg), device=device, dtype=torch.long)
            for t in torch.unique(dst_t).tolist():
                t_int = int(t)
                mask = (dst_t == t_int).nonzero(as_tuple=False).view(-1)
                if mask.numel() == 0:
                    continue
                max_id = int(ev_arrays.node_type_to_n[t_int])
                neg_ids[mask] = torch.randint(0, max_id, (mask.numel(), num_neg), device=device, dtype=torch.long)

            batch_rng = np.random.default_rng(int(args.seed) * 1_000_000 + epoch * 10_000 + step)
            z_src = _encode_mixed(
                model,
                src_t,
                src,
                role="src",
                fanout=int(args.fanout),
                rng=batch_rng,
                ev_arrays=ev_arrays,
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                device=device,
            )
            z_pos = _encode_mixed(
                model,
                dst_t,
                pos_dst,
                role="dst",
                fanout=int(args.fanout),
                rng=batch_rng,
                ev_arrays=ev_arrays,
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                device=device,
            )
            rel_h = model.edge_lin(model.edge_emb(rel_t))
            pos_score = ((z_src + rel_h) * z_pos).sum(dim=1, keepdim=True)

            neg_flat = neg_ids.reshape(-1)
            neg_types = dst_t.repeat_interleave(num_neg)
            z_neg = _encode_mixed(
                model,
                neg_types,
                neg_flat,
                role="dst",
                fanout=int(args.fanout),
                rng=batch_rng,
                ev_arrays=ev_arrays,
                ts_min_s=ts_min_s,
                ts_range_s=ts_range_s,
                device=device,
            ).view(src.shape[0], num_neg, -1)
            neg_score = torch.einsum("bd,bnd->bn", z_src + rel_h, z_neg)

            loss = F.softplus(neg_score - pos_score).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / float(steps_per_epoch)

        model.eval()
        val = _load_edges_by_time_thgl(
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
            ev_arrays,
            src_type=val[0],
            dst_type=val[1],
            edge_type=val[2],
            src_id=val[3],
            dst_id=val[4],
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            num_neg=int(args.num_neg_eval),
            fanout=int(args.fanout),
            max_edges=int(args.max_eval_edges),
            seed=int(args.seed) + 13 + epoch,
            device=device,
        )
        test = _load_edges_by_time_thgl(
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
            ev_arrays,
            src_type=test[0],
            dst_type=test[1],
            edge_type=test[2],
            src_id=test[3],
            dst_id=test[4],
            ts_min_s=ts_min_s,
            ts_range_s=ts_range_s,
            num_neg=int(args.num_neg_eval),
            fanout=int(args.fanout),
            max_edges=int(args.max_eval_edges),
            seed=int(args.seed) + 17 + epoch,
            device=device,
        )

        print(f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} test_mrr={test_mrr:.4f}")
        if wb is not None:
            wb.log({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr, "test_mrr": test_mrr})

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = save_dir / f"releventsage_{args.dataset}.pt"
        torch.save({"dataset": args.dataset, "meta": {"val_timestamp_s": meta.val_timestamp_s, "test_timestamp_s": meta.test_timestamp_s}, "config": config, "state_dict": model.state_dict()}, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()

