#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn


@dataclass(frozen=True)
class ExportMeta:
    dataset: str
    tgb_internal_name: str
    val_timestamp_s: int
    test_timestamp_s: int
    tables: list[str]


def _load_export_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    return ExportMeta(
        dataset=meta["dataset"],
        tgb_internal_name=meta["tgb_internal_name"],
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
        tables=list(meta["tables"]),
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
    run = wandb.init(project=args.wandb_project, entity=args.wandb_entity, name=args.wandb_name, config=config)
    return run


def _iter_parquet_batches(path: Path, batch_size: int, columns: list[str]):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        yield batch


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    # arr is a PyArrow TimestampArray; convert to int seconds without pandas.
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _load_node_counts(exports_root: Path, dataset: str) -> tuple[bool, int, int]:
    db_dir = exports_root / dataset / "db"
    if (db_dir / "src_nodes.parquet").exists() and (db_dir / "dst_nodes.parquet").exists():
        n_src = pq.ParquetFile(db_dir / "src_nodes.parquet").metadata.num_rows
        n_dst = pq.ParquetFile(db_dir / "dst_nodes.parquet").metadata.num_rows
        return True, int(n_src), int(n_dst)
    n = pq.ParquetFile(db_dir / "nodes.parquet").metadata.num_rows
    return False, int(n), 0


def _build_coo_undirected(
    exports_root: Path,
    dataset: str,
    *,
    use_self_loops: bool,
    undirected: bool,
    parquet_batch_size: int,
    adj_max_edges: Optional[int],
    seed: int,
) -> tuple[torch.Tensor, int]:
    db_dir = exports_root / dataset / "db"
    events_path = db_dir / "events.parquet"
    is_bipartite, n_src, n_dst = _load_node_counts(exports_root, dataset)

    if is_bipartite:
        num_nodes = n_src + n_dst
        src_offset = 0
        dst_offset = n_src
    else:
        num_nodes = n_src
        src_offset = 0
        dst_offset = 0

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    total_edges = pq.ParquetFile(events_path).metadata.num_rows
    if adj_max_edges is not None and adj_max_edges > 0:
        p = min(1.0, float(adj_max_edges) / float(total_edges))
    else:
        p = 1.0

    rng = np.random.default_rng(seed)

    for batch in _iter_parquet_batches(events_path, parquet_batch_size, columns=["src_id", "dst_id"]):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        if p < 1.0:
            keep = rng.random(src.shape[0]) < p
            if not keep.any():
                continue
            src = src[keep]
            dst = dst[keep]
        r = src + src_offset
        c = dst + dst_offset
        rows.append(r)
        cols.append(c)
        if undirected:
            rows.append(c)
            cols.append(r)

    row = np.concatenate(rows, axis=0)
    col = np.concatenate(cols, axis=0)

    if use_self_loops:
        loop = np.arange(num_nodes, dtype=np.int64)
        row = np.concatenate([row, loop], axis=0)
        col = np.concatenate([col, loop], axis=0)

    idx = torch.from_numpy(np.stack([row, col], axis=0))
    return idx, num_nodes


def _normalize_adj(indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # Build A_hat with symmetric normalization: D^{-1/2} A D^{-1/2}
    device = indices.device
    row, col = indices[0], indices[1]
    deg = torch.bincount(row, minlength=num_nodes).to(device=device, dtype=torch.float32)
    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
    val = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj = torch.sparse_coo_tensor(indices, val, size=(num_nodes, num_nodes))
    return adj.coalesce()


class GCN(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        layers: list[nn.Module] = []
        in_dim = emb_dim
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb.weight, std=0.02)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        x = self.emb.weight
        x = torch.sparse.mm(adj, x)
        x = self.mlp(x)
        return x


def _sample_negatives(num_nodes: int, size: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_nodes, (size,), device=device, dtype=torch.long)


@torch.no_grad()
def _evaluate_mrr(
    z: torch.Tensor,
    edges: np.ndarray,
    *,
    num_neg: int,
    dst_offset: int,
    num_dst: int,
    device: torch.device,
    max_edges: Optional[int],
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    if max_edges is not None and edges.shape[0] > max_edges:
        idx = rng.choice(edges.shape[0], size=max_edges, replace=False)
        edges = edges[idx]

    src = torch.from_numpy(edges[:, 0]).to(device=device, dtype=torch.long)
    pos_dst = torch.from_numpy(edges[:, 1]).to(device=device, dtype=torch.long)

    if dst_offset:
        src_global = src
        pos_dst_global = pos_dst + dst_offset
    else:
        src_global = src
        pos_dst_global = pos_dst

    # Sample negatives on the destination side.
    neg_dst = torch.randint(0, num_dst, (src.shape[0], num_neg), device=device, dtype=torch.long)
    neg_dst_global = neg_dst + dst_offset if dst_offset else neg_dst

    src_z = z[src_global]  # [B, H]
    pos_score = (src_z * z[pos_dst_global]).sum(dim=1, keepdim=True)  # [B,1]
    neg_score = torch.einsum("bd,bnd->bn", src_z, z[neg_dst_global])  # [B,N]

    # rank = 1 + number of negatives with score >= pos (ties pessimistic)
    rank = 1 + (neg_score >= pos_score).sum(dim=1)
    mrr = (1.0 / rank.float()).mean().item()
    return float(mrr)


def _load_edges_by_time(
    exports_root: Path,
    dataset: str,
    *,
    start_s: Optional[int],
    end_s_inclusive: Optional[int],
    parquet_batch_size: int,
    max_edges: Optional[int],
    seed: int,
) -> tuple[np.ndarray, bool, int, int]:
    db_dir = exports_root / dataset / "db"
    events_path = db_dir / "events.parquet"
    is_bipartite, n_src, n_dst = _load_node_counts(exports_root, dataset)

    total_edges = pq.ParquetFile(events_path).metadata.num_rows
    if max_edges is not None and max_edges > 0:
        p = min(1.0, float(max_edges) / float(total_edges))
    else:
        p = 1.0
    rng = np.random.default_rng(seed)

    edges: list[np.ndarray] = []
    for batch in _iter_parquet_batches(events_path, parquet_batch_size, columns=["src_id", "dst_id", "event_ts"]):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))

        mask = np.ones_like(ts_s, dtype=bool)
        if start_s is not None:
            mask &= ts_s > int(start_s)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Static GCN baseline for TGB (RelBench-exported) link prediction datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset name under relbench_exports/, e.g. tgbl-wiki-v2")
    parser.add_argument("--exports_root", default="relbench_exports", help="Directory containing exported databases.")
    parser.add_argument("--device", default="cpu", help="torch device, e.g. cpu or cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--num_neg_train", type=int, default=1)
    parser.add_argument("--num_neg_eval", type=int, default=100)
    parser.add_argument("--max_eval_edges", type=int, default=20000)
    parser.add_argument("--parquet_batch_size", type=int, default=1_000_000)
    parser.add_argument("--adj_max_edges", type=int, default=None, help="Max edges sampled to build adjacency (approx).")
    parser.add_argument("--max_train_edges", type=int, default=None, help="Max edges sampled for training set (approx).")
    parser.add_argument("--max_val_edges", type=int, default=None, help="Max edges sampled for validation set (approx).")
    parser.add_argument("--max_test_edges", type=int, default=None, help="Max edges sampled for test set (approx).")
    parser.add_argument("--undirected", action="store_true", help="Symmetrize edges for message passing.")
    parser.add_argument("--no_self_loops", action="store_true")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", default="tgb-relbench-gcn")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_name", default=None)
    parser.add_argument("--wandb_mode", default="offline", choices=["offline", "online", "disabled"])
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    exports_root = Path(args.exports_root)
    meta = _load_export_meta(exports_root, args.dataset)

    device = torch.device(args.device)
    indices, num_nodes = _build_coo_undirected(
        exports_root,
        args.dataset,
        use_self_loops=not args.no_self_loops,
        undirected=args.undirected,
        parquet_batch_size=args.parquet_batch_size,
        adj_max_edges=args.adj_max_edges,
        seed=args.seed,
    )
    adj = _normalize_adj(indices.to(device), num_nodes).to(device)

    # For evaluation sampling on the destination side:
    is_bipartite, n_src, n_dst = _load_node_counts(exports_root, args.dataset)
    dst_offset = n_src if is_bipartite else 0
    num_dst = n_dst if is_bipartite else num_nodes

    model = GCN(num_nodes=num_nodes, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, num_layers=args.layers, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    bce = nn.BCEWithLogitsLoss()

    run = _maybe_init_wandb(
        args,
        config={
            "dataset": args.dataset,
            "val_timestamp_s": meta.val_timestamp_s,
            "test_timestamp_s": meta.test_timestamp_s,
            "num_nodes": num_nodes,
            "bipartite": is_bipartite,
            "emb_dim": args.emb_dim,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "lr": args.lr,
            "wd": args.weight_decay,
            "batch_size": args.batch_size,
            "neg_train": args.num_neg_train,
            "neg_eval": args.num_neg_eval,
            "undirected": args.undirected,
        },
    )

    # Training edges = events with ts <= val_timestamp (inclusive), like RelBench’s db.upto(val_timestamp).
    train_edges, _, _, _ = _load_edges_by_time(
        exports_root,
        args.dataset,
        start_s=None,
        end_s_inclusive=meta.val_timestamp_s,
        parquet_batch_size=args.parquet_batch_size,
        max_edges=args.max_train_edges,
        seed=args.seed + 100,
    )

    # Validation edges = (val_timestamp, test_timestamp]
    val_edges, _, _, _ = _load_edges_by_time(
        exports_root,
        args.dataset,
        start_s=meta.val_timestamp_s,
        end_s_inclusive=meta.test_timestamp_s,
        parquet_batch_size=args.parquet_batch_size,
        max_edges=args.max_val_edges,
        seed=args.seed + 200,
    )

    # Test edges = (test_timestamp, +inf]
    test_edges, _, _, _ = _load_edges_by_time(
        exports_root,
        args.dataset,
        start_s=meta.test_timestamp_s,
        end_s_inclusive=None,
        parquet_batch_size=args.parquet_batch_size,
        max_edges=args.max_test_edges,
        seed=args.seed + 300,
    )

    if train_edges.shape[0] == 0:
        raise RuntimeError("No training edges found. Check timestamps/cutoffs.")

    perm = np.random.permutation(train_edges.shape[0])
    train_edges = train_edges[perm]

    steps_per_epoch = max(1, math.ceil(train_edges.shape[0] / args.batch_size))

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            batch = train_edges[step * args.batch_size : (step + 1) * args.batch_size]
            src = torch.from_numpy(batch[:, 0]).to(device=device, dtype=torch.long)
            dst = torch.from_numpy(batch[:, 1]).to(device=device, dtype=torch.long)
            if dst_offset:
                src_g = src
                dst_g = dst + dst_offset
            else:
                src_g = src
                dst_g = dst

            z = model(adj)
            pos_logit = (z[src_g] * z[dst_g]).sum(dim=1)

            # Negatives sampled on destination side.
            neg_dst = _sample_negatives(num_dst, src.shape[0] * args.num_neg_train, device=device)
            neg_dst_g = neg_dst + dst_offset if dst_offset else neg_dst
            src_rep = z[src_g].repeat_interleave(args.num_neg_train, dim=0)
            neg_logit = (src_rep * z[neg_dst_g]).sum(dim=1)

            loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(neg_logit, torch.zeros_like(neg_logit))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach())

        model.eval()
        with torch.no_grad():
            z = model(adj)
            val_mrr = _evaluate_mrr(
                z,
                val_edges,
                num_neg=args.num_neg_eval,
                dst_offset=dst_offset,
                num_dst=num_dst,
                device=device,
                max_edges=args.max_eval_edges,
                seed=args.seed + epoch,
            )
            test_mrr = _evaluate_mrr(
                z,
                test_edges,
                num_neg=args.num_neg_eval,
                dst_offset=dst_offset,
                num_dst=num_dst,
                device=device,
                max_edges=args.max_eval_edges,
                seed=args.seed + 10_000 + epoch,
            )

        avg_loss = epoch_loss / steps_per_epoch
        print(f"epoch={epoch} loss={avg_loss:.4f} val_mrr@{args.num_neg_eval}={val_mrr:.4f} test_mrr@{args.num_neg_eval}={test_mrr:.4f}")
        if run is not None:
            run.log({"epoch": epoch, "loss": avg_loss, "val_mrr": val_mrr, "test_mrr": test_mrr})

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
