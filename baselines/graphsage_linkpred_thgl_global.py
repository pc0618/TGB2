#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn

from baselines.graphsage_linkpred import CSRAdjacency, SampledGraphSAGE


@dataclass(frozen=True)
class TaskEdges:
    src_global: np.ndarray
    dst_global: np.ndarray
    ts_s: np.ndarray
    edge_type: np.ndarray


def _parse_node_type_from_table_name(name: str) -> int:
    m = re.fullmatch(r"nodes_type_(\d+)", str(name))
    if not m:
        raise ValueError(f"Unexpected node table name: {name}")
    return int(m.group(1))


def _read_task_fkeys(pq_path: Path) -> tuple[int, int]:
    md = pq.ParquetFile(pq_path).metadata.metadata or {}
    raw = md.get(b"fkey_col_to_pkey_table")
    if raw is None:
        raise RuntimeError(f"Missing fkey_col_to_pkey_table metadata on {pq_path}")
    fkeys = json.loads(raw.decode("utf-8"))
    return _parse_node_type_from_table_name(fkeys["src_id"]), _parse_node_type_from_table_name(fkeys["dst_id"])


def _load_globals_by_type(ds_dir: Path) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    node_type = np.load(ds_dir / "mappings" / "node_type.npy", mmap_mode="r").astype(np.int64, copy=False)
    globals_by_type: dict[int, np.ndarray] = {}
    for p in sorted((ds_dir / "mappings").glob("globals_type_*.npy")):
        m = re.fullmatch(r"globals_type_(\d+)\.npy", p.name)
        if not m:
            continue
        t = int(m.group(1))
        globals_by_type[t] = np.load(p, mmap_mode="r").astype(np.int64, copy=False)
    if not globals_by_type:
        raise RuntimeError(f"Missing globals_type_*.npy under {ds_dir/'mappings'}")
    return node_type, globals_by_type


def _load_negatives(ds_dir: Path, split: str) -> dict:
    p = ds_dir / "negatives" / f"{split}_ns.pkl"
    with p.open("rb") as f:
        return pickle.load(f)


def _load_task_edges(
    *,
    ds_dir: Path,
    split: str,
    max_edges: int,
    seed: int,
) -> TaskEdges:
    rng = np.random.default_rng(int(seed))
    task_root = ds_dir / "tasks"
    task_dirs = sorted(task_root.glob("edge-type-*-mrr"))
    if not task_dirs:
        raise FileNotFoundError(f"No thgl task dirs found under {task_root}")

    node_type, globals_by_type = _load_globals_by_type(ds_dir)
    num_nodes = int(node_type.shape[0])

    src_all: list[np.ndarray] = []
    dst_all: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []
    et_all: list[np.ndarray] = []

    remaining = int(max_edges) if int(max_edges) > 0 else None
    for td in task_dirs:
        m = re.fullmatch(r"edge-type-(\d+)-mrr", td.name)
        if not m:
            continue
        et = int(m.group(1))
        pq_path = td / f"{split}.parquet"
        if not pq_path.exists():
            continue

        src_t, dst_t = _read_task_fkeys(pq_path)
        df = pd.read_parquet(pq_path, columns=["src_id", "dst_id", "event_ts"])
        if df.shape[0] == 0:
            continue
        if remaining is not None and df.shape[0] > remaining:
            idx = rng.choice(df.shape[0], size=remaining, replace=False)
            df = df.iloc[idx]

        ts_s = (pd.to_datetime(df["event_ts"], utc=True).astype("int64").to_numpy(copy=False) // 1_000_000_000).astype(
            np.int64, copy=False
        )
        src_local = df["src_id"].to_numpy(dtype=np.int64, copy=False)
        dst_local = df["dst_id"].to_numpy(dtype=np.int64, copy=False)
        src_g = globals_by_type[src_t][src_local]
        dst_g = globals_by_type[dst_t][dst_local]

        # Basic safety: global ids are in-range.
        if src_g.max(initial=0) >= num_nodes or dst_g.max(initial=0) >= num_nodes:
            raise RuntimeError("Global id mapping produced out-of-range ids.")

        k = int(df.shape[0])
        src_all.append(src_g.astype(np.int64, copy=False))
        dst_all.append(dst_g.astype(np.int64, copy=False))
        ts_all.append(ts_s)
        et_all.append(np.full(k, et, dtype=np.int64))

        if remaining is not None:
            remaining -= k
            if remaining <= 0:
                break

    if not src_all:
        return TaskEdges(
            src_global=np.zeros((0,), dtype=np.int64),
            dst_global=np.zeros((0,), dtype=np.int64),
            ts_s=np.zeros((0,), dtype=np.int64),
            edge_type=np.zeros((0,), dtype=np.int64),
        )

    return TaskEdges(
        src_global=np.concatenate(src_all, axis=0),
        dst_global=np.concatenate(dst_all, axis=0),
        ts_s=np.concatenate(ts_all, axis=0),
        edge_type=np.concatenate(et_all, axis=0),
    )


def _build_csr(*, num_nodes: int, src: np.ndarray, dst: np.ndarray, undirected: bool) -> CSRAdjacency:
    src = src.astype(np.int64, copy=False)
    dst = dst.astype(np.int64, copy=False)
    if undirected:
        src_all = np.concatenate([src, dst], axis=0)
        dst_all = np.concatenate([dst, src], axis=0)
    else:
        src_all = src
        dst_all = dst

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
    return CSRAdjacency(indptr=indptr, indices=indices)


@torch.no_grad()
def _eval_official(
    *,
    model: SampledGraphSAGE,
    edge_emb: nn.Embedding,
    adj: CSRAdjacency,
    edges: TaskEdges,
    neg_dict: dict,
    device: torch.device,
    seed: int,
    k_value: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    n = int(edges.src_global.shape[0])
    if n == 0:
        return float("nan"), float("nan")

    src = edges.src_global
    dst = edges.dst_global
    ts_s = edges.ts_s
    et = edges.edge_type

    # Build ragged negatives.
    neg_lists: list[np.ndarray] = []
    lens = np.empty((n,), dtype=np.int64)
    for i in range(n):
        key = (int(ts_s[i]), int(src[i]), int(et[i]))
        neg = np.asarray(neg_dict[key], dtype=np.int64)
        neg_lists.append(neg)
        lens[i] = int(neg.shape[0])
    neg_ptr = np.zeros((n + 1,), dtype=np.int64)
    np.cumsum(lens, out=neg_ptr[1:])
    neg_flat = np.concatenate(neg_lists, axis=0) if neg_lists else np.zeros((0,), dtype=np.int64)
    neg_row = np.repeat(np.arange(n, dtype=np.int64), lens.astype(np.int64, copy=False))

    src_t = torch.from_numpy(src).to(device=device, dtype=torch.long)
    dst_t = torch.from_numpy(dst).to(device=device, dtype=torch.long)
    et_t = torch.from_numpy(et).to(device=device, dtype=torch.long)
    neg_flat_t = torch.from_numpy(neg_flat).to(device=device, dtype=torch.long)
    neg_row_t = torch.from_numpy(neg_row).to(device=device, dtype=torch.long)

    z_src = model.encode(src_t, adj, rng=rng, device=device)
    z_dst = model.encode(dst_t, adj, rng=rng, device=device)
    rel_h = edge_emb(et_t)
    pos_score = ((z_src + rel_h) * z_dst).sum(dim=1)  # [N]

    z_neg = model.encode(neg_flat_t, adj, rng=rng, device=device)  # [sumK, H]
    neg_score_flat = ((z_src[neg_row_t] + rel_h[neg_row_t]) * z_neg).sum(dim=1)  # [sumK]

    optimistic = torch.zeros((n,), device=device, dtype=torch.float32)
    pessimistic = torch.zeros((n,), device=device, dtype=torch.float32)
    for i in range(n):
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
    parser = argparse.ArgumentParser(description="Global GraphSAGE baseline for thgl-* tasks using official one-vs-many negatives.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--relbench_cache_root", default="/home/pc0618/tmp_relbench_cache_official")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--fanouts", default="15,10")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_neg_train", type=int, default=1)
    parser.add_argument("--max_train_edges", type=int, default=50000)
    parser.add_argument("--max_eval_edges", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--undirected", action="store_true")
    args = parser.parse_args()

    ds_dir = Path(args.relbench_cache_root) / f"rel-tgb-{args.dataset}"
    node_type, globals_by_type = _load_globals_by_type(ds_dir)
    num_nodes = int(node_type.shape[0])

    train_edges = _load_task_edges(ds_dir=ds_dir, split="train", max_edges=int(args.max_train_edges), seed=int(args.seed) + 10)
    val_edges = _load_task_edges(ds_dir=ds_dir, split="val", max_edges=int(args.max_eval_edges), seed=int(args.seed) + 20)
    test_edges = _load_task_edges(ds_dir=ds_dir, split="test", max_edges=int(args.max_eval_edges), seed=int(args.seed) + 30)

    if train_edges.src_global.size == 0:
        raise RuntimeError("No training edges loaded from tasks.")

    adj = _build_csr(num_nodes=num_nodes, src=train_edges.src_global, dst=train_edges.dst_global, undirected=bool(args.undirected))
    fan1, fan2 = (int(x) for x in str(args.fanouts).split(","))
    device = torch.device(args.device)

    model = SampledGraphSAGE(
        num_nodes=int(num_nodes),
        emb_dim=int(args.emb_dim),
        hidden_dim=int(args.hidden_dim),
        fanouts=(fan1, fan2),
        dropout=float(args.dropout),
    ).to(device)
    edge_emb = nn.Embedding(int(train_edges.edge_type.max(initial=0) + 1), int(args.hidden_dim)).to(device)
    nn.init.normal_(edge_emb.weight, std=0.02)

    opt = torch.optim.AdamW(
        list(model.parameters()) + list(edge_emb.parameters()),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    bce = nn.BCEWithLogitsLoss()

    # Precompute global ids by destination type for negative sampling.
    nodes_by_type: dict[int, np.ndarray] = {}
    for t in np.unique(node_type).tolist():
        t_int = int(t)
        nodes_by_type[t_int] = np.flatnonzero(node_type == t_int).astype(np.int64, copy=False)

    # Also keep per-edge dst type for train edges (using node_type array over global ids).
    train_dst_type = node_type[train_edges.dst_global].astype(np.int64, copy=False)

    steps_per_epoch = max(1, int(np.ceil(train_edges.src_global.size / int(args.batch_size))))
    print(
        f"Dataset={args.dataset} num_nodes={num_nodes} train_edges={train_edges.src_global.size} "
        f"val_edges={val_edges.src_global.size} test_edges={test_edges.src_global.size} undirected={bool(args.undirected)}"
    )

    val_neg = _load_negatives(ds_dir, "val")
    test_neg = _load_negatives(ds_dir, "test")

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        edge_emb.train()
        rng = np.random.default_rng(int(args.seed) + 1000 + epoch)
        perm = rng.permutation(train_edges.src_global.size)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            sel = perm[step * int(args.batch_size) : (step + 1) * int(args.batch_size)]
            if sel.size == 0:
                continue

            src = train_edges.src_global[sel]
            dst = train_edges.dst_global[sel]
            et = train_edges.edge_type[sel]
            dst_t = train_dst_type[sel]

            src_t = torch.from_numpy(src).to(device=device, dtype=torch.long)
            dst_torch = torch.from_numpy(dst).to(device=device, dtype=torch.long)
            et_t = torch.from_numpy(et).to(device=device, dtype=torch.long)

            z_src = model.encode(src_t, adj, rng=rng, device=device)
            z_dst = model.encode(dst_torch, adj, rng=rng, device=device)
            rel_h = edge_emb(et_t)
            pos_logit = ((z_src + rel_h) * z_dst).sum(dim=1)

            # Sample type-correct negatives.
            neg = np.empty((src.shape[0] * int(args.num_neg_train),), dtype=np.int64)
            for t in np.unique(dst_t).tolist():
                t_int = int(t)
                mask = np.flatnonzero(dst_t == t_int)
                if mask.size == 0:
                    continue
                cand = nodes_by_type[t_int]
                neg_pick = rng.choice(cand, size=mask.size * int(args.num_neg_train), replace=True)
                neg[np.repeat(mask, int(args.num_neg_train))] = neg_pick

            neg_t = torch.from_numpy(neg).to(device=device, dtype=torch.long)
            src_rep = src_t.repeat_interleave(int(args.num_neg_train))
            et_rep = et_t.repeat_interleave(int(args.num_neg_train))
            z_src_n = model.encode(src_rep, adj, rng=rng, device=device)
            z_neg = model.encode(neg_t, adj, rng=rng, device=device)
            rel_h_n = edge_emb(et_rep)
            neg_logit = ((z_src_n + rel_h_n) * z_neg).sum(dim=1)

            loss = bce(pos_logit, torch.ones_like(pos_logit)) + bce(neg_logit, torch.zeros_like(neg_logit))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / float(steps_per_epoch)
        model.eval()
        edge_emb.eval()
        val_mrr, val_hits = _eval_official(
            model=model,
            edge_emb=edge_emb,
            adj=adj,
            edges=val_edges,
            neg_dict=val_neg,
            device=device,
            seed=int(args.seed) + 2000 + epoch,
            k_value=10,
        )
        test_mrr, test_hits = _eval_official(
            model=model,
            edge_emb=edge_emb,
            adj=adj,
            edges=test_edges,
            neg_dict=test_neg,
            device=device,
            seed=int(args.seed) + 3000 + epoch,
            k_value=10,
        )
        print(
            f"epoch={epoch} loss={avg_loss:.4f} val_mrr={val_mrr:.4f} val_hits@10={val_hits:.4f} "
            f"test_mrr={test_mrr:.4f} test_hits@10={test_hits:.4f}"
        )


if __name__ == "__main__":
    main()

