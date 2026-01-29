#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import GRUCell, Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.seed import seed_everything
from torch_geometric.utils import scatter
from tqdm import tqdm


@dataclass(frozen=True)
class ExportMeta:
    dataset: str
    val_timestamp_s: int
    test_timestamp_s: int


@dataclass(frozen=True)
class EventTableInfo:
    name: str
    path: Path
    src_table: str
    dst_table: str
    src_off: int
    dst_off: int
    dst_min: int
    dst_max_exclusive: int
    total_rows: int


def _load_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    return ExportMeta(
        dataset=str(meta["dataset"]),
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
    )


def _table_num_rows(db_dir: Path, table: str) -> int:
    return int(pq.ParquetFile(db_dir / f"{table}.parquet").metadata.num_rows)


def _read_table_metadata(path: Path) -> tuple[dict[str, str], Optional[str]]:
    pf = pq.ParquetFile(path)
    md = pf.schema_arrow.metadata or {}
    fkeys = json.loads(md[b"fkey_col_to_pkey_table"].decode("utf-8")) if b"fkey_col_to_pkey_table" in md else {}
    time_col = json.loads(md[b"time_col"].decode("utf-8")) if b"time_col" in md else None
    return fkeys, time_col


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _iter_events(path: Path, batch_size: int):
    pf = pq.ParquetFile(path)
    yield from pf.iter_batches(batch_size=batch_size, columns=["src_id", "dst_id", "event_ts"])


def _load_edges_by_time(
    events_path: Path,
    *,
    start_s_exclusive: Optional[int],
    end_s_inclusive: Optional[int],
    parquet_batch_size: int,
    max_events: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    pf = pq.ParquetFile(events_path)
    total = int(pf.metadata.num_rows)
    p = 1.0
    if max_events is not None and int(max_events) > 0:
        p = min(1.0, float(int(max_events)) / float(max(total, 1)))
    rng = np.random.default_rng(int(seed))

    cols = pf.schema.names
    time_idx = cols.index("event_ts") if "event_ts" in cols else None

    def _stat_to_s(x) -> Optional[int]:
        if x is None:
            return None
        if isinstance(x, (int, np.integer)):
            return int(x)
        try:
            return int(pd.to_datetime(x, utc=True).timestamp())
        except Exception:
            return None

    edges: list[np.ndarray] = []
    ts_all: list[np.ndarray] = []
    for rg in range(pf.num_row_groups):
        if time_idx is not None:
            try:
                col_meta = pf.metadata.row_group(rg).column(time_idx)
                stats = col_meta.statistics
                min_s = _stat_to_s(getattr(stats, "min", None)) if stats is not None else None
                max_s = _stat_to_s(getattr(stats, "max", None)) if stats is not None else None
            except Exception:
                min_s = max_s = None
        else:
            min_s = max_s = None

        if start_s_exclusive is not None and max_s is not None and max_s <= int(start_s_exclusive):
            continue
        if end_s_inclusive is not None and min_s is not None and min_s > int(end_s_inclusive):
            break

        tbl = pf.read_row_group(rg, columns=["src_id", "dst_id", "event_ts"], use_threads=True)
        src = tbl.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = tbl.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(tbl.column(2))

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
        ts_all.append(ts_s[mask])

    out_edges = np.concatenate(edges, axis=0) if edges else np.zeros((0, 2), dtype=np.int64)
    out_ts = np.concatenate(ts_all, axis=0) if ts_all else np.zeros((0,), dtype=np.int64)
    if max_events is not None and int(max_events) > 0 and out_edges.shape[0] > int(max_events):
        idx = rng.choice(out_edges.shape[0], size=int(max_events), replace=False)
        out_edges = out_edges[idx]
        out_ts = out_ts[idx]

    order = np.argsort(out_ts, kind="mergesort")
    return out_edges[order], out_ts[order]


def _infer_event_tables(db_dir: Path, requested: Optional[str]) -> list[str]:
    if requested is not None:
        return [str(requested)]
    if (db_dir / "events.parquet").exists():
        return ["events"]
    event_paths = sorted(db_dir.glob("events_edge_type_*.parquet"), key=lambda p: p.name)
    if event_paths:
        return [p.stem for p in event_paths]
    raise SystemExit("Could not infer event_table(s); pass --event_table (e.g. events or events_edge_type_0).")


def _type_id(name: str) -> int:
    if not name.startswith("nodes_type_"):
        raise ValueError(f"Expected nodes_type_* table name, got {name!r}")
    return int(name.split("_")[-1])


def _build_node_offsets(db_dir: Path, *, referenced_tables: set[str]) -> tuple[dict[str, int], int]:
    if referenced_tables == {"src_nodes", "dst_nodes"}:
        n_src = _table_num_rows(db_dir, "src_nodes")
        offsets = {"src_nodes": 0, "dst_nodes": int(n_src)}
        return offsets, int(n_src + _table_num_rows(db_dir, "dst_nodes"))

    if len(referenced_tables) == 1:
        tbl = next(iter(referenced_tables))
        offsets = {tbl: 0}
        return offsets, int(_table_num_rows(db_dir, tbl))

    node_tables = [t for t in referenced_tables if t.startswith("nodes_type_")]
    if not node_tables:
        raise SystemExit(f"Could not map heterogeneous node tables from: {sorted(referenced_tables)}")
    node_tables = sorted(set(node_tables), key=_type_id)
    offsets: dict[str, int] = {}
    cur = 0
    for t in node_tables:
        offsets[t] = int(cur)
        cur += _table_num_rows(db_dir, t)
    return offsets, int(cur)


def _build_event_table_infos(db_dir: Path, *, event_tables: list[str]) -> tuple[list[EventTableInfo], int]:
    # Collect (src_table, dst_table) from parquet metadata and build a global node-id mapping.
    src_dst: list[tuple[str, str, Path, int]] = []
    referenced_tables: set[str] = set()
    for name in event_tables:
        path = db_dir / f"{name}.parquet"
        if not path.exists():
            raise SystemExit(f"Missing events parquet: {path}")
        fkeys, time_col = _read_table_metadata(path)
        if time_col != "event_ts":
            raise SystemExit(f"{name}: Expected time_col=event_ts, got {time_col!r}")
        src_table = fkeys.get("src_id")
        dst_table = fkeys.get("dst_id")
        if src_table is None or dst_table is None:
            raise SystemExit(f"{name}: Could not infer src/dst tables from parquet metadata: {fkeys}")
        pf = pq.ParquetFile(path)
        total_rows = int(pf.metadata.num_rows)
        src_dst.append((src_table, dst_table, path, total_rows))
        referenced_tables.add(src_table)
        referenced_tables.add(dst_table)

    offsets, num_nodes = _build_node_offsets(db_dir, referenced_tables=referenced_tables)

    infos: list[EventTableInfo] = []
    for name, (src_table, dst_table, path, total_rows) in zip(event_tables, src_dst):
        if src_table not in offsets or dst_table not in offsets:
            raise SystemExit(f"{name}: Missing node-table offsets for src={src_table!r} dst={dst_table!r}. Offsets={offsets}")
        dst_min = int(offsets[dst_table])
        dst_max_exclusive = int(dst_min + _table_num_rows(db_dir, dst_table))
        infos.append(
            EventTableInfo(
                name=name,
                path=path,
                src_table=src_table,
                dst_table=dst_table,
                src_off=int(offsets[src_table]),
                dst_off=int(offsets[dst_table]),
                dst_min=dst_min,
                dst_max_exclusive=dst_max_exclusive,
                total_rows=int(total_rows),
            )
        )
    return infos, int(num_nodes)


def _load_edges_by_time_multi(
    event_infos: list[EventTableInfo],
    *,
    start_s_exclusive: Optional[int],
    end_s_inclusive: Optional[int],
    parquet_batch_size: int,
    max_events: Optional[int],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not event_infos:
        return (
            np.zeros((0, 2), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )

    total_rows = int(sum(int(i.total_rows) for i in event_infos))
    if max_events is None or int(max_events) <= 0 or total_rows <= 0:
        p = 1.0
    else:
        p = min(1.0, float(int(max_events)) / float(total_rows))

    all_edges: list[np.ndarray] = []
    all_ts: list[np.ndarray] = []
    all_dst_min: list[np.ndarray] = []
    all_dst_max: list[np.ndarray] = []

    rng = np.random.default_rng(int(seed))
    for idx, info in enumerate(event_infos):
        per_table_max: Optional[int]
        if p >= 1.0:
            per_table_max = None if (max_events is None or int(max_events) <= 0) else int(max_events)
        else:
            per_table_max = max(1, int(math.ceil(p * float(int(info.total_rows)))))

        edges, ts_s = _load_edges_by_time(
            info.path,
            start_s_exclusive=start_s_exclusive,
            end_s_inclusive=end_s_inclusive,
            parquet_batch_size=parquet_batch_size,
            max_events=per_table_max,
            seed=int(seed) + 17 * (idx + 1),
        )
        if edges.size == 0:
            continue

        edges = edges.astype(np.int64, copy=False)
        edges = edges.copy()
        edges[:, 0] += int(info.src_off)
        edges[:, 1] += int(info.dst_off)

        all_edges.append(edges)
        all_ts.append(ts_s.astype(np.int64, copy=False))
        all_dst_min.append(np.full((edges.shape[0],), int(info.dst_min), dtype=np.int64))
        all_dst_max.append(np.full((edges.shape[0],), int(info.dst_max_exclusive), dtype=np.int64))

    edges_all = np.concatenate(all_edges, axis=0) if all_edges else np.zeros((0, 2), dtype=np.int64)
    ts_all = np.concatenate(all_ts, axis=0) if all_ts else np.zeros((0,), dtype=np.int64)
    dst_min_all = np.concatenate(all_dst_min, axis=0) if all_dst_min else np.zeros((0,), dtype=np.int64)
    dst_max_all = np.concatenate(all_dst_max, axis=0) if all_dst_max else np.zeros((0,), dtype=np.int64)

    if max_events is not None and int(max_events) > 0 and edges_all.shape[0] > int(max_events):
        pick = rng.choice(edges_all.shape[0], size=int(max_events), replace=False)
        edges_all = edges_all[pick]
        ts_all = ts_all[pick]
        dst_min_all = dst_min_all[pick]
        dst_max_all = dst_max_all[pick]

    order = np.argsort(ts_all, kind="mergesort")
    return edges_all[order], ts_all[order], dst_min_all[order], dst_max_all[order]


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = int(out_channels)
        self.lin = Linear(1, self.out_channels)

    def forward(self, t: Tensor) -> Tensor:
        return self.lin(t.view(-1, 1)).cos()


class IdentityMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = int(raw_msg_dim + 2 * memory_dim + time_dim)

    def forward(self, z_src: Tensor, z_dst: Tensor, raw_msg: Tensor, t_enc: Tensor) -> Tensor:
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)


class LastAggregator(torch.nn.Module):
    def forward(self, msg: Tensor, index: Tensor, t: Tensor, dim_size: int) -> Tensor:
        if msg.size(0) == 0:
            return msg.new_zeros((dim_size, msg.size(-1)))
        max_t = scatter(t, index, dim=0, dim_size=dim_size, reduce="max")
        is_max = t == max_t[index]
        pos = torch.arange(t.size(0), device=t.device, dtype=torch.long)
        pos = torch.where(is_max, pos, pos.new_full(pos.shape, -1))
        argmax = scatter(pos, index, dim=0, dim_size=dim_size, reduce="max")

        out = msg.new_zeros((dim_size, msg.size(-1)))
        mask = argmax >= 0
        out[mask] = msg[argmax[mask]]
        return out


class LastNeighborLoader:
    """Keeps the last K neighbors per node (undirected)."""

    def __init__(self, num_nodes: int, size: int, *, device: Optional[torch.device] = None):
        self.size = int(size)
        self.neighbors = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self.e_id = torch.empty((num_nodes, self.size), dtype=torch.long, device=device)
        self._assoc = torch.empty(num_nodes, dtype=torch.long, device=device)
        self.reset_state()

    def reset_state(self) -> None:
        self.cur_e_id = 0
        self.e_id.fill_(-1)

    def __call__(self, n_id: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1, 1).repeat(1, self.size)
        e_id = self.e_id[n_id]

        mask = e_id >= 0
        neighbors, nodes, e_id = neighbors[mask], nodes[mask], e_id[mask]

        n_id = torch.cat([n_id, neighbors]).unique()
        self._assoc[n_id] = torch.arange(n_id.size(0), device=n_id.device)
        neighbors, nodes = self._assoc[neighbors], self._assoc[nodes]
        return n_id, torch.stack([neighbors, nodes]), e_id

    def insert(self, src: Tensor, dst: Tensor) -> None:
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0), device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self._assoc[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.size
        dense_id += self._assoc[nodes].mul_(self.size)

        dense_e_id = e_id.new_full((n_id.numel() * self.size,), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.size)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.size)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.size)

        e_id = torch.cat([self.e_id[n_id, : self.size], dense_e_id], dim=-1)
        neighbors = torch.cat([self.neighbors[n_id, : self.size], dense_neighbors], dim=-1)

        e_id, perm = e_id.topk(self.size, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)


class TGNMemory(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        raw_msg_dim: int,
        memory_dim: int,
        time_dim: int,
        *,
        message_module: torch.nn.Module,
        aggregator_module: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.num_nodes = int(num_nodes)
        self.raw_msg_dim = int(raw_msg_dim)
        self.memory_dim = int(memory_dim)
        self.time_dim = int(time_dim)

        self.msg_module = message_module
        self.aggr_module = aggregator_module
        self.time_enc = TimeEncoder(time_dim)
        self.memory_updater = GRUCell(self.msg_module.out_channels, memory_dim)

        self.register_buffer("memory", torch.zeros(num_nodes, memory_dim))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))

    def reset_state(self) -> None:
        self.memory.zero_()
        self.last_update.zero_()

    def detach(self) -> None:
        self.memory.detach_()

    def forward(self, n_id: Tensor) -> tuple[Tensor, Tensor]:
        return self.memory[n_id], self.last_update[n_id]

    def update_state(self, src: Tensor, dst: Tensor, t: Tensor, raw_msg: Tensor) -> None:
        if src.numel() == 0:
            return

        t_rel_s = t - self.last_update[src]
        t_enc_s = self.time_enc(t_rel_s.to(torch.float32))
        msg_s = self.msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc_s)
        self.memory[src] = self.memory_updater(msg_s, self.memory[src])
        self.last_update[src] = torch.maximum(self.last_update[src], t)

        t_rel_d = t - self.last_update[dst]
        t_enc_d = self.time_enc(t_rel_d.to(torch.float32))
        msg_d = self.msg_module(self.memory[dst], self.memory[src], raw_msg, t_enc_d)
        self.memory[dst] = self.memory_updater(msg_d, self.memory[dst])
        self.last_update[dst] = torch.maximum(self.last_update[dst], t)


class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: TimeEncoder):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = int(msg_dim + time_enc.out_channels)
        if out_channels % 2 != 0:
            raise ValueError("out_channels must be divisible by 2 (TransformerConv heads=2).")
        self.conv = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(self, x: Tensor, last_update: Tensor, edge_index: Tensor, t: Tensor, msg: Tensor) -> Tensor:
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


def _dedup_params(*modules: torch.nn.Module) -> list[torch.nn.Parameter]:
    seen: set[int] = set()
    out: list[torch.nn.Parameter] = []
    for m in modules:
        for p in m.parameters():
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
    return out


@torch.no_grad()
def _evaluate_mrr(
    *,
    memory: TGNMemory,
    gnn: GraphAttentionEmbedding,
    neighbor_loader: LastNeighborLoader,
    hist_t: Tensor,
    hist_msg: Tensor,
    edges_src: np.ndarray,
    edges_dst: np.ndarray,
    edges_dst_min: np.ndarray,
    edges_dst_max_exclusive: np.ndarray,
    num_neg: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    assoc: Tensor,
) -> float:
    if edges_src.size == 0:
        return float("nan")
    rng = np.random.default_rng(int(seed))

    mrr_sum = 0.0
    n_total = 0
    for start in range(0, edges_src.shape[0], int(batch_size)):
        end = min(start + int(batch_size), edges_src.shape[0])
        src_np = edges_src[start:end].astype(np.int64, copy=False)
        dst_np = edges_dst[start:end].astype(np.int64, copy=False)
        dst_min_np = edges_dst_min[start:end].astype(np.int64, copy=False)
        dst_max_np = edges_dst_max_exclusive[start:end].astype(np.int64, copy=False)

        src = torch.from_numpy(src_np).to(device=device, dtype=torch.long)
        pos_dst = torch.from_numpy(dst_np).to(device=device, dtype=torch.long)

        if dst_min_np.size > 0 and int(dst_min_np.min()) == int(dst_min_np.max()) and int(dst_max_np.min()) == int(dst_max_np.max()):
            neg_np = rng.integers(int(dst_min_np[0]), int(dst_max_np[0]), size=(src_np.shape[0], int(num_neg)), dtype=np.int64)
        else:
            neg_np = np.empty((src_np.shape[0], int(num_neg)), dtype=np.int64)
            for i in range(src_np.shape[0]):
                lo = int(dst_min_np[i])
                hi = int(dst_max_np[i])
                neg_np[i] = rng.integers(lo, hi, size=(int(num_neg),), dtype=np.int64)
        neg_dst = torch.from_numpy(neg_np).to(device=device, dtype=torch.long)

        n_id = torch.cat([src, pos_dst, neg_dst.reshape(-1)]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, hist_t[e_id], hist_msg[e_id])

        src_z = z[assoc[src]]
        pos_z = z[assoc[pos_dst]]
        neg_z = z[assoc[neg_dst]]

        pos_score = (src_z * pos_z).sum(dim=-1, keepdim=True)  # [B,1]
        neg_score = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)  # [B, num_neg]

        rank = 1 + (neg_score >= pos_score).sum(dim=1)  # pessimistic ties
        mrr_sum += float((1.0 / rank.float()).sum().item())
        n_total += int(rank.numel())

    return mrr_sum / max(n_total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="TGN + TransformerConv attention on RelBench-exported TGB datasets (sampled-negative MRR).")
    parser.add_argument("--dataset", required=True, help="Export dataset id under exports_root, e.g. tgbl-wiki-v2")
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--event_table", default=None, help="Event table name under db/ (default: infer; thgl uses events_edge_type_*)")
    parser.add_argument("--adj", default="val", help="History cutoff used to build memory: val | test | all | <unix_seconds>")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=96, help="Train batch size (events per step).")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Eval batch size (edges per step).")
    parser.add_argument("--num_neighbors", type=int, default=10)
    parser.add_argument("--mem_dim", type=int, default=64)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_neg_train", type=int, default=10)
    parser.add_argument("--num_neg_eval", type=int, default=100)

    parser.add_argument("--max_train_events", type=int, default=200000)
    parser.add_argument("--max_val_events", type=int, default=20000)
    parser.add_argument("--max_test_events", type=int, default=20000)
    parser.add_argument("--parquet_batch_size", type=int, default=500000)
    args = parser.parse_args()

    seed_everything(int(args.seed))

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    db_dir = exports_root / args.dataset / "db"

    if args.adj == "val":
        hist_cut_s = meta.val_timestamp_s
    elif args.adj == "test":
        hist_cut_s = meta.test_timestamp_s
    elif args.adj == "all":
        hist_cut_s = None
    else:
        hist_cut_s = int(args.adj)

    event_tables = _infer_event_tables(db_dir, str(args.event_table) if args.event_table is not None else None)
    event_infos, num_nodes = _build_event_table_infos(db_dir, event_tables=event_tables)
    print(f"[dataset] {args.dataset} (event_tables={len(event_tables)})")
    for info in event_infos[:5]:
        print(f"[events] {info.name}: {info.src_table}->{info.dst_table} dst_range=[{info.dst_min:,},{info.dst_max_exclusive:,}) rows={info.total_rows:,}")
    if len(event_infos) > 5:
        print(f"[events] ... +{len(event_infos) - 5} more")
    print(f"[num_nodes] {num_nodes:,}")

    # Training stream: events with timestamp <= history cutoff (or all if adj=all).
    train_edges, train_ts_s, train_dst_min, train_dst_max = _load_edges_by_time_multi(
        event_infos,
        start_s_exclusive=None,
        end_s_inclusive=hist_cut_s,
        parquet_batch_size=int(args.parquet_batch_size),
        max_events=int(args.max_train_events),
        seed=int(args.seed) + 10,
    )

    # Eval sets are always defined by the dataset cutoffs.
    val_edges, _val_ts_s, val_dst_min, val_dst_max = _load_edges_by_time_multi(
        event_infos,
        start_s_exclusive=meta.val_timestamp_s,
        end_s_inclusive=meta.test_timestamp_s,
        parquet_batch_size=int(args.parquet_batch_size),
        max_events=int(args.max_val_events),
        seed=int(args.seed) + 20,
    )
    test_edges, _test_ts_s, test_dst_min, test_dst_max = _load_edges_by_time_multi(
        event_infos,
        start_s_exclusive=meta.test_timestamp_s,
        end_s_inclusive=None,
        parquet_batch_size=int(args.parquet_batch_size),
        max_events=int(args.max_test_events),
        seed=int(args.seed) + 30,
    )

    device = torch.device(args.device)
    memory = TGNMemory(
        num_nodes=num_nodes,
        raw_msg_dim=0,
        memory_dim=int(args.mem_dim),
        time_dim=int(args.time_dim),
        message_module=IdentityMessage(0, int(args.mem_dim), int(args.time_dim)),
        aggregator_module=LastAggregator(),
    ).to(device)
    gnn = GraphAttentionEmbedding(
        in_channels=int(args.mem_dim),
        out_channels=int(args.emb_dim),
        msg_dim=0,
        time_enc=memory.time_enc,
    ).to(device)
    neighbor_loader = LastNeighborLoader(num_nodes, size=int(args.num_neighbors), device=device)
    optimizer = torch.optim.Adam(_dedup_params(memory, gnn), lr=float(args.lr))
    assoc = torch.empty(num_nodes, device=device, dtype=torch.long)

    src_tr = torch.from_numpy(train_edges[:, 0].astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    dst_tr = torch.from_numpy(train_edges[:, 1].astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    t_tr = torch.from_numpy(train_ts_s.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    msg_tr = torch.zeros((t_tr.numel(), 0), device=device, dtype=torch.float32)
    dst_min_tr = torch.from_numpy(train_dst_min.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)
    dst_max_tr = torch.from_numpy(train_dst_max.astype(np.int64, copy=False)).to(device=device, dtype=torch.long)

    def train_epoch() -> float:
        memory.train()
        gnn.train()
        memory.reset_state()
        neighbor_loader.reset_state()

        total_loss = 0.0
        total_events = 0
        for start in tqdm(range(0, t_tr.numel(), int(args.batch_size)), desc="train", leave=False):
            end = min(start + int(args.batch_size), t_tr.numel())
            src_b = src_tr[start:end]
            pos_dst_b = dst_tr[start:end]
            t_b = t_tr[start:end]
            msg_b = msg_tr[start:end]
            dst_min_b = dst_min_tr[start:end]
            dst_max_b = dst_max_tr[start:end]
            if dst_min_b.numel() > 0 and int(dst_min_b.min()) == int(dst_min_b.max()) and int(dst_max_b.min()) == int(dst_max_b.max()):
                neg_dst_b = torch.randint(
                    int(dst_min_b[0]),
                    int(dst_max_b[0]),
                    (pos_dst_b.size(0), int(args.num_neg_train)),
                    device=device,
                )
            else:
                neg_dst_b = torch.empty((pos_dst_b.size(0), int(args.num_neg_train)), device=device, dtype=torch.long)
                for i in range(pos_dst_b.size(0)):
                    neg_dst_b[i] = torch.randint(
                        int(dst_min_b[i]),
                        int(dst_max_b[i]),
                        (int(args.num_neg_train),),
                        device=device,
                    )

            n_id = torch.cat([src_b, pos_dst_b, neg_dst_b.reshape(-1)]).unique()
            n_id, edge_index, e_id = neighbor_loader(n_id)
            assoc[n_id] = torch.arange(n_id.size(0), device=device)

            z, last_update = memory(n_id)
            z = gnn(z, last_update, edge_index, t_tr[e_id], msg_tr[e_id])

            src_z = z[assoc[src_b]]
            pos_z = z[assoc[pos_dst_b]]
            neg_z = z[assoc[neg_dst_b]]

            pos_score = (src_z * pos_z).sum(dim=-1)  # [B]
            neg_score = (src_z.unsqueeze(1) * neg_z).sum(dim=-1)  # [B, N]

            loss = F.softplus(-(pos_score.unsqueeze(1) - neg_score)).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            memory.update_state(src_b, pos_dst_b, t_b, msg_b)
            neighbor_loader.insert(src_b, pos_dst_b)
            memory.detach()

            total_loss += float(loss.detach()) * (end - start)
            total_events += (end - start)
        return total_loss / max(total_events, 1)

    best_val = -1.0
    best_test = float("nan")
    for epoch in range(1, int(args.epochs) + 1):
        loss = train_epoch()
        memory.eval()
        gnn.eval()
        val_mrr = _evaluate_mrr(
            memory=memory,
            gnn=gnn,
            neighbor_loader=neighbor_loader,
            hist_t=t_tr,
            hist_msg=msg_tr,
            edges_src=val_edges[:, 0] if val_edges.size else np.zeros((0,), dtype=np.int64),
            edges_dst=val_edges[:, 1] if val_edges.size else np.zeros((0,), dtype=np.int64),
            edges_dst_min=val_dst_min if val_dst_min.size else np.zeros((0,), dtype=np.int64),
            edges_dst_max_exclusive=val_dst_max if val_dst_max.size else np.zeros((0,), dtype=np.int64),
            num_neg=int(args.num_neg_eval),
            device=device,
            batch_size=int(args.eval_batch_size),
            seed=int(args.seed) + 2000 + epoch,
            assoc=assoc,
        )
        test_mrr = _evaluate_mrr(
            memory=memory,
            gnn=gnn,
            neighbor_loader=neighbor_loader,
            hist_t=t_tr,
            hist_msg=msg_tr,
            edges_src=test_edges[:, 0] if test_edges.size else np.zeros((0,), dtype=np.int64),
            edges_dst=test_edges[:, 1] if test_edges.size else np.zeros((0,), dtype=np.int64),
            edges_dst_min=test_dst_min if test_dst_min.size else np.zeros((0,), dtype=np.int64),
            edges_dst_max_exclusive=test_dst_max if test_dst_max.size else np.zeros((0,), dtype=np.int64),
            num_neg=int(args.num_neg_eval),
            device=device,
            batch_size=int(args.eval_batch_size),
            seed=int(args.seed) + 3000 + epoch,
            assoc=assoc,
        )
        print(f"epoch={epoch} loss={loss:.4f} val_mrr@{int(args.num_neg_eval)}={val_mrr:.4f} test_mrr@{int(args.num_neg_eval)}={test_mrr:.4f}")
        if not math.isnan(val_mrr) and val_mrr >= best_val:
            best_val = float(val_mrr)
            best_test = float(test_mrr)

    print(f"Best val_mrr@{int(args.num_neg_eval)}={best_val:.4f}  Best test_mrr@{int(args.num_neg_eval)}={best_test:.4f}")


if __name__ == "__main__":
    main()
