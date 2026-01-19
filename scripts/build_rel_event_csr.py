#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ExportMeta:
    dataset: str
    val_timestamp_s: int
    test_timestamp_s: int


def _load_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    return ExportMeta(
        dataset=meta["dataset"],
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
    )


def _iter_event_batches(events_path: Path, batch_size: int):
    pf = pq.ParquetFile(events_path)
    yield from pf.iter_batches(
        batch_size=batch_size, columns=["src_id", "dst_id", "event_ts", "weight"]
    )


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _make_memmap_npy(path: Path, dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def _detect_bipartite(db_dir: Path) -> tuple[bool, int, int]:
    src_path = db_dir / "src_nodes.parquet"
    dst_path = db_dir / "dst_nodes.parquet"
    if src_path.exists() and dst_path.exists():
        n_src = pq.ParquetFile(src_path).metadata.num_rows
        n_dst = pq.ParquetFile(dst_path).metadata.num_rows
        return True, int(n_src), int(n_dst)
    n = pq.ParquetFile(db_dir / "nodes.parquet").metadata.num_rows
    return False, int(n), 0


def build_rel_event_csr(
    exports_root: Path,
    dataset: str,
    *,
    upto_timestamp_s: Optional[int],
    batch_size: int,
    out_dir: Optional[Path],
) -> dict[str, Path]:
    db_dir = exports_root / dataset / "db"
    events_path = db_dir / "events.parquet"
    is_bipartite, n_src, n_dst = _detect_bipartite(db_dir)
    num_src_nodes = n_src
    num_dst_nodes = n_dst if is_bipartite else n_src

    suffix = f"upto_{upto_timestamp_s}" if upto_timestamp_s is not None else "all"
    if out_dir is None:
        out_dir = exports_root / dataset / "adj"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1: degrees and number of kept events.
    deg_src = np.zeros(num_src_nodes, dtype=np.int64)
    deg_dst = np.zeros(num_dst_nodes, dtype=np.int64)
    num_events = 0
    for batch in _iter_event_batches(events_path, batch_size=batch_size):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))
        if upto_timestamp_s is not None:
            mask = ts_s <= int(upto_timestamp_s)
            if not mask.any():
                continue
            src = src[mask]
            dst = dst[mask]
        num_events += int(src.shape[0])
        deg_src += np.bincount(src, minlength=num_src_nodes).astype(np.int64, copy=False)
        deg_dst += np.bincount(dst, minlength=num_dst_nodes).astype(np.int64, copy=False)

    # Allocate CSR structures.
    indptr_src = np.zeros(num_src_nodes + 1, dtype=np.int64)
    indptr_dst = np.zeros(num_dst_nodes + 1, dtype=np.int64)
    indptr_src[1:] = np.cumsum(deg_src, dtype=np.int64)
    indptr_dst[1:] = np.cumsum(deg_dst, dtype=np.int64)
    assert int(indptr_src[-1]) == num_events
    assert int(indptr_dst[-1]) == num_events

    # Memmap outputs.
    src_indptr_path = out_dir / f"csr_events_by_src_id_indptr_{suffix}.npy"
    src_indices_path = out_dir / f"csr_events_by_src_id_indices_{suffix}.npy"
    dst_indptr_path = out_dir / f"csr_events_by_dst_id_indptr_{suffix}.npy"
    dst_indices_path = out_dir / f"csr_events_by_dst_id_indices_{suffix}.npy"

    event_src_path = out_dir / f"events_src_id_{suffix}.npy"
    event_dst_path = out_dir / f"events_dst_id_{suffix}.npy"
    event_ts_path = out_dir / f"events_ts_s_{suffix}.npy"
    event_w_path = out_dir / f"events_weight_{suffix}.npy"

    src_indptr_mm = _make_memmap_npy(src_indptr_path, dtype=np.int64, shape=indptr_src.shape)
    dst_indptr_mm = _make_memmap_npy(dst_indptr_path, dtype=np.int64, shape=indptr_dst.shape)
    src_indptr_mm[:] = indptr_src
    dst_indptr_mm[:] = indptr_dst
    src_indptr_mm.flush()
    dst_indptr_mm.flush()

    src_indices_mm = _make_memmap_npy(src_indices_path, dtype=np.int64, shape=(num_events,))
    dst_indices_mm = _make_memmap_npy(dst_indices_path, dtype=np.int64, shape=(num_events,))

    event_src_mm = _make_memmap_npy(event_src_path, dtype=np.int64, shape=(num_events,))
    event_dst_mm = _make_memmap_npy(event_dst_path, dtype=np.int64, shape=(num_events,))
    event_ts_mm = _make_memmap_npy(event_ts_path, dtype=np.int64, shape=(num_events,))
    event_w_mm = _make_memmap_npy(event_w_path, dtype=np.float32, shape=(num_events,))

    # Pass 2: fill.
    cur_src = indptr_src.copy()
    cur_dst = indptr_dst.copy()
    event_idx = 0
    for batch in _iter_event_batches(events_path, batch_size=batch_size):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))
        w = batch.column(3).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        if upto_timestamp_s is not None:
            mask = ts_s <= int(upto_timestamp_s)
            if not mask.any():
                continue
            src = src[mask]
            dst = dst[mask]
            ts_s = ts_s[mask]
            w = w[mask]

        if src.size == 0:
            continue

        # Ensure deterministic event ordering (in file scan order).
        batch_len = int(src.shape[0])
        ev = np.arange(event_idx, event_idx + batch_len, dtype=np.int64)

        # Fill per-event arrays:
        event_src_mm[ev] = src
        event_dst_mm[ev] = dst
        event_ts_mm[ev] = ts_s
        event_w_mm[ev] = w

        # Fill CSR indices: node -> list(event_idx)
        # Vectorized scatter-fill by grouping on node id.
        for node_ids, cur, indices_mm in (
            (src, cur_src, src_indices_mm),
            (dst, cur_dst, dst_indices_mm),
        ):
            order = np.argsort(node_ids, kind="mergesort")
            node_s = node_ids[order]
            ev_s = ev[order]
            change = np.flatnonzero(node_s[1:] != node_s[:-1]) + 1
            group_starts = np.concatenate(
                [np.array([0], dtype=np.int64), change.astype(np.int64, copy=False)], axis=0
            )
            group_sizes = np.diff(
                np.concatenate([group_starts, np.array([node_s.size], dtype=np.int64)], axis=0)
            )
            group_nodes = node_s[group_starts]
            base = np.repeat(cur[group_nodes], group_sizes)
            start_rep = np.repeat(group_starts, group_sizes)
            offset = np.arange(node_s.size, dtype=np.int64) - start_rep
            pos = base + offset
            indices_mm[pos] = ev_s
            cur[group_nodes] += group_sizes

        event_idx += batch_len

    assert event_idx == num_events, (event_idx, num_events)
    src_indices_mm.flush()
    dst_indices_mm.flush()
    event_src_mm.flush()
    event_dst_mm.flush()
    event_ts_mm.flush()
    event_w_mm.flush()

    return {
        "csr_events_by_src_id_indptr": src_indptr_path,
        "csr_events_by_src_id_indices": src_indices_path,
        "csr_events_by_dst_id_indptr": dst_indptr_path,
        "csr_events_by_dst_id_indices": dst_indices_path,
        "events_src_id": event_src_path,
        "events_dst_id": event_dst_path,
        "events_ts_s": event_ts_path,
        "events_weight": event_w_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build CSR adjacency for RelBench-style event-node relational graphs. "
            "Outputs node->events CSR for src_id and dst_id, plus per-event arrays."
        )
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--upto", default="val", help="Cutoff: val | test | all | <unix_seconds>")
    parser.add_argument("--batch_size", type=int, default=1_000_000)
    args = parser.parse_args()

    exports_root = Path(args.exports_root)
    meta = _load_meta(exports_root, args.dataset)
    if args.upto == "val":
        upto_s = meta.val_timestamp_s
    elif args.upto == "test":
        upto_s = meta.test_timestamp_s
    elif args.upto == "all":
        upto_s = None
    else:
        upto_s = int(args.upto)

    out = build_rel_event_csr(
        exports_root,
        args.dataset,
        upto_timestamp_s=upto_s,
        batch_size=int(args.batch_size),
        out_dir=None,
    )
    for k, v in out.items():
        print(f"Wrote {k}: {v}")


if __name__ == "__main__":
    main()

