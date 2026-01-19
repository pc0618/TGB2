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


def _detect_bipartite(db_dir: Path) -> tuple[bool, int, int]:
    src_path = db_dir / "src_nodes.parquet"
    dst_path = db_dir / "dst_nodes.parquet"
    if src_path.exists() and dst_path.exists():
        n_src = pq.ParquetFile(src_path).metadata.num_rows
        n_dst = pq.ParquetFile(dst_path).metadata.num_rows
        return True, int(n_src), int(n_dst)
    n = pq.ParquetFile(db_dir / "nodes.parquet").metadata.num_rows
    return False, int(n), 0


def _iter_events_batches(events_path: Path, batch_size: int):
    pf = pq.ParquetFile(events_path)
    yield from pf.iter_batches(batch_size=batch_size, columns=["src_id", "dst_id", "event_ts"])


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _make_memmap_npy(path: Path, dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def build_csr(
    exports_root: Path,
    dataset: str,
    *,
    upto_timestamp_s: Optional[int],
    undirected: bool,
    batch_size: int,
    out_dir: Optional[Path],
) -> tuple[Path, Path]:
    db_dir = exports_root / dataset / "db"
    events_path = db_dir / "events.parquet"
    is_bipartite, n_src, n_dst = _detect_bipartite(db_dir)
    num_nodes = n_src + n_dst if is_bipartite else n_src
    dst_offset = n_src if is_bipartite else 0

    suffix = f"upto_{upto_timestamp_s}" if upto_timestamp_s is not None else "all"
    suffix += "_undirected" if undirected else "_directed"

    if out_dir is None:
        out_dir = exports_root / dataset / "adj"
    out_dir.mkdir(parents=True, exist_ok=True)
    indptr_path = out_dir / f"csr_indptr_{suffix}.npy"
    indices_path = out_dir / f"csr_indices_{suffix}.npy"

    # Pass 1: degrees.
    deg = np.zeros(num_nodes, dtype=np.int64)
    for batch in _iter_events_batches(events_path, batch_size=batch_size):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))
        if upto_timestamp_s is not None:
            mask = ts_s <= int(upto_timestamp_s)
            if not mask.any():
                continue
            src = src[mask]
            dst = dst[mask]
        u = src
        v = dst + dst_offset if is_bipartite else dst
        # Prefer bincount over np.add.at for speed on large arrays.
        deg += np.bincount(u, minlength=num_nodes).astype(np.int64, copy=False)
        if undirected:
            deg += np.bincount(v, minlength=num_nodes).astype(np.int64, copy=False)

    indptr = np.zeros(num_nodes + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg, dtype=np.int64)
    total = int(indptr[-1])

    indptr_mm = _make_memmap_npy(indptr_path, dtype=np.int64, shape=indptr.shape)
    indptr_mm[:] = indptr
    indptr_mm.flush()

    indices_mm = _make_memmap_npy(indices_path, dtype=np.int64, shape=(total,))
    cur = indptr.copy()

    # Pass 2: fill indices.
    for batch in _iter_events_batches(events_path, batch_size=batch_size):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))
        if upto_timestamp_s is not None:
            mask = ts_s <= int(upto_timestamp_s)
            if not mask.any():
                continue
            src = src[mask]
            dst = dst[mask]
        u = src
        v = dst + dst_offset if is_bipartite else dst

        if undirected:
            u = np.concatenate([u, v], axis=0)
            v = np.concatenate([v, src if is_bipartite else dst], axis=0)

        # Vectorized CSR fill: sort by source, then compute per-edge write positions.
        order = np.argsort(u, kind="mergesort")
        u_s = u[order]
        v_s = v[order]

        if u_s.size == 0:
            continue

        # Group boundaries for u_s.
        change = np.flatnonzero(u_s[1:] != u_s[:-1]) + 1
        group_starts = np.concatenate([np.array([0], dtype=np.int64), change.astype(np.int64, copy=False)], axis=0)
        group_sizes = np.diff(np.concatenate([group_starts, np.array([u_s.size], dtype=np.int64)], axis=0))
        group_nodes = u_s[group_starts]

        base = np.repeat(cur[group_nodes], group_sizes)
        start_rep = np.repeat(group_starts, group_sizes)
        offset = np.arange(u_s.size, dtype=np.int64) - start_rep
        pos = base + offset

        indices_mm[pos] = v_s
        cur[group_nodes] += group_sizes

    indices_mm.flush()
    return indptr_path, indices_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSR adjacency arrays from relbench_exports/<dataset>/db/events.parquet.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--upto", default="val", help="Cutoff: val | test | all | <unix_seconds>")
    parser.add_argument("--undirected", action="store_true")
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

    indptr_path, indices_path = build_csr(
        exports_root,
        args.dataset,
        upto_timestamp_s=upto_s,
        undirected=bool(args.undirected),
        batch_size=int(args.batch_size),
        out_dir=None,
    )
    print(f"Wrote {indptr_path}")
    print(f"Wrote {indices_path}")


if __name__ == "__main__":
    main()
