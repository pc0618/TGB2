#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
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


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _make_memmap_npy(path: Path, dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


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
    yield from pf.iter_batches(batch_size=batch_size, columns=["src_id", "dst_id", "event_ts", "weight"])


def build_rel_event_csr_thgl(
    exports_root: Path,
    dataset: str,
    *,
    upto_timestamp_s: Optional[int],
    batch_size: int,
    out_dir: Optional[Path],
) -> dict[str, Path]:
    db_dir = exports_root / dataset / "db"
    event_paths = sorted(db_dir.glob("events_edge_type_*.parquet"), key=lambda p: p.name)
    if not event_paths:
        raise FileNotFoundError(f"No thgl event tables found under {db_dir}")

    node_paths = sorted(db_dir.glob("nodes_type_*.parquet"), key=lambda p: p.name)
    if not node_paths:
        raise FileNotFoundError(f"No thgl node tables found under {db_dir}")

    node_type_to_n: dict[int, int] = {}
    for p in node_paths:
        t = _parse_node_type_from_table_name(p.stem)
        node_type_to_n[t] = int(pq.ParquetFile(p).metadata.num_rows)

    # Per-node-type degrees for src-role and dst-role.
    deg_src: dict[int, np.ndarray] = {t: np.zeros(n, dtype=np.int64) for t, n in node_type_to_n.items()}
    deg_dst: dict[int, np.ndarray] = {t: np.zeros(n, dtype=np.int64) for t, n in node_type_to_n.items()}

    # Gather per-edge-type (src_type, dst_type) from parquet metadata.
    event_specs: list[tuple[int, Path, int, int]] = []
    for p in event_paths:
        et = _parse_edge_type_from_file(p)
        md = _read_relbench_parquet_metadata(p)
        fkeys = md.get("fkey_col_to_pkey_table")
        if not isinstance(fkeys, dict):
            raise RuntimeError(f"Missing RelBench metadata on {p}")
        src_tbl = fkeys.get("src_id")
        dst_tbl = fkeys.get("dst_id")
        if not isinstance(src_tbl, str) or not isinstance(dst_tbl, str):
            raise RuntimeError(f"Malformed fkey metadata on {p}: {fkeys}")
        src_t = _parse_node_type_from_table_name(src_tbl)
        dst_t = _parse_node_type_from_table_name(dst_tbl)
        event_specs.append((et, p, src_t, dst_t))

    # Pass 1: count kept events and degrees.
    num_events = 0
    for et, path, src_t, dst_t in event_specs:
        for batch in _iter_event_batches(path, batch_size=batch_size):
            src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            dst = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(2))
            if upto_timestamp_s is not None:
                mask = ts_s <= int(upto_timestamp_s)
                if not mask.any():
                    continue
                src = src[mask]
                dst = dst[mask]
            if src.size == 0:
                continue
            num_events += int(src.size)
            deg_src[src_t] += np.bincount(src, minlength=deg_src[src_t].shape[0]).astype(np.int64, copy=False)
            deg_dst[dst_t] += np.bincount(dst, minlength=deg_dst[dst_t].shape[0]).astype(np.int64, copy=False)

    suffix = f"upto_{upto_timestamp_s}" if upto_timestamp_s is not None else "all"
    if out_dir is None:
        out_dir = exports_root / dataset / "adj"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Allocate CSR arrays per node type.
    src_indptr: dict[int, np.ndarray] = {}
    dst_indptr: dict[int, np.ndarray] = {}
    src_indices_mm: dict[int, np.memmap] = {}
    dst_indices_mm: dict[int, np.memmap] = {}
    cur_src: dict[int, np.ndarray] = {}
    cur_dst: dict[int, np.ndarray] = {}

    out_paths: dict[str, Path] = {}
    for t, n in sorted(node_type_to_n.items(), key=lambda kv: kv[0]):
        indptr_s = np.zeros(n + 1, dtype=np.int64)
        indptr_d = np.zeros(n + 1, dtype=np.int64)
        indptr_s[1:] = np.cumsum(deg_src[t], dtype=np.int64)
        indptr_d[1:] = np.cumsum(deg_dst[t], dtype=np.int64)

        src_indptr_path = out_dir / f"csr_src_events_indptr_nodes_type_{t}_{suffix}.npy"
        src_indices_path = out_dir / f"csr_src_events_indices_nodes_type_{t}_{suffix}.npy"
        dst_indptr_path = out_dir / f"csr_dst_events_indptr_nodes_type_{t}_{suffix}.npy"
        dst_indices_path = out_dir / f"csr_dst_events_indices_nodes_type_{t}_{suffix}.npy"

        src_indptr_mm = _make_memmap_npy(src_indptr_path, dtype=np.int64, shape=indptr_s.shape)
        dst_indptr_mm = _make_memmap_npy(dst_indptr_path, dtype=np.int64, shape=indptr_d.shape)
        src_indptr_mm[:] = indptr_s
        dst_indptr_mm[:] = indptr_d
        src_indptr_mm.flush()
        dst_indptr_mm.flush()

        src_indices_mm[t] = _make_memmap_npy(src_indices_path, dtype=np.int64, shape=(int(indptr_s[-1]),))
        dst_indices_mm[t] = _make_memmap_npy(dst_indices_path, dtype=np.int64, shape=(int(indptr_d[-1]),))

        src_indptr[t] = indptr_s
        dst_indptr[t] = indptr_d
        cur_src[t] = indptr_s.copy()
        cur_dst[t] = indptr_d.copy()

        out_paths[f"csr_src_events_indptr_nodes_type_{t}"] = src_indptr_path
        out_paths[f"csr_src_events_indices_nodes_type_{t}"] = src_indices_path
        out_paths[f"csr_dst_events_indptr_nodes_type_{t}"] = dst_indptr_path
        out_paths[f"csr_dst_events_indices_nodes_type_{t}"] = dst_indices_path

    # Allocate global event arrays.
    event_src_type_path = out_dir / f"events_src_type_{suffix}.npy"
    event_dst_type_path = out_dir / f"events_dst_type_{suffix}.npy"
    event_edge_type_path = out_dir / f"events_edge_type_{suffix}.npy"
    event_src_id_path = out_dir / f"events_src_id_{suffix}.npy"
    event_dst_id_path = out_dir / f"events_dst_id_{suffix}.npy"
    event_ts_path = out_dir / f"events_ts_s_{suffix}.npy"
    event_w_path = out_dir / f"events_weight_{suffix}.npy"

    ev_src_type_mm = _make_memmap_npy(event_src_type_path, dtype=np.int16, shape=(num_events,))
    ev_dst_type_mm = _make_memmap_npy(event_dst_type_path, dtype=np.int16, shape=(num_events,))
    ev_edge_type_mm = _make_memmap_npy(event_edge_type_path, dtype=np.int16, shape=(num_events,))
    ev_src_id_mm = _make_memmap_npy(event_src_id_path, dtype=np.int64, shape=(num_events,))
    ev_dst_id_mm = _make_memmap_npy(event_dst_id_path, dtype=np.int64, shape=(num_events,))
    ev_ts_mm = _make_memmap_npy(event_ts_path, dtype=np.int64, shape=(num_events,))
    ev_w_mm = _make_memmap_npy(event_w_path, dtype=np.float32, shape=(num_events,))

    out_paths["events_src_type"] = event_src_type_path
    out_paths["events_dst_type"] = event_dst_type_path
    out_paths["events_edge_type"] = event_edge_type_path
    out_paths["events_src_id"] = event_src_id_path
    out_paths["events_dst_id"] = event_dst_id_path
    out_paths["events_ts_s"] = event_ts_path
    out_paths["events_weight"] = event_w_path

    # Pass 2: fill event arrays and CSR indices.
    event_idx = 0
    for et, path, src_t, dst_t in event_specs:
        for batch in _iter_event_batches(path, batch_size=batch_size):
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

            k = int(src.size)
            ev = np.arange(event_idx, event_idx + k, dtype=np.int64)
            ev_src_type_mm[ev] = np.int16(src_t)
            ev_dst_type_mm[ev] = np.int16(dst_t)
            ev_edge_type_mm[ev] = np.int16(et)
            ev_src_id_mm[ev] = src
            ev_dst_id_mm[ev] = dst
            ev_ts_mm[ev] = ts_s.astype(np.int64, copy=False)
            ev_w_mm[ev] = w

            # Fill src-role CSR for src_t: src_id -> event_id
            order_src = np.argsort(src, kind="mergesort")
            src_s = src[order_src]
            ev_s = ev[order_src]
            if src_s.size:
                change = np.flatnonzero(src_s[1:] != src_s[:-1]) + 1
                group_starts = np.concatenate(
                    [np.array([0], dtype=np.int64), change.astype(np.int64, copy=False)], axis=0
                )
                group_sizes = np.diff(
                    np.concatenate([group_starts, np.array([src_s.size], dtype=np.int64)], axis=0)
                )
                group_nodes = src_s[group_starts]
                base = np.repeat(cur_src[src_t][group_nodes], group_sizes)
                start_rep = np.repeat(group_starts, group_sizes)
                offset = np.arange(src_s.size, dtype=np.int64) - start_rep
                pos = base + offset
                src_indices_mm[src_t][pos] = ev_s
                cur_src[src_t][group_nodes] += group_sizes

            # Fill dst-role CSR for dst_t: dst_id -> event_id
            order_dst = np.argsort(dst, kind="mergesort")
            dst_s = dst[order_dst]
            ev_d = ev[order_dst]
            if dst_s.size:
                change = np.flatnonzero(dst_s[1:] != dst_s[:-1]) + 1
                group_starts = np.concatenate(
                    [np.array([0], dtype=np.int64), change.astype(np.int64, copy=False)], axis=0
                )
                group_sizes = np.diff(
                    np.concatenate([group_starts, np.array([dst_s.size], dtype=np.int64)], axis=0)
                )
                group_nodes = dst_s[group_starts]
                base = np.repeat(cur_dst[dst_t][group_nodes], group_sizes)
                start_rep = np.repeat(group_starts, group_sizes)
                offset = np.arange(dst_s.size, dtype=np.int64) - start_rep
                pos = base + offset
                dst_indices_mm[dst_t][pos] = ev_d
                cur_dst[dst_t][group_nodes] += group_sizes

            event_idx += k

    assert event_idx == num_events, (event_idx, num_events)

    for mm in src_indices_mm.values():
        mm.flush()
    for mm in dst_indices_mm.values():
        mm.flush()
    for mm in [ev_src_type_mm, ev_dst_type_mm, ev_edge_type_mm, ev_src_id_mm, ev_dst_id_mm, ev_ts_mm, ev_w_mm]:
        mm.flush()

    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build CSR adjacencies and unified event arrays for RelBench-exported thgl-* datasets. "
            "Outputs node->events CSR per node type (src-role and dst-role) plus per-event arrays."
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

    out = build_rel_event_csr_thgl(
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

