#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ExportMeta:
    dataset: str
    val_timestamp_s: int
    test_timestamp_s: int
    num_classes: int | None


def _load_meta(exports_root: Path, dataset: str) -> ExportMeta:
    meta_path = exports_root / dataset / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cutoffs = meta["cutoffs"]
    extra = meta.get("extra", {})
    num_classes = extra.get("num_classes")
    return ExportMeta(
        dataset=meta["dataset"],
        val_timestamp_s=int(cutoffs["val_timestamp_s"]),
        test_timestamp_s=int(cutoffs["test_timestamp_s"]),
        num_classes=int(num_classes) if num_classes is not None else None,
    )


def _timestamp_s_from_arrow_timestamp_ns(arr) -> np.ndarray:
    ns = arr.cast("timestamp[ns]").to_numpy(zero_copy_only=False).astype("datetime64[ns]")
    return (ns.astype("int64") // 1_000_000_000).astype(np.int64)


def _make_memmap_npy(path: Path, dtype, shape: tuple[int, ...]):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(str(path), mode="w+", dtype=dtype, shape=shape)


def build_label_event_csr(exports_root: Path, dataset: str, *, batch_size: int) -> dict[str, Path]:
    db_dir = exports_root / dataset / "db"
    label_events_path = db_dir / "label_events.parquet"
    label_items_path = db_dir / "label_event_items.parquet"
    if not label_events_path.exists() or not label_items_path.exists():
        raise FileNotFoundError(f"Missing label tables under {db_dir} (expected label_events.parquet and label_event_items.parquet).")

    n_label_events = int(pq.ParquetFile(label_events_path).metadata.num_rows)

    # Pass 1: degrees.
    deg = np.zeros(n_label_events, dtype=np.int64)
    pf = pq.ParquetFile(label_items_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["label_event_id"]):
        le = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        deg += np.bincount(le, minlength=n_label_events).astype(np.int64, copy=False)

    indptr = np.zeros(n_label_events + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(deg, dtype=np.int64)
    total = int(indptr[-1])

    out_dir = exports_root / dataset / "adj"
    out_dir.mkdir(parents=True, exist_ok=True)
    indptr_path = out_dir / "label_event_indptr.npy"
    label_id_path = out_dir / "label_event_label_ids.npy"
    weight_path = out_dir / "label_event_label_weights.npy"
    src_id_path = out_dir / "label_event_src_ids.npy"
    ts_s_path = out_dir / "label_event_ts_s.npy"

    indptr_mm = _make_memmap_npy(indptr_path, dtype=np.int64, shape=indptr.shape)
    indptr_mm[:] = indptr
    indptr_mm.flush()

    label_id_mm = _make_memmap_npy(label_id_path, dtype=np.int64, shape=(total,))
    weight_mm = _make_memmap_npy(weight_path, dtype=np.float32, shape=(total,))

    # label_event attributes:
    src_id_mm = _make_memmap_npy(src_id_path, dtype=np.int64, shape=(n_label_events,))
    ts_s_mm = _make_memmap_npy(ts_s_path, dtype=np.int64, shape=(n_label_events,))
    pf_le = pq.ParquetFile(label_events_path)
    # label_events are written with consecutive pkeys starting at 0, so row order equals label_event_id.
    offset = 0
    for batch in pf_le.iter_batches(batch_size=batch_size, columns=["src_id", "label_ts"]):
        src = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        ts_s = _timestamp_s_from_arrow_timestamp_ns(batch.column(1))
        n = int(src.shape[0])
        src_id_mm[offset : offset + n] = src
        ts_s_mm[offset : offset + n] = ts_s
        offset += n
    assert offset == n_label_events
    src_id_mm.flush()
    ts_s_mm.flush()

    # Pass 2: fill label ids and weights.
    cur = indptr.copy()
    for batch in pf.iter_batches(batch_size=batch_size, columns=["label_event_id", "label_id", "label_weight"]):
        le = batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        lab = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        w = batch.column(2).to_numpy(zero_copy_only=False).astype(np.float32, copy=False)

        order = np.argsort(le, kind="mergesort")
        le_s = le[order]
        lab_s = lab[order]
        w_s = w[order]
        if le_s.size == 0:
            continue

        change = np.flatnonzero(le_s[1:] != le_s[:-1]) + 1
        group_starts = np.concatenate([np.array([0], dtype=np.int64), change.astype(np.int64, copy=False)], axis=0)
        group_sizes = np.diff(np.concatenate([group_starts, np.array([le_s.size], dtype=np.int64)], axis=0))
        group_nodes = le_s[group_starts]

        base = np.repeat(cur[group_nodes], group_sizes)
        start_rep = np.repeat(group_starts, group_sizes)
        offset_arr = np.arange(le_s.size, dtype=np.int64) - start_rep
        pos = base + offset_arr

        label_id_mm[pos] = lab_s
        weight_mm[pos] = w_s
        cur[group_nodes] += group_sizes

    label_id_mm.flush()
    weight_mm.flush()

    return {
        "label_event_indptr": indptr_path,
        "label_event_label_ids": label_id_path,
        "label_event_label_weights": weight_path,
        "label_event_src_ids": src_id_path,
        "label_event_ts_s": ts_s_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CSR mapping label_event_id -> (label_id, label_weight) for tgbn-* exports.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exports_root", default="relbench_exports")
    parser.add_argument("--batch_size", type=int, default=1_000_000)
    args = parser.parse_args()

    exports_root = Path(args.exports_root)
    _ = _load_meta(exports_root, args.dataset)  # validate metadata exists
    out = build_label_event_csr(exports_root, args.dataset, batch_size=int(args.batch_size))
    for k, v in out.items():
        print(f"Wrote {k}: {v}")


if __name__ == "__main__":
    main()

