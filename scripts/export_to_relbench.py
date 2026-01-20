#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from relbench.base import Database, Table

from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.utils.info import DATA_NUM_CLASSES
from tgb.utils.utils import load_pkl


@dataclass(frozen=True)
class Cutoffs:
    val_timestamp_s: int
    test_timestamp_s: int

    @property
    def val_timestamp(self) -> pd.Timestamp:
        return pd.to_datetime(self.val_timestamp_s, unit="s", utc=True)

    @property
    def test_timestamp(self) -> pd.Timestamp:
        return pd.to_datetime(self.test_timestamp_s, unit="s", utc=True)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _detect_bipartite_offset(src: np.ndarray, dst: np.ndarray) -> int | None:
    if src.size == 0 or dst.size == 0:
        return None
    offset = int(src.max()) + 1
    if int(dst.min()) >= offset:
        return offset
    # Some datasets encode bipartite ids such that dst are in [0, offset) and
    # src are in [offset, ...). In that case, src.min() is the natural offset.
    if int(dst.max()) < int(src.min()):
        return int(src.min())
    return None


def _compute_cutoffs(ts: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> Cutoffs:
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Expected non-empty train/val/test masks.")
    return Cutoffs(val_timestamp_s=int(ts[val_mask].min()), test_timestamp_s=int(ts[test_mask].min()))


def _ts_to_utc(ts_s: np.ndarray) -> pd.Series:
    return pd.Series(pd.to_datetime(ts_s.astype(np.int64, copy=False), unit="s", utc=True))


def _maybe_convert_years_to_unix_seconds(ts: np.ndarray) -> np.ndarray:
    """
    Some nodeprop datasets (notably tgbn-trade) use integer years as timestamps.
    Convert these to unix seconds at Jan 1 of each year so the RelBench time column
    is a real timestamp while preserving ordering.
    """
    ts_i = np.asarray(ts, dtype=np.int64)
    if ts_i.size == 0:
        return ts_i
    t_min = int(ts_i.min())
    t_max = int(ts_i.max())
    # Heuristic: years in [1900, 2100] and small magnitudes compared to unix seconds.
    if 1900 <= t_min <= 2100 and 1900 <= t_max <= 2100:
        out = np.empty(ts_i.shape[0], dtype=np.int64)
        for i, y in enumerate(ts_i.tolist()):
            out[i] = int(datetime(int(y), 1, 1, tzinfo=timezone.utc).timestamp())
        return out
    return ts_i


def _generate_splits_by_quantiles(
    ts_s: np.ndarray, *, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    val_time, test_time = list(np.quantile(ts_s, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    train_mask = ts_s <= val_time
    val_mask = np.logical_and(ts_s <= test_time, ts_s > val_time)
    test_mask = ts_s > test_time
    return train_mask, val_mask, test_mask


def _load_nodeprop_processed_edges_without_labels(name: str, root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Load (src, dst, ts_s, weight) for nodeprop datasets WITHOUT materializing the dense node_label_dict.

    For large datasets (tgbn-reddit/token), the official loader constructs dense label vectors and can
    exhaust memory. Here we rely on the already-created processed pickle files (ml_*.pkl + ml_*_edge.pkl).
    """
    dir_name = "_".join(name.split("-"))
    ds_dir = Path("tgb") / "datasets" / dir_name
    df_path = ds_dir / f"ml_{name}.pkl"
    edge_feat_path = ds_dir / f"ml_{name}_edge.pkl"
    if not df_path.exists() or not edge_feat_path.exists():
        # Fallback: use official dataset (may be memory heavy).
        ds = NodePropPredDataset(name=name, root=root, download=True)
        full = ds.full_data
        src = np.asarray(full["sources"], dtype=np.int64)
        dst = np.asarray(full["destinations"], dtype=np.int64)
        ts_s = np.asarray(full["timestamps"], dtype=np.int64)
        edge_feat = np.asarray(full["edge_feat"])
        if edge_feat.ndim == 1:
            weight = edge_feat.astype(np.float32, copy=False)
        elif edge_feat.ndim == 2 and edge_feat.shape[1] >= 1:
            weight = edge_feat[:, 0].astype(np.float32, copy=False)
        else:
            weight = np.ones(src.shape[0], dtype=np.float32)
        return src, dst, ts_s, weight, int(ds.num_classes)

    df = pd.read_pickle(df_path)
    src = np.asarray(df["u"], dtype=np.int64)
    dst = np.asarray(df["i"], dtype=np.int64)
    ts_s = np.asarray(df["ts"], dtype=np.int64)
    edge_feat = np.asarray(load_pkl(str(edge_feat_path)))
    if name == "tgbn-token":
        # Match NodePropPredDataset.generate_processed_files() stability tweak.
        edge_feat = edge_feat.copy()
        edge_feat[:, 0] = np.log(edge_feat[:, 0])
    if edge_feat.ndim == 1:
        weight = edge_feat.astype(np.float32, copy=False)
    elif edge_feat.ndim == 2 and edge_feat.shape[1] >= 1:
        weight = edge_feat[:, 0].astype(np.float32, copy=False)
    else:
        weight = np.ones(src.shape[0], dtype=np.float32)

    num_classes = int(DATA_NUM_CLASSES.get(name, 0) or 0)
    if num_classes <= 0:
        # Conservative fallback: assume labels are 0..max(dst) if bipartite encoding was used.
        num_classes = int(dst.max()) + 1 if dst.size else 0
    return src, dst, ts_s, weight, num_classes

def _relbench_table_metadata_bytes(
    *,
    fkey_col_to_pkey_table: dict[str, str],
    pkey_col: str | None,
    time_col: str | None,
) -> dict[bytes, bytes]:
    # Match `relbench.base.Table.save()` metadata keys.
    return {
        b"fkey_col_to_pkey_table": json.dumps(fkey_col_to_pkey_table).encode("utf-8"),
        b"pkey_col": json.dumps(pkey_col).encode("utf-8"),
        b"time_col": json.dumps(time_col).encode("utf-8"),
    }


def _write_relbench_parquet_table_chunked(
    out_path: Path,
    *,
    num_rows: int,
    chunk_size: int,
    make_chunk: "callable[[int, int], dict[str, pa.Array]]",
    fkey_col_to_pkey_table: dict[str, str],
    pkey_col: str | None,
    time_col: str | None,
) -> None:
    if num_rows == 0:
        # Still write an empty parquet with the right metadata if desired.
        schema = pa.schema([]).with_metadata(
            _relbench_table_metadata_bytes(
                fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                pkey_col=pkey_col,
                time_col=time_col,
            )
        )
        pq.write_table(pa.Table.from_arrays([], schema=schema), out_path)
        return

    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, num_rows, chunk_size):
            end = min(num_rows, start + chunk_size)
            arrays = make_chunk(start, end)
            table = pa.Table.from_pydict(arrays)
            if writer is None:
                schema = table.schema.with_metadata(
                    _relbench_table_metadata_bytes(
                        fkey_col_to_pkey_table=fkey_col_to_pkey_table,
                        pkey_col=pkey_col,
                        time_col=time_col,
                    )
                )
                writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def _dbml_from_database(db: Database, *, dataset: str) -> str:
    lines: list[str] = []
    lines.append(f"// Auto-generated RelBench DB schema for {dataset}")
    lines.append(f"// Generated at {datetime.now(timezone.utc).isoformat()}")
    lines.append("")
    for table_name, table in db.table_dict.items():
        lines.append(f"Table {table_name} {{")
        cols = list(table.df.columns)
        for col in cols:
            dtype = table.df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                dbml_type = "bigint"
            elif pd.api.types.is_float_dtype(dtype):
                dbml_type = "float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                dbml_type = "datetime"
            else:
                dbml_type = "text"

            attrs: list[str] = []
            if table.pkey_col == col:
                attrs.append("pk")
            if col in table.fkey_col_to_pkey_table:
                ref_table = table.fkey_col_to_pkey_table[col]
                ref_col = db.table_dict[ref_table].pkey_col or "id"
                attrs.append(f"ref: > {ref_table}.{ref_col}")
            if attrs:
                lines.append(f"  {col} {dbml_type} [{', '.join(attrs)}]")
            else:
                lines.append(f"  {col} {dbml_type}")
        if table.time_col:
            lines.append(f"  // time_col: {table.time_col}")
        lines.append("}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _export_link_dataset(name: str, root: str) -> tuple[Database, Cutoffs, dict]:
    ds = LinkPropPredDataset(name=name, root=root, download=True)
    full = ds.full_data

    src = np.asarray(full["sources"], dtype=np.int64)
    dst = np.asarray(full["destinations"], dtype=np.int64)
    ts_s = np.asarray(full["timestamps"], dtype=np.int64)
    weight = np.asarray(full["w"], dtype=np.float32)

    train_mask = np.asarray(ds.train_mask, dtype=bool)
    val_mask = np.asarray(ds.val_mask, dtype=bool)
    test_mask = np.asarray(ds.test_mask, dtype=bool)
    cutoffs = _compute_cutoffs(ts_s, train_mask, val_mask, test_mask)

    bipartite_offset = _detect_bipartite_offset(src, dst)
    extra_meta: dict = {"bipartite_offset": int(bipartite_offset) if bipartite_offset is not None else None}

    # Heterogeneous: thgl-* expose node types + edge types.
    node_type = getattr(ds, "node_type", None)
    edge_type = full.get("edge_type", None)
    if node_type is not None and edge_type is not None:
        node_type = np.asarray(node_type, dtype=np.int64)
        edge_type = np.asarray(edge_type, dtype=np.int64)
        extra_meta["node_type_count"] = int(node_type.max()) + 1 if node_type.size else 0
        extra_meta["edge_type_count"] = int(edge_type.max()) + 1 if edge_type.size else 0

        # Build per-node-type entity tables.
        type_to_globals: dict[int, np.ndarray] = {}
        for t in np.unique(node_type):
            type_to_globals[int(t)] = np.where(node_type == t)[0].astype(np.int64)
        global_to_local: dict[int, np.ndarray] = {}
        for t, globals_ in type_to_globals.items():
            mapper = np.full(node_type.shape[0], -1, dtype=np.int64)
            mapper[globals_] = np.arange(globals_.shape[0], dtype=np.int64)
            global_to_local[t] = mapper

        table_dict: dict[str, Table] = {}
        for t, globals_ in sorted(type_to_globals.items(), key=lambda kv: kv[0]):
            table_name = f"nodes_type_{t}"
            df = pd.DataFrame({f"node_type_{t}_id": np.arange(globals_.shape[0], dtype=np.int64)})
            table_dict[table_name] = Table(df=df, pkey_col=f"node_type_{t}_id", fkey_col_to_pkey_table={}, time_col=None)

        # Build one event table per edge_type id; each should have fixed (src_type, dst_type).
        #
        # NOTE: For large `thgl-*` datasets, materializing pandas DataFrames for every
        # edge-type table can blow up memory. We therefore return placeholder
        # (empty) event tables and stream-write the corresponding parquet files in
        # `main()` based on the metadata below.
        order = np.argsort(edge_type, kind="mergesort")
        edge_sorted = edge_type[order]
        ets, starts, counts = np.unique(edge_sorted, return_index=True, return_counts=True)

        placeholder_events = pd.DataFrame(
            {
                "event_id": np.array([], dtype=np.int64),
                "src_id": np.array([], dtype=np.int64),
                "dst_id": np.array([], dtype=np.int64),
                "event_ts": pd.to_datetime([], utc=True),
                "weight": np.array([], dtype=np.float32),
            }
        )

        edge_type_specs: list[dict] = []
        for et, start, count in zip(ets.tolist(), starts.tolist(), counts.tolist()):
            et = int(et)
            start = int(start)
            count = int(count)
            if count == 0:
                continue

            idx = order[start : start + count]
            src_g = src[idx]
            dst_g = dst[idx]
            src_types = np.unique(node_type[src_g])
            dst_types = np.unique(node_type[dst_g])
            if src_types.size != 1 or dst_types.size != 1:
                raise RuntimeError(
                    f"edge_type={et} is not type-homogeneous "
                    f"(src_types={src_types.tolist()}, dst_types={dst_types.tolist()})."
                )
            src_t = int(src_types[0])
            dst_t = int(dst_types[0])

            src_table = f"nodes_type_{src_t}"
            dst_table = f"nodes_type_{dst_t}"
            table_dict[f"events_edge_type_{et}"] = Table(
                df=placeholder_events,
                pkey_col="event_id",
                time_col="event_ts",
                fkey_col_to_pkey_table={"src_id": src_table, "dst_id": dst_table},
            )
            edge_type_specs.append(
                {
                    "edge_type": et,
                    "src_type": src_t,
                    "dst_type": dst_t,
                    "start_in_sorted": start,
                    "num_rows": count,
                }
            )

        extra_meta["__streaming_events_hetero__"] = {"mode": "hetero", "edge_types": edge_type_specs}
        return Database(table_dict), cutoffs, extra_meta

    # Homogeneous or bipartite:
    if bipartite_offset is not None:
        n_src = int(src.max()) + 1
        n_dst = int(dst.max() - bipartite_offset) + 1

        src_nodes = pd.DataFrame({"src_id": np.arange(n_src, dtype=np.int64)})
        dst_nodes = pd.DataFrame({"dst_id": np.arange(n_dst, dtype=np.int64)})

        # We return a Database object for schema/dbml generation and downstream loading,
        # but we avoid building the massive events DataFrame here.
        placeholder_events = pd.DataFrame(
            {
                "event_id": np.array([], dtype=np.int64),
                "src_id": np.array([], dtype=np.int64),
                "dst_id": np.array([], dtype=np.int64),
                "event_ts": pd.to_datetime([], utc=True),
                "weight": np.array([], dtype=np.float32),
            }
        )
        db = Database(
            {
                "src_nodes": Table(df=src_nodes, pkey_col="src_id", fkey_col_to_pkey_table={}, time_col=None),
                "dst_nodes": Table(df=dst_nodes, pkey_col="dst_id", fkey_col_to_pkey_table={}, time_col=None),
                "events": Table(
                    df=placeholder_events,
                    pkey_col="event_id",
                    time_col="event_ts",
                    fkey_col_to_pkey_table={"src_id": "src_nodes", "dst_id": "dst_nodes"},
                ),
            }
        )
        extra_meta["__streaming_events__"] = {
            "mode": "bipartite",
            "num_rows": int(src.shape[0]),
        }
        return db, cutoffs, extra_meta

    n_nodes = int(max(src.max(), dst.max())) + 1
    nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64)})
    placeholder_events = pd.DataFrame(
        {
            "event_id": np.array([], dtype=np.int64),
            "src_id": np.array([], dtype=np.int64),
            "dst_id": np.array([], dtype=np.int64),
            "event_ts": pd.to_datetime([], utc=True),
            "weight": np.array([], dtype=np.float32),
        }
    )
    db = Database(
        {
            "nodes": Table(df=nodes, pkey_col="node_id", fkey_col_to_pkey_table={}, time_col=None),
            "events": Table(
                df=placeholder_events,
                pkey_col="event_id",
                time_col="event_ts",
                fkey_col_to_pkey_table={"src_id": "nodes", "dst_id": "nodes"},
            ),
        }
    )
    extra_meta["__streaming_events__"] = {"mode": "homogeneous", "num_rows": int(src.shape[0])}
    return db, cutoffs, extra_meta


def _export_node_dataset(name: str, root: str) -> tuple[Database, Cutoffs, dict]:
    src, dst, ts_raw, weight, num_classes = _load_nodeprop_processed_edges_without_labels(name, root)
    ts_s = _maybe_convert_years_to_unix_seconds(ts_raw)
    train_mask, val_mask, test_mask = _generate_splits_by_quantiles(ts_s)
    cutoffs = _compute_cutoffs(ts_s, train_mask, val_mask, test_mask)

    extra_meta: dict = {
        "num_classes": int(num_classes),
        "label_timestamps": None,
        "__streaming_label_events__": True,
    }

    n_nodes = int(max(src.max(), dst.max())) + 1
    nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64)})
    labels = pd.DataFrame({"label_id": np.arange(int(num_classes), dtype=np.int64)})

    placeholder_events = pd.DataFrame(
        {
            "event_id": np.array([], dtype=np.int64),
            "src_id": np.array([], dtype=np.int64),
            "dst_id": np.array([], dtype=np.int64),
            "event_ts": pd.to_datetime([], utc=True),
            "weight": np.array([], dtype=np.float32),
        }
    )
    placeholder_label_events = pd.DataFrame(
        {
            "label_event_id": np.array([], dtype=np.int64),
            "src_id": np.array([], dtype=np.int64),
            "label_ts": pd.to_datetime([], utc=True),
        }
    )
    placeholder_label_items = pd.DataFrame(
        {
            "item_id": np.array([], dtype=np.int64),
            "label_event_id": np.array([], dtype=np.int64),
            "label_id": np.array([], dtype=np.int64),
            "label_weight": np.array([], dtype=np.float32),
        }
    )
    db = Database(
        {
            "nodes": Table(df=nodes, pkey_col="node_id", fkey_col_to_pkey_table={}, time_col=None),
            "labels": Table(df=labels, pkey_col="label_id", fkey_col_to_pkey_table={}, time_col=None),
            "events": Table(
                df=placeholder_events,
                pkey_col="event_id",
                time_col="event_ts",
                fkey_col_to_pkey_table={"src_id": "nodes", "dst_id": "nodes"},
            ),
            "label_events": Table(
                df=placeholder_label_events,
                pkey_col="label_event_id",
                time_col="label_ts",
                fkey_col_to_pkey_table={"src_id": "nodes"},
            ),
            "label_event_items": Table(
                df=placeholder_label_items,
                pkey_col="item_id",
                time_col=None,
                fkey_col_to_pkey_table={"label_event_id": "label_events", "label_id": "labels"},
            ),
        }
    )
    extra_meta["__streaming_events__"] = {"mode": "homogeneous", "num_rows": int(src.shape[0])}
    return db, cutoffs, extra_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TGB datasets into RelBench Database parquet format.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. tgbl-wiki-v2")
    parser.add_argument("--root", default="datasets", help="TGB dataset root.")
    parser.add_argument("--out_dir", default="relbench_exports", help="Output directory root.")
    parser.add_argument("--chunk_size", type=int, default=2_000_000, help="Rows per parquet chunk for large event tables.")
    args = parser.parse_args()

    dataset_name = args.dataset
    if dataset_name.startswith("tkgl-"):
        raise ValueError("tkgl-* datasets are excluded from this pipeline (knowledge graphs).")

    # Map “-v2” dataset ids (docs) to the library ids (info.py).
    internal_name = dataset_name
    if dataset_name in ("tgbl-wiki-v2",):
        internal_name = "tgbl-wiki"
    if dataset_name in ("tgbl-review-v2",):
        internal_name = "tgbl-review"
    if dataset_name in ("tgbl-coin-v2",):
        internal_name = "tgbl-coin"
    if dataset_name in ("tgbl-flight-v2",):
        internal_name = "tgbl-flight"

    out_root = Path(args.out_dir)
    out_dir = out_root / dataset_name
    db_dir = out_dir / "db"
    _ensure_dir(db_dir)

    if dataset_name.startswith("tgbn-"):
        db, cutoffs, meta_extra = _export_node_dataset(internal_name, args.root)
    else:
        db, cutoffs, meta_extra = _export_link_dataset(internal_name, args.root)

    # Persist as parquet with RelBench table metadata.
    db.reindex_pkeys_and_fkeys()
    # For very large link datasets, write `events.parquet` in chunks to avoid building a massive DataFrame.
    if "__streaming_events__" in meta_extra:
        stream = meta_extra["__streaming_events__"]
        # Save non-event tables via normal Table.save (small).
        for name, table in db.table_dict.items():
            if name == "events":
                continue
            table.save(db_dir / f"{name}.parquet")

        # Re-load the raw arrays from TGB again (cheap vs holding huge DF) and stream-write parquet.
        # Note: tgbn-* is nodeprop and has a different loader than linkproppred.
        if dataset_name.startswith("tgbn-"):
            src, dst, ts_raw, weight, _num_classes = _load_nodeprop_processed_edges_without_labels(internal_name, args.root)
            ts_s = _maybe_convert_years_to_unix_seconds(ts_raw)
            bipartite_offset = None
        else:
            ds = LinkPropPredDataset(name=internal_name, root=args.root, download=True)
            full = ds.full_data
            src = np.asarray(full["sources"], dtype=np.int64)
            dst = np.asarray(full["destinations"], dtype=np.int64)
            ts_s = np.asarray(full["timestamps"], dtype=np.int64)
            weight = np.asarray(full["w"], dtype=np.float32)
            bipartite_offset = stream["mode"] == "bipartite" and _detect_bipartite_offset(src, dst) or None

        out_events = db_dir / "events.parquet"
        ts_ns = ts_s.astype(np.int64, copy=False) * 1_000_000_000
        ts_type = pa.timestamp("ns", tz="UTC")

        if bipartite_offset is not None:
            dst_local = (dst - int(bipartite_offset)).astype(np.int64, copy=False)

            def make_chunk(start: int, end: int):
                size = end - start
                return {
                    "event_id": pa.array(np.arange(start, end, dtype=np.int64)),
                    "src_id": pa.array(src[start:end]),
                    "dst_id": pa.array(dst_local[start:end]),
                    "event_ts": pa.array(ts_ns[start:end], type=ts_type),
                    "weight": pa.array(weight[start:end]),
                }

            _write_relbench_parquet_table_chunked(
                out_events,
                num_rows=int(src.shape[0]),
                chunk_size=int(args.chunk_size),
                make_chunk=make_chunk,
                fkey_col_to_pkey_table={"src_id": "src_nodes", "dst_id": "dst_nodes"},
                pkey_col="event_id",
                time_col="event_ts",
            )
        else:

            def make_chunk(start: int, end: int):
                size = end - start
                return {
                    "event_id": pa.array(np.arange(start, end, dtype=np.int64)),
                    "src_id": pa.array(src[start:end]),
                    "dst_id": pa.array(dst[start:end]),
                    "event_ts": pa.array(ts_ns[start:end], type=ts_type),
                    "weight": pa.array(weight[start:end]),
                }

            _write_relbench_parquet_table_chunked(
                out_events,
                num_rows=int(src.shape[0]),
                chunk_size=int(args.chunk_size),
                make_chunk=make_chunk,
                fkey_col_to_pkey_table={"src_id": "nodes", "dst_id": "nodes"},
                pkey_col="event_id",
                time_col="event_ts",
            )
    elif "__streaming_events_hetero__" in meta_extra:
        # Save non-event tables via normal Table.save (small).
        for name, table in db.table_dict.items():
            if name.startswith("events_edge_type_"):
                continue
            table.save(db_dir / f"{name}.parquet")

        # Reload the raw arrays from TGB again (cheap vs holding huge DF) and stream-write parquets.
        ds = LinkPropPredDataset(name=internal_name, root=args.root, download=True)
        full = ds.full_data
        src = np.asarray(full["sources"], dtype=np.int64)
        dst = np.asarray(full["destinations"], dtype=np.int64)
        ts_s = np.asarray(full["timestamps"], dtype=np.int64)
        weight = np.asarray(full["w"], dtype=np.float32)
        edge_type = np.asarray(full["edge_type"], dtype=np.int64)
        node_type = np.asarray(getattr(ds, "node_type"), dtype=np.int64)

        # Build global->local mapper per node type for writing local ids.
        type_to_globals: dict[int, np.ndarray] = {}
        for t in np.unique(node_type):
            type_to_globals[int(t)] = np.where(node_type == t)[0].astype(np.int64)
        global_to_local: dict[int, np.ndarray] = {}
        for t, globals_ in type_to_globals.items():
            mapper = np.full(node_type.shape[0], -1, dtype=np.int64)
            mapper[globals_] = np.arange(globals_.shape[0], dtype=np.int64)
            global_to_local[t] = mapper

        order = np.argsort(edge_type, kind="mergesort")
        ts_ns = ts_s.astype(np.int64, copy=False) * 1_000_000_000
        ts_type = pa.timestamp("ns", tz="UTC")

        for spec in meta_extra["__streaming_events_hetero__"]["edge_types"]:
            et = int(spec["edge_type"])
            src_t = int(spec["src_type"])
            dst_t = int(spec["dst_type"])
            start_in_sorted = int(spec["start_in_sorted"])
            num_rows = int(spec["num_rows"])

            meta_tbl = db.table_dict[f"events_edge_type_{et}"]
            out_events = db_dir / f"events_edge_type_{et}.parquet"

            def make_chunk(start: int, end: int):
                idx = order[start_in_sorted + start : start_in_sorted + end]
                src_g = src[idx]
                dst_g = dst[idx]
                src_local = global_to_local[src_t][src_g]
                dst_local = global_to_local[dst_t][dst_g]
                if (src_local < 0).any() or (dst_local < 0).any():
                    raise RuntimeError(f"Found unmapped node ids while streaming edge_type={et}.")
                return {
                    "event_id": pa.array(np.arange(start, end, dtype=np.int64)),
                    "src_id": pa.array(src_local.astype(np.int64, copy=False)),
                    "dst_id": pa.array(dst_local.astype(np.int64, copy=False)),
                    "event_ts": pa.array(ts_ns[idx], type=ts_type),
                    "weight": pa.array(weight[idx].astype(np.float32, copy=False), type=pa.float32()),
                }

            _write_relbench_parquet_table_chunked(
                out_events,
                num_rows=num_rows,
                chunk_size=int(args.chunk_size),
                make_chunk=make_chunk,
                fkey_col_to_pkey_table=meta_tbl.fkey_col_to_pkey_table,
                pkey_col=meta_tbl.pkey_col,
                time_col=meta_tbl.time_col,
            )
    else:
        db.save(db_dir)

    # NodeProp: stream-write label tables (can be large); implemented in a normalized
    # relational form (label_events + label_event_items).
    if dataset_name.startswith("tgbn-") and meta_extra.get("__streaming_label_events__"):
        # IMPORTANT: For large datasets (tgbn-reddit/token), the official loader builds dense label vectors.
        # Instead, stream from the raw node-label CSV and only emit non-zeros.
        stream_from_csv = internal_name in ("tgbn-token", "tgbn-reddit")
        dir_name = "_".join(internal_name.split("-"))
        ds_dir = Path("tgb") / "datasets" / dir_name
        nodefile = ds_dir / f"{internal_name}_node_labels.csv"
        if stream_from_csv and nodefile.exists():
            label_dict = None
        else:
            # Fallback to the official dataset (may be memory heavy for large datasets).
            ds = NodePropPredDataset(name=internal_name, root=args.root, download=True, preprocess=True)
            label_dict = ds.full_data["node_label_dict"]  # {ts: {src_id: label_vec}}

        # Table metadata (RelBench).
        label_events_meta = _relbench_table_metadata_bytes(
            fkey_col_to_pkey_table={"src_id": "nodes"},
            pkey_col="label_event_id",
            time_col="label_ts",
        )
        label_items_meta = _relbench_table_metadata_bytes(
            fkey_col_to_pkey_table={"label_event_id": "label_events", "label_id": "labels"},
            pkey_col="item_id",
            time_col=None,
        )

        label_events_path = db_dir / "label_events.parquet"
        label_items_path = db_dir / "label_event_items.parquet"

        # Arrow schemas (normalized; avoids list columns).
        label_events_schema = pa.schema(
            [
                ("label_event_id", pa.int64()),
                ("src_id", pa.int64()),
                ("label_ts", pa.timestamp("ns", tz="UTC")),
            ]
        ).with_metadata(label_events_meta)
        label_items_schema = pa.schema(
            [
                ("item_id", pa.int64()),
                ("label_event_id", pa.int64()),
                ("label_id", pa.int64()),
                ("label_weight", pa.float32()),
            ]
        ).with_metadata(label_items_meta)

        # Stream-write.
        chunk_rows = int(args.chunk_size)
        le_writer = pq.ParquetWriter(label_events_path, label_events_schema, compression="zstd")
        li_writer = pq.ParquetWriter(label_items_path, label_items_schema, compression="zstd")
        try:
            le_id = 0
            item_id = 0
            le_buf_id: list[int] = []
            le_buf_src: list[int] = []
            le_buf_ts_ns: list[int] = []

            li_buf_item: list[int] = []
            li_buf_le: list[int] = []
            li_buf_label: list[int] = []
            li_buf_weight: list[float] = []

            def flush():
                nonlocal le_buf_id, le_buf_src, le_buf_ts_ns, li_buf_item, li_buf_le, li_buf_label, li_buf_weight
                if le_buf_id:
                    le_tbl = pa.Table.from_pydict(
                        {
                            "label_event_id": pa.array(le_buf_id, type=pa.int64()),
                            "src_id": pa.array(le_buf_src, type=pa.int64()),
                            "label_ts": pa.array(le_buf_ts_ns, type=pa.timestamp("ns", tz="UTC")),
                        },
                        schema=label_events_schema,
                    )
                    le_writer.write_table(le_tbl)
                    le_buf_id = []
                    le_buf_src = []
                    le_buf_ts_ns = []
                if li_buf_item:
                    li_tbl = pa.Table.from_pydict(
                        {
                            "item_id": pa.array(li_buf_item, type=pa.int64()),
                            "label_event_id": pa.array(li_buf_le, type=pa.int64()),
                            "label_id": pa.array(li_buf_label, type=pa.int64()),
                            "label_weight": pa.array(li_buf_weight, type=pa.float32()),
                        },
                        schema=label_items_schema,
                    )
                    li_writer.write_table(li_tbl)
                    li_buf_item = []
                    li_buf_le = []
                    li_buf_label = []
                    li_buf_weight = []

            if label_dict is not None:
                # Iterate in timestamp order for reproducibility.
                for ts in sorted(label_dict.keys()):
                    ts_s = int(_maybe_convert_years_to_unix_seconds(np.asarray([int(ts)], dtype=np.int64))[0])
                    ts_ns = ts_s * 1_000_000_000
                    node_map = label_dict[ts]
                    for src_id, label_vec in node_map.items():
                        label_vec = np.asarray(label_vec)
                        idx = np.flatnonzero(label_vec)
                        if idx.size == 0:
                            continue
                        weights = label_vec[idx].astype(np.float32, copy=False)

                        le_buf_id.append(le_id)
                        le_buf_src.append(int(src_id))
                        le_buf_ts_ns.append(ts_ns)

                        for lab, w in zip(idx.tolist(), weights.tolist()):
                            li_buf_item.append(item_id)
                            li_buf_le.append(le_id)
                            li_buf_label.append(int(lab))
                            li_buf_weight.append(float(w))
                            item_id += 1

                        le_id += 1

                        if len(le_buf_id) >= chunk_rows:
                            flush()
            else:
                # Stream directly from nodefile CSV grouped by (ts, src).
                import csv

                # For large datasets, the preprocessing stores mapping dicts on disk:
                # - <ds_dir>/ml_<name>_node.pkl: node_ids mapping (raw id -> global int id)
                # - <ds_dir>/ml_<name>_label.pkl: label mapping (raw label -> label id in [0, num_classes))
                node_map_path = ds_dir / f"ml_{internal_name}_node.pkl"
                label_map_path = ds_dir / f"ml_{internal_name}_label.pkl"
                node_ids = load_pkl(str(node_map_path)) if node_map_path.exists() else None
                label_ids = load_pkl(str(label_map_path)) if label_map_path.exists() else None

                # Column names differ slightly across datasets.
                # tgbn-trade: year,nation,trading nation,weight  (no label_map_path; ids come from node_ids)
                # tgbn-genre: ts,user_id,genre,weight (no label_map_path in current preprocessing)
                # tgbn-reddit: ts,user,subreddit,weight
                # tgbn-token: ts,user_address,token_address,weight
                with open(nodefile, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    # Determine user + label columns.
                    if "user_address" in reader.fieldnames:
                        user_col = "user_address"
                        label_col = "token_address"
                    elif "user" in reader.fieldnames:
                        user_col = "user"
                        label_col = "subreddit"
                    elif "nation" in reader.fieldnames:
                        user_col = "nation"
                        label_col = "trading nation"
                    else:
                        user_col = reader.fieldnames[1]
                        label_col = reader.fieldnames[2]

                    # Basic guardrail: require consecutive grouping by (ts,user) to avoid duplicates.
                    cur_ts = None
                    cur_user = None
                    cur_src_id: int | None = None
                    cur_ts_s: int | None = None
                    cur_items: dict[int, float] = {}
                    seen_users_this_ts: set[int] = set()

                    def flush_group():
                        nonlocal le_id, item_id, cur_items
                        if cur_src_id is None or cur_ts_s is None or not cur_items:
                            cur_items = {}
                            return
                        ts_s = int(cur_ts_s)
                        ts_s = int(_maybe_convert_years_to_unix_seconds(np.asarray([ts_s], dtype=np.int64))[0])
                        ts_ns = ts_s * 1_000_000_000

                        le_buf_id.append(le_id)
                        le_buf_src.append(int(cur_src_id))
                        le_buf_ts_ns.append(int(ts_ns))
                        for lab, w in cur_items.items():
                            li_buf_item.append(item_id)
                            li_buf_le.append(le_id)
                            li_buf_label.append(int(lab))
                            li_buf_weight.append(float(w))
                            item_id += 1
                        le_id += 1
                        cur_items = {}
                        if len(le_buf_id) >= chunk_rows:
                            flush()

                    for row in reader:
                        ts_raw = int(row.get("ts") or row.get("year") or row.get("timestamp"))
                        user_raw = row[user_col]
                        label_raw = row[label_col]
                        w = float(row.get("weight") or row.get("value") or row.get("w") or 1.0)

                        if node_ids is not None:
                            src_id = int(node_ids[user_raw])
                        else:
                            # Fall back to integer parsing (already numeric).
                            src_id = int(user_raw)

                        if label_ids is not None:
                            lab_id = int(label_ids[label_raw])
                        elif node_ids is not None and label_raw in node_ids:
                            # trade uses node_ids for both nation and trading nation.
                            lab_id = int(node_ids[label_raw])
                        else:
                            lab_id = int(label_raw)

                        key = (ts_raw, src_id)
                        if cur_ts is None:
                            cur_ts, cur_user, cur_src_id, cur_ts_s = ts_raw, src_id, src_id, ts_raw
                        if ts_raw != cur_ts or src_id != cur_user:
                            # Close previous group.
                            seen_users_this_ts.add(int(cur_user))
                            flush_group()
                            if ts_raw != cur_ts:
                                seen_users_this_ts = set()
                            else:
                                if src_id in seen_users_this_ts:
                                    raise RuntimeError(
                                        f"node_labels.csv is not grouped by (ts,user): saw user={src_id} twice at ts={ts_raw}."
                                    )
                            cur_ts, cur_user, cur_src_id, cur_ts_s = ts_raw, src_id, src_id, ts_raw

                        # Accumulate latest weight per label id.
                        cur_items[lab_id] = w

                    flush_group()

            flush()
        finally:
            le_writer.close()
            li_writer.close()

    metadata = {
        "dataset": dataset_name,
        "tgb_internal_name": internal_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cutoffs": {
            "val_timestamp_s": cutoffs.val_timestamp_s,
            "test_timestamp_s": cutoffs.test_timestamp_s,
            "val_timestamp": str(cutoffs.val_timestamp),
            "test_timestamp": str(cutoffs.test_timestamp),
        },
        "tables": sorted(db.table_dict.keys()),
        "extra": meta_extra,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "schema.dbml").write_text(_dbml_from_database(db, dataset=dataset_name), encoding="utf-8")
    print(f"Wrote RelBench Database to {db_dir}")


if __name__ == "__main__":
    main()
