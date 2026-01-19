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
    return None


def _compute_cutoffs(ts: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> Cutoffs:
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Expected non-empty train/val/test masks.")
    return Cutoffs(val_timestamp_s=int(ts[val_mask].min()), test_timestamp_s=int(ts[test_mask].min()))


def _ts_to_utc(ts_s: np.ndarray) -> pd.Series:
    return pd.Series(pd.to_datetime(ts_s.astype(np.int64, copy=False), unit="s", utc=True))

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
    ds = NodePropPredDataset(name=name, root=root, download=True)
    full = ds.full_data

    src = np.asarray(full["sources"], dtype=np.int64)
    dst = np.asarray(full["destinations"], dtype=np.int64)
    ts_s = np.asarray(full["timestamps"], dtype=np.int64)

    # NodeProp datasets use edge_feat for attributes and store edge_label=1.
    # We still keep a scalar "weight" if the edge_feat is 1-d, else omit here.
    edge_feat = np.asarray(full["edge_feat"])
    weight: Optional[np.ndarray]
    if edge_feat.ndim == 2 and edge_feat.shape[1] == 1:
        weight = edge_feat[:, 0].astype(np.float32, copy=False)
    else:
        weight = None

    train_mask = np.asarray(ds.train_mask, dtype=bool)
    val_mask = np.asarray(ds.val_mask, dtype=bool)
    test_mask = np.asarray(ds.test_mask, dtype=bool)
    cutoffs = _compute_cutoffs(ts_s, train_mask, val_mask, test_mask)

    bipartite_offset = _detect_bipartite_offset(src, dst)
    extra_meta: dict = {
        "bipartite_offset": int(bipartite_offset) if bipartite_offset is not None else None,
        "num_classes": int(ds.num_classes),
        "label_timestamps": int(len(ds.label_ts)) if hasattr(ds, "label_ts") else None,
    }

    ts = _ts_to_utc(ts_s)
    if bipartite_offset is not None:
        n_src = int(src.max()) + 1
        n_dst = int(dst.max() - bipartite_offset) + 1
        src_nodes = pd.DataFrame({"src_id": np.arange(n_src, dtype=np.int64)})
        dst_nodes = pd.DataFrame({"dst_id": np.arange(n_dst, dtype=np.int64)})
        events_dict = {
            "event_id": np.arange(src.shape[0], dtype=np.int64),
            "src_id": src,
            "dst_id": (dst - bipartite_offset).astype(np.int64),
            "event_ts": ts,
        }
        if weight is not None:
            events_dict["weight"] = weight
        events = pd.DataFrame(events_dict)
        db = Database(
            {
                "src_nodes": Table(df=src_nodes, pkey_col="src_id", fkey_col_to_pkey_table={}, time_col=None),
                "dst_nodes": Table(df=dst_nodes, pkey_col="dst_id", fkey_col_to_pkey_table={}, time_col=None),
                "events": Table(df=events, pkey_col="event_id", time_col="event_ts", fkey_col_to_pkey_table={"src_id": "src_nodes", "dst_id": "dst_nodes"}),
            }
        )
        return db, cutoffs, extra_meta

    n_nodes = int(max(src.max(), dst.max())) + 1
    nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64)})
    events_dict = {
        "event_id": np.arange(src.shape[0], dtype=np.int64),
        "src_id": src,
        "dst_id": dst,
        "event_ts": ts,
    }
    if weight is not None:
        events_dict["weight"] = weight
    events = pd.DataFrame(events_dict)
    db = Database(
        {
            "nodes": Table(df=nodes, pkey_col="node_id", fkey_col_to_pkey_table={}, time_col=None),
            "events": Table(df=events, pkey_col="event_id", time_col="event_ts", fkey_col_to_pkey_table={"src_id": "nodes", "dst_id": "nodes"}),
        }
    )
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
