#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from tgb.linkproppred.dataset import LinkPropPredDataset


@dataclass(frozen=True)
class SplitCutoffs:
    val_timestamp: int
    test_timestamp: int


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _detect_bipartite_offset(src: np.ndarray, dst: np.ndarray) -> int | None:
    """
    Detect the common TGB/JODIE-style bipartite encoding where dst ids are offset
    by (max(src) + 1) to avoid collisions in a single id space.
    """
    if src.size == 0 or dst.size == 0:
        return None
    src_max = int(src.max())
    expected_offset = src_max + 1
    if int(dst.min()) >= expected_offset:
        return expected_offset
    return None


def _compute_cutoffs(ts: np.ndarray, train_mask: np.ndarray, val_mask: np.ndarray, test_mask: np.ndarray) -> SplitCutoffs:
    if not (train_mask.any() and val_mask.any() and test_mask.any()):
        raise ValueError("Expected non-empty train/val/test masks.")
    val_timestamp = int(ts[val_mask].min())
    test_timestamp = int(ts[test_mask].min())
    return SplitCutoffs(val_timestamp=val_timestamp, test_timestamp=test_timestamp)


def _write_dbml_schema(
    out_path: Path,
    *,
    dataset: str,
    bipartite_offset: int | None,
    msg_dim: int,
    msg_note: str | None = None,
) -> None:
    lines: list[str] = []
    lines.append(f"// Auto-generated schema for {dataset}")
    lines.append(f"// Generated at {datetime.now(timezone.utc).isoformat()}")
    if msg_note:
        lines.append(f"// Edge features: {msg_note}")
    lines.append("")

    if bipartite_offset is not None:
        lines.append("Table users {")
        lines.append("  user_id bigint [pk]")
        lines.append("}")
        lines.append("")
        lines.append("Table pages {")
        lines.append("  page_id bigint [pk]")
        lines.append("}")
        lines.append("")
        lines.append("Table user_edited_page {")
        lines.append("  event_id bigint [pk]")
        lines.append("  user_id bigint [ref: > users.user_id]")
        lines.append("  page_id bigint [ref: > pages.page_id]")
        lines.append("  event_ts bigint")
        lines.append("  weight float")
        lines.append("}")
    else:
        lines.append("Table nodes {")
        lines.append("  node_id bigint [pk]")
        lines.append("}")
        lines.append("")
        lines.append("Table events {")
        lines.append("  event_id bigint [pk]")
        lines.append("  src_id bigint [ref: > nodes.node_id]")
        lines.append("  dst_id bigint [ref: > nodes.node_id]")
        lines.append("  event_ts bigint")
        lines.append("  weight float")
        lines.append("}")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_tgbl_wiki_v2(
    dataset: LinkPropPredDataset,
    out_dir: Path,
    *,
    export_dataset_name: str,
    save_msg_npy: bool,
) -> None:
    full = dataset.full_data
    src = np.asarray(full["sources"], dtype=np.int64)
    dst = np.asarray(full["destinations"], dtype=np.int64)
    ts = np.asarray(full["timestamps"], dtype=np.int64)
    msg = np.asarray(full["edge_feat"])
    weight = np.asarray(full["w"], dtype=np.float32)

    train_mask = np.asarray(dataset.train_mask, dtype=bool)
    val_mask = np.asarray(dataset.val_mask, dtype=bool)
    test_mask = np.asarray(dataset.test_mask, dtype=bool)

    bipartite_offset = _detect_bipartite_offset(src, dst)
    cutoffs = _compute_cutoffs(ts, train_mask, val_mask, test_mask)

    tables_dir = out_dir / "tables"
    _ensure_dir(tables_dir)

    # Clean up legacy filenames from earlier exports.
    for legacy in (
        "src_nodes.csv",
        "dst_nodes.csv",
        "src_dst_events.csv",
    ):
        legacy_path = tables_dir / legacy
        if legacy_path.exists():
            legacy_path.unlink()

    # --- Node tables
    if bipartite_offset is not None:
        n_src = int(src.max()) + 1
        n_dst = int(dst.max() - bipartite_offset) + 1

        users = pd.DataFrame(
            {
                "user_id": np.arange(n_src, dtype=np.int64),
            }
        )
        pages = pd.DataFrame(
            {
                "page_id": np.arange(n_dst, dtype=np.int64),
            }
        )
        users.to_csv(tables_dir / "users.csv", index=False)
        pages.to_csv(tables_dir / "pages.csv", index=False)
    else:
        n_nodes = int(max(src.max(), dst.max())) + 1
        nodes = pd.DataFrame({"node_id": np.arange(n_nodes, dtype=np.int64)})
        nodes.to_csv(tables_dir / "nodes.csv", index=False)

    # --- Edge features (kept out of the relational schema by default, like thgl-software)
    msg_dim = 0 if msg.ndim != 2 else int(msg.shape[1])
    msg_path = None
    if save_msg_npy and msg_dim > 0:
        msg_path = tables_dir / "user_edited_page_msg.npy"
        np.save(msg_path, msg.astype(np.float32, copy=False))

    # --- Event table (streaming writer to avoid huge intermediate DataFrames)
    event_path = tables_dir / ("user_edited_page.csv" if bipartite_offset is not None else "events.csv")
    with event_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "event_id",
            "user_id" if bipartite_offset is not None else "src_id",
            "page_id" if bipartite_offset is not None else "dst_id",
            "event_ts",
            "weight",
        ]
        writer.writerow(header)

        if bipartite_offset is not None:
            dst_local = dst - int(bipartite_offset)
            for event_id in range(src.shape[0]):
                writer.writerow(
                    [
                        event_id,
                        int(src[event_id]),
                        int(dst_local[event_id]),
                        int(ts[event_id]),
                        float(weight[event_id]),
                    ]
                )
        else:
            for event_id in range(src.shape[0]):
                writer.writerow(
                    [
                        event_id,
                        int(src[event_id]),
                        int(dst[event_id]),
                        int(ts[event_id]),
                        float(weight[event_id]),
                    ]
                )

    # --- Schema + metadata
    msg_note = f"{msg_dim}-dim float32 vector saved separately as .npy" if msg_dim > 0 else None
    _write_dbml_schema(
        out_dir / "schema.dbml",
        dataset=export_dataset_name,
        bipartite_offset=bipartite_offset,
        msg_dim=msg_dim,
        msg_note=msg_note,
    )

    metadata = {
        "dataset": export_dataset_name,
        "tgb_internal_name": dataset.name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir),
        "num_events": int(src.shape[0]),
        "msg_dim": msg_dim,
        "bipartite_offset": int(bipartite_offset) if bipartite_offset is not None else None,
        "cutoffs": {"val_timestamp": cutoffs.val_timestamp, "test_timestamp": cutoffs.test_timestamp},
        "split_counts": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "tables": {
            "nodes": "tables/users.csv + tables/pages.csv" if bipartite_offset is not None else "tables/nodes.csv",
            "events": str(event_path.relative_to(out_dir)),
            "edge_features": str(msg_path.relative_to(out_dir)) if msg_path is not None else None,
        },
    }
    # Note: repo .gitignore ignores *.json by default; this is still useful locally.
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export TGB datasets into relational (table) format.")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. tgbl-wiki-v2")
    parser.add_argument("--root", default="datasets", help="TGB dataset root (relative to repo).")
    parser.add_argument("--out_dir", default="exports", help="Output root directory (relative to repo).")
    parser.add_argument(
        "--save_msg_npy",
        action="store_true",
        help="Save dense edge features to tables/user_edited_page_msg.npy (kept out of schema.dbml).",
    )
    args = parser.parse_args()

    dataset_name = args.dataset
    root = args.root
    out_root = Path(args.out_dir)
    out_dir = out_root / dataset_name
    _ensure_dir(out_dir)

    if dataset_name not in ("tgbl-wiki-v2", "tgbl-wiki"):
        raise ValueError(
            f"This exporter currently implements tgbl-wiki-v2 only (got {dataset_name}). "
            "Extend scripts/export_relational.py to support more datasets."
        )

    # TGB 2.0 docs sometimes refer to "tgbl-wiki-v2", but the library’s dataset id is "tgbl-wiki".
    internal_name = "tgbl-wiki" if dataset_name == "tgbl-wiki-v2" else dataset_name
    tgb_dataset = LinkPropPredDataset(name=internal_name, root=root, download=True)
    _export_tgbl_wiki_v2(
        tgb_dataset,
        out_dir,
        export_dataset_name=dataset_name,
        save_msg_npy=bool(args.save_msg_npy),
    )
    print(f"Wrote relational export to: {out_dir}")


if __name__ == "__main__":
    main()
