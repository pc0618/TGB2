from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from relbench.base import Database, Dataset, Table


@dataclass(frozen=True)
class ExportPaths:
    export_dir: Path
    tables_dir: Path
    metadata_path: Path


def _load_paths(export_dir: str | Path) -> ExportPaths:
    export_dir = Path(export_dir)
    return ExportPaths(
        export_dir=export_dir,
        tables_dir=export_dir / "tables",
        metadata_path=export_dir / "metadata.json",
    )


def _to_utc_timestamp_from_seconds(seconds: int) -> pd.Timestamp:
    # Many TGB datasets already store UNIX seconds; some store “relative seconds”.
    # For RelBench we only need consistent ordering and arithmetic, so we map to UTC.
    return pd.to_datetime(int(seconds), unit="s", utc=True)


class TGBExportedDataset(Dataset):
    r"""A RelBench Dataset backed by local exports.

    Supported export formats:
    - `scripts/export_to_relbench.py` (preferred): `<export_dir>/db/*.parquet` + `metadata.json`
    - `scripts/export_relational.py` (legacy): `<export_dir>/tables/*.csv` + `metadata.json`
    """

    def __init__(self, export_dir: str | Path, cache_dir: Optional[str] = None) -> None:
        super().__init__(cache_dir=cache_dir)
        self._paths = _load_paths(export_dir)
        meta = json.loads(self._paths.metadata_path.read_text(encoding="utf-8"))

        self.name = meta.get("dataset", self._paths.export_dir.name)
        cutoffs = meta.get("cutoffs", {})
        if "val_timestamp_s" in cutoffs and "test_timestamp_s" in cutoffs:
            self.val_timestamp = _to_utc_timestamp_from_seconds(cutoffs["val_timestamp_s"])
            self.test_timestamp = _to_utc_timestamp_from_seconds(cutoffs["test_timestamp_s"])
        else:
            self.val_timestamp = _to_utc_timestamp_from_seconds(cutoffs["val_timestamp"])
            self.test_timestamp = _to_utc_timestamp_from_seconds(cutoffs["test_timestamp"])

    def make_db(self) -> Database:
        parquet_db_dir = self._paths.export_dir / "db"
        if parquet_db_dir.exists() and any(parquet_db_dir.glob("*.parquet")):
            return Database.load(parquet_db_dir)

        # Legacy CSV-backed export: currently only implemented for tgbl-wiki-v2.
        name = self.name
        tables = self._paths.tables_dir
        if name == "tgbl-wiki-v2":
            users = pd.read_csv(tables / "users.csv")
            pages = pd.read_csv(tables / "pages.csv")
            events = pd.read_csv(tables / "user_edited_page.csv")
            events["event_ts"] = pd.to_datetime(events["event_ts"].astype(np.int64), unit="s", utc=True)
            return Database(
                table_dict={
                    "users": Table(df=users, pkey_col="user_id", fkey_col_to_pkey_table={}, time_col=None),
                    "pages": Table(df=pages, pkey_col="page_id", fkey_col_to_pkey_table={}, time_col=None),
                    "user_edited_page": Table(
                        df=events,
                        pkey_col="event_id",
                        time_col="event_ts",
                        fkey_col_to_pkey_table={"user_id": "users", "page_id": "pages"},
                    ),
                }
            )

        raise ValueError(
            f"Export at '{self._paths.export_dir}' does not contain a RelBench parquet db, "
            f"and no legacy CSV reader exists for dataset '{name}'."
        )


def get_exported_dataset(name: str, *, exports_root: str | Path = "exports") -> TGBExportedDataset:
    export_dir = Path(exports_root) / name
    return TGBExportedDataset(export_dir=export_dir, cache_dir=str(export_dir))
