import numpy as np
import pandas as pd
import torch

FEATURE_VERSION = "ageact_v1"
SEVEN_DAYS_SECONDS = 7 * 24 * 3600


def add_temporal_features(data: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Append temporal age/activity features to data.msg."""
    features = _load_or_compute_features(data, num_nodes)
    features = features.to(data.msg.device)
    if data.msg.dtype != torch.float32:
        data.msg = data.msg.float()
    data.msg = torch.cat([data.msg, features], dim=1)
    return data


def _load_or_compute_features(data: torch.Tensor, num_nodes: int) -> torch.Tensor:
    # No on-disk caching here; caller is expected to cache the entire TemporalData.
    return _compute_temporal_features(data, num_nodes)


def _compute_temporal_features(data: torch.Tensor, num_nodes: int) -> torch.Tensor:
    ts = data.t.cpu().numpy().astype(np.int64)
    src = data.src.cpu().numpy().astype(np.int64)
    dst = data.dst.cpu().numpy().astype(np.int64)

    df = pd.DataFrame({
        "ts": ts,
        "src": src,
        "dst": dst,
    })
    df["ts_dt"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # Node ages (days since first observation)
    df["src_first_ts"] = df.groupby("src")["ts"].transform("min")
    df["dst_first_ts"] = df.groupby("dst")["ts"].transform("min")
    df["src_age_days"] = (df["ts"] - df["src_first_ts"]) / 86400.0
    df["dst_age_days"] = (df["ts"] - df["dst_first_ts"]) / 86400.0

    # Hours since previous event for src/dst
    df["src_hours_since_prev"] = (
        df.groupby("src")["ts_dt"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .fillna(-1.0)
    )
    df["dst_hours_since_prev"] = (
        df.groupby("dst")["ts_dt"]
        .diff()
        .dt.total_seconds()
        .div(3600.0)
        .fillna(-1.0)
    )

    # Rolling 7-day event counts per src/dst (excluding current event)
    src_roll = (
        df.groupby("src")
        .rolling(window="7D", on="ts_dt")["ts_dt"]
        .count()
        - 1
    )
    df["src_events_7d"] = src_roll.reset_index(level=0, drop=True).fillna(0.0)

    dst_roll = (
        df.groupby("dst")
        .rolling(window="7D", on="ts_dt")["ts_dt"]
        .count()
        - 1
    )
    df["dst_events_7d"] = dst_roll.reset_index(level=0, drop=True).fillna(0.0)

    feature_cols = [
        "src_age_days",
        "dst_age_days",
        "src_hours_since_prev",
        "dst_hours_since_prev",
        "src_events_7d",
        "dst_events_7d",
    ]
    feature_array = df[feature_cols].to_numpy(dtype=np.float32)
    return torch.from_numpy(feature_array)
