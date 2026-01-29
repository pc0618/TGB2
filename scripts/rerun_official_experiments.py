#!/usr/bin/env python3

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunSpec:
    name: str
    argv: list[str]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _venv_python(repo_root: Path) -> str:
    py = repo_root / ".venv" / "bin" / "python"
    return str(py)


def _write_header(fh, *, dataset: str) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    fh.write(f"# Dataset: {dataset}\n")
    fh.write(f"# Timestamp: {now}\n")
    fh.write(f"# Host: {platform.node()}\n")
    fh.write(f"# Platform: {platform.platform()}\n")
    fh.write("\n")
    fh.flush()


def _run_one(*, fh, env: dict[str, str], repo_root: Path, spec: RunSpec) -> int:
    fh.write("\n")
    fh.write("=" * 80 + "\n")
    fh.write(f"MODEL: {spec.name}\n")
    fh.write(f"CMD: {' '.join(spec.argv)}\n")
    fh.write("=" * 80 + "\n")
    fh.flush()

    proc = subprocess.Popen(
        spec.argv,
        cwd=str(repo_root),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        text=True,
    )
    rc = int(proc.wait())
    fh.write(f"\n# Exit code: {rc}\n")
    fh.flush()
    return rc


def main() -> int:
    repo_root = _repo_root()
    py = _venv_python(repo_root)
    if not Path(py).exists():
        raise SystemExit(f"Missing venv python at {py}. Create it with `python -m venv .venv` and install deps.")

    relbench_cache_root = os.environ.get("REL_TGB_CACHE_ROOT", "/home/pc0618/tmp_relbench_cache_official")
    exports_root = os.environ.get("REL_TGB_EXPORTS_ROOT", "relbench_exports")

    # Budget guardrails (override via env for longer runs).
    epochs = int(os.environ.get("REL_TGB_EPOCHS", "1"))
    max_train_edges = int(os.environ.get("REL_TGB_MAX_TRAIN_EDGES", "50000"))
    max_eval_edges = int(os.environ.get("REL_TGB_MAX_EVAL_EDGES", "5000"))
    max_train_events = int(os.environ.get("REL_TGB_MAX_TRAIN_EVENTS", "50000"))
    max_eval_events = int(os.environ.get("REL_TGB_MAX_EVAL_EVENTS", "5000"))

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = repo_root / "logs" / f"official_rerun_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CPU guardrails.
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("NUMEXPR_NUM_THREADS", "8")

    tgbl = [
        "tgbl-wiki",
        "tgbl-wiki-v2",
        "tgbl-review",
        "tgbl-review-v2",
        "tgbl-coin",
        "tgbl-comment",
        "tgbl-flight",
    ]
    thgl = ["thgl-software", "thgl-forum", "thgl-github", "thgl-myket"]
    tgbn = ["tgbn-trade", "tgbn-genre", "tgbn-reddit", "tgbn-token"]
    family = os.environ.get("REL_TGB_FAMILY", "all").strip().lower()

    def _run_tgbl(dataset: str, fh) -> None:
        specs = [
            RunSpec(
                "GraphSAGE (linkpred; relbench exports; official TGB one-vs-many)",
                [
                    py,
                    "baselines/graphsage_linkpred.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--model",
                    "sage",
                    "--eval_mode",
                    "tgb",
                    "--relbench_cache_root",
                    relbench_cache_root,
                    "--device",
                    "cpu",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "1024",
                    "--undirected",
                    "--max_train_edges",
                    str(max_train_edges),
                    "--max_eval_edges",
                    str(max_eval_edges),
                    "--wandb_mode",
                    "disabled",
                ],
            ),
            RunSpec(
                "RelEventSAGE (linkpred; PK/FK row-as-node; official TGB one-vs-many)",
                [
                    py,
                    "baselines/relational_eventsage_linkpred.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--eval_mode",
                    "tgb",
                    "--relbench_cache_root",
                    relbench_cache_root,
                    "--device",
                    "cpu",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "1024",
                    "--max_train_edges",
                    str(max_train_edges),
                    "--max_eval_edges",
                    str(max_eval_edges),
                    "--wandb_mode",
                    "disabled",
                ],
            ),
        ]
        specs.append(
            RunSpec(
                "GAT (linkpred; relbench exports; official TGB one-vs-many)",
                [
                    py,
                    "baselines/graphsage_linkpred.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--model",
                    "gat",
                    "--eval_mode",
                    "tgb",
                    "--relbench_cache_root",
                    relbench_cache_root,
                    "--device",
                    "cpu",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "1024",
                    "--undirected",
                    "--max_train_edges",
                    str(max_train_edges),
                    "--max_eval_edges",
                    str(max_eval_edges),
                    "--wandb_mode",
                    "disabled",
                ],
            )
        )
        for spec in specs:
            _run_one(fh=fh, env=env, repo_root=repo_root, spec=spec)

    def _run_thgl(dataset: str, fh) -> None:
        specs = [
            RunSpec(
                "GraphSAGE (THGL global; official TGB one-vs-many)",
                [
                    py,
                    "baselines/graphsage_linkpred_thgl_global.py",
                    "--dataset",
                    dataset,
                    "--relbench_cache_root",
                    relbench_cache_root,
                    "--device",
                    "cpu",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "1024",
                    "--undirected",
                    "--max_train_edges",
                    str(max_train_edges),
                    "--max_eval_edges",
                    str(max_eval_edges),
                ],
            ),
            RunSpec(
                "RelEventSAGE (THGL linkpred; hetero; official TGB one-vs-many)",
                [
                    py,
                    "baselines/relational_eventsage_linkpred_thgl.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--eval_mode",
                    "tgb",
                    "--relbench_cache_root",
                    relbench_cache_root,
                    "--device",
                    "cpu",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "512",
                    "--max_train_edges",
                    str(max_train_edges),
                    "--max_eval_edges",
                    str(max_eval_edges),
                    "--wandb_mode",
                    "disabled",
                ],
            ),
            RunSpec(
                "TGN + GraphAttention (raw TGB loader; official one-vs-many)",
                [
                    py,
                    "examples/linkproppred/thgl-forum/tgn.py",
                    "--data",
                    dataset,
                    "--eval_mode",
                    "tgb",
                    "--num_run",
                    "1",
                    "--num_epoch",
                    str(epochs),
                    "--patience",
                    "2",
                    "--tolerance",
                    "1e-6",
                    "--num_workers",
                    "0",
                    "--bs",
                    "200",
                    "--lr",
                    "1e-4",
                    "--num_neg_samples",
                    "1",
                    "--split_frac",
                    "1.0",
                    "--max_train_events",
                    str(max_train_edges),
                    "--max_val_events",
                    str(max_eval_edges),
                    "--max_test_events",
                    str(max_eval_edges),
                ],
            ),
        ]
        for spec in specs:
            _run_one(fh=fh, env=env, repo_root=repo_root, spec=spec)

    def _run_tgbn(dataset: str, fh) -> None:
        specs = [
            RunSpec(
                "GraphSAGE (nodeprop; all-label scoring; NDCG@10)",
                [
                    py,
                    "baselines/graphsage_nodeprop.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--device",
                    "cpu",
                    "--model",
                    "sage",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "512",
                    "--max_train_events",
                    str(max_train_events),
                    "--max_val_events",
                    str(max_eval_events),
                    "--max_test_events",
                    str(max_eval_events),
                    "--max_eval_events",
                    str(max_eval_events),
                    "--ndcg_k",
                    "10",
                    "--max_rss_gb",
                    "50",
                    "--wandb_mode",
                    "disabled",
                ],
            ),
            RunSpec(
                "GAT (nodeprop; sampled neighbor attention; NDCG@10)",
                [
                    py,
                    "baselines/graphsage_nodeprop.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--device",
                    "cpu",
                    "--model",
                    "gat",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "512",
                    "--max_train_events",
                    str(max_train_events),
                    "--max_val_events",
                    str(max_eval_events),
                    "--max_test_events",
                    str(max_eval_events),
                    "--max_eval_events",
                    str(max_eval_events),
                    "--ndcg_k",
                    "10",
                    "--max_rss_gb",
                    "50",
                    "--wandb_mode",
                    "disabled",
                ],
            ),
            RunSpec(
                "EmbeddingOnly (nodeprop; no message passing; NDCG@10)",
                [
                    py,
                    "baselines/graphsage_nodeprop.py",
                    "--dataset",
                    dataset,
                    "--exports_root",
                    exports_root,
                    "--device",
                    "cpu",
                    "--model",
                    "emb",
                    "--epochs",
                    str(epochs),
                    "--batch_size",
                    "512",
                    "--max_train_events",
                    str(max_train_events),
                    "--max_val_events",
                    str(max_eval_events),
                    "--max_test_events",
                    str(max_eval_events),
                    "--max_eval_events",
                    str(max_eval_events),
                    "--ndcg_k",
                    "10",
                    "--max_rss_gb",
                    "50",
                    "--wandb_mode",
                    "disabled",
                ],
            ),
        ]
        for spec in specs:
            _run_one(fh=fh, env=env, repo_root=repo_root, spec=spec)

    if family in {"all", "tgbl"}:
        for ds in tgbl:
            log_path = out_dir / f"{ds}.log"
            with log_path.open("w", encoding="utf-8") as fh:
                _write_header(fh, dataset=ds)
                _run_tgbl(ds, fh)

    if family in {"all", "thgl"}:
        for ds in thgl:
            log_path = out_dir / f"{ds}.log"
            with log_path.open("w", encoding="utf-8") as fh:
                _write_header(fh, dataset=ds)
                _run_thgl(ds, fh)

    if family in {"all", "tgbn"}:
        for ds in tgbn:
            log_path = out_dir / f"{ds}.log"
            with log_path.open("w", encoding="utf-8") as fh:
                _write_header(fh, dataset=ds)
                _run_tgbn(ds, fh)

    print(f"Wrote logs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
