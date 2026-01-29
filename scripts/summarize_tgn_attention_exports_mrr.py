from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", required=True)
    args = ap.parse_args()

    log_dir = Path(args.log_dir)
    logs = sorted(log_dir.glob("*.txt"))
    if not logs:
        raise SystemExit(f"No .txt logs under {log_dir}")

    pat = re.compile(r"Best val_mrr@(\d+)=([0-9.]+)\s+Best test_mrr@\1=([0-9.]+)")

    print("| dataset | val_mrr@K | test_mrr@K | log |")
    print("|---|---:|---:|---|")
    for p in logs:
        text = p.read_text(encoding="utf-8", errors="replace")
        m = pat.search(text)
        if not m:
            continue
        k = int(m.group(1))
        val = float(m.group(2))
        test = float(m.group(3))
        ds = p.stem
        print(f"| {ds} | {val:.4f} | {test:.4f} | {p.name} |")


if __name__ == "__main__":
    main()

