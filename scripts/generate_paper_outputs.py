#!/usr/bin/env python3
"""Aggregate per-setting metrics JSON files into summary tables (computed, not hardcoded).

Expects a directory tree like:
  outputs/metrics/{Dataset}_{Base}/{method}/metrics_*.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-root", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    rows = []
    for path in sorted(args.metrics_root.rglob("metrics_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        setting = path.parent.parent.name if path.parent.parent != args.metrics_root else path.parent.name
        rows.append(
            {
                "setting": setting,
                "method": data.get("method") or path.stem.replace("metrics_", ""),
                "AUROC": data.get("AUROC"),
                "AUPRC": data.get("AUPRC"),
                "ErrorCapture@10": data.get("ErrorCapture@10"),
                "ErrorCapture@20": data.get("ErrorCapture@20"),
                "ErrorCapture@30": data.get("ErrorCapture@30"),
                "n": data.get("n"),
                "n_errors": data.get("n_errors"),
                "source": str(path),
            }
        )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"No metrics_*.json under {args.metrics_root}")
        return 1
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows → {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
