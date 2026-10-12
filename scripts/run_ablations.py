#!/usr/bin/env python3
"""Offline ablations over frozen PBCE evidence + base predictions.

Computes: A−S, −S, A, evidence-share, softmax-support, lexicographic, LD.
Writes per-example risks (no gold required for risk; gold optional for later eval).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcare.risk import (  # noqa: E402
    alternative_only_risk,
    evidence_label_disagreement,
    evidence_share_risk,
    evidencerisk_as,
    lexicographic_risk,
    softmax_support_risk,
    support_only_risk,
)


def load_jsonl(path: Path) -> dict[str, dict]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[str(r["example_id"])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--base-preds", type=Path, required=True)
    ap.add_argument("--pbce", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    gold = {str(r["example_id"]): r for r in csv.DictReader(args.manifest.open())}
    base = load_jsonl(args.base_preds)
    pbce = load_jsonl(args.pbce)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for eid, g in gold.items():
            if eid not in base or eid not in pbce:
                continue
            yhat = base[eid].get("base_pred")
            ev = pbce[eid].get("evidence")
            if not yhat or not ev:
                continue
            labels = tuple(g.get("label_space", "FAVOR|AGAINST|NONE").split("|"))
            if yhat not in labels:
                continue
            rec = {
                "example_id": eid,
                "base_pred": yhat,
                "gold": g.get("gold"),
                "error": int(yhat != g.get("gold")),
                "ComparativeRisk": evidencerisk_as(ev, yhat, labels),
                "SupportOnlyRisk": support_only_risk(ev, yhat, labels),
                "AlternativeOnlyRisk": alternative_only_risk(ev, yhat, labels),
                "EvidenceShareRisk": evidence_share_risk(ev, yhat, labels),
                "SoftmaxSupportRisk": softmax_support_risk(ev, yhat, labels),
                "LexicographicRisk": lexicographic_risk(ev, yhat, labels),
                "LD": evidence_label_disagreement(ev, yhat, labels),
            }
            f.write(json.dumps(rec) + "\n")
            n += 1
    print(f"wrote {n} ablation rows → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
