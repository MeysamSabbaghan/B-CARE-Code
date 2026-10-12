#!/usr/bin/env python3
"""Compute ranking metrics from per-example JSONL outputs (no hardcoded paper numbers).

Joins gold + base predictions + risk channels and writes metrics JSON/CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcare.evaluation.aurc import excess_aurc  # noqa: E402
from bcare.evaluation.metrics import (  # noqa: E402
    full_ranking_metrics,
    paired_bootstrap_delta_auroc,
)
from bcare.risk import evidencerisk_as, llamaprob_risk  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def index_by_id(rows: list[dict], key: str = "example_id") -> dict[str, dict]:
    return {str(r[key]): r for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True, help="Prepared CSV with gold")
    ap.add_argument("--base-preds", type=Path, required=True, help="Base prediction JSONL")
    ap.add_argument("--pbce", type=Path, default=None, help="B-CARE PBCE evidence JSONL")
    ap.add_argument("--llamaprob", type=Path, default=None, help="LlamaProb closed-set JSONL")
    ap.add_argument("--risk-jsonl", type=Path, default=None, help="Precomputed risk JSONL with 'risk'")
    ap.add_argument(
        "--suite-jsonl",
        type=Path,
        default=None,
        help="Local baseline suite JSONL; use with --risk-field (e.g. MSP_risk, p_true_risk, SemanticEntropy, EVSE)",
    )
    ap.add_argument("--risk-field", type=str, default=None)
    ap.add_argument("--method-name", type=str, default="EvidenceRisk")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--bootstrap-B", type=int, default=10000)
    ap.add_argument("--bootstrap-seed", type=int, default=2026100617)
    ap.add_argument("--compare-risk-jsonl", type=Path, default=None)
    ap.add_argument("--compare-name", type=str, default="baseline")
    args = ap.parse_args()

    gold = {str(r["example_id"]): r for r in csv.DictReader(args.manifest.open())}
    base = index_by_id(load_jsonl(args.base_preds))
    pbce = index_by_id(load_jsonl(args.pbce)) if args.pbce else {}
    lp = index_by_id(load_jsonl(args.llamaprob)) if args.llamaprob else {}
    pre = index_by_id(load_jsonl(args.risk_jsonl)) if args.risk_jsonl else {}
    suite = index_by_id(load_jsonl(args.suite_jsonl)) if args.suite_jsonl else {}
    if args.suite_jsonl and not args.risk_field:
        raise SystemExit("--suite-jsonl requires --risk-field")

    errors, risks, ids = [], [], []
    per_ex = []
    for eid, g in gold.items():
        if eid not in base:
            continue
        yhat = base[eid].get("base_pred")
        gold_lab = g["gold"]
        err = int(yhat != gold_lab) if yhat is not None else 1
        labels = tuple(g.get("label_space", "FAVOR|AGAINST|NONE").split("|"))
        risk = None
        if eid in suite and args.risk_field:
            if args.risk_field not in suite[eid] or suite[eid][args.risk_field] is None:
                continue
            risk = float(suite[eid][args.risk_field])
        elif eid in pre and "risk" in pre[eid]:
            risk = float(pre[eid]["risk"])
        elif args.method_name in ("EvidenceRisk", "B-CARE", "PBCE") and eid in pbce:
            ev = pbce[eid].get("evidence")
            if ev and yhat and yhat in labels:
                try:
                    risk = evidencerisk_as(ev, yhat, labels)
                except KeyError:
                    risk = None
        elif args.method_name == "LlamaProb" and eid in lp and yhat:
            P = lp[eid].get("P")
            if P:
                risk = llamaprob_risk(P, yhat)
        if risk is None:
            continue
        errors.append(err)
        risks.append(risk)
        ids.append(eid)
        per_ex.append(
            {
                "example_id": eid,
                "gold": gold_lab,
                "base_pred": yhat,
                "error": err,
                "risk": risk,
                "method": args.method_name,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    err_arr = np.asarray(errors)
    risk_arr = np.asarray(risks, dtype=float)
    metrics = full_ranking_metrics(err_arr, risk_arr)
    metrics.update(excess_aurc(err_arr, risk_arr, np.arange(len(err_arr))))
    metrics["method"] = args.method_name
    (args.out_dir / f"metrics_{args.method_name}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    with (args.out_dir / f"per_example_{args.method_name}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_ex[0].keys()) if per_ex else ["example_id"])
        w.writeheader()
        w.writerows(per_ex)

    if args.compare_risk_jsonl:
        other = index_by_id(load_jsonl(args.compare_risk_jsonl))
        ra, rb = [], []
        ee = []
        for eid, err, r in zip(ids, errors, risks):
            if eid not in other or "risk" not in other[eid]:
                continue
            ee.append(err)
            ra.append(r)
            rb.append(float(other[eid]["risk"]))
        boot = paired_bootstrap_delta_auroc(
            np.asarray(ee), np.asarray(ra), np.asarray(rb), B=args.bootstrap_B, seed=args.bootstrap_seed
        )
        boot["method_a"] = args.method_name
        boot["method_b"] = args.compare_name
        (args.out_dir / f"bootstrap_{args.method_name}_vs_{args.compare_name}.json").write_text(
            json.dumps(boot, indent=2) + "\n"
        )
        print(json.dumps(boot, indent=2))

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
