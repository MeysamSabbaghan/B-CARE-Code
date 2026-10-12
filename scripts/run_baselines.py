#!/usr/bin/env python3
"""Run offline baselines/controls that do not need extra model calls.

Supported here:
  --method ld|cbd|random|llamaprob_join|same_model_ld|llama_ld

GPU baselines (MSP, PE, DSV, p(True), LE, VR, SE, EVSE, Same-Model evidence)
are executed via scripts/run_local_baseline_suite.py (same frozen configs).
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

from bcare.risk import (  # noqa: E402
    cross_base_disagreement,
    evidence_label_disagreement,
    llama_label_disagreement,
    llamaprob_risk,
    same_model_label_disagreement,
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
    ap.add_argument(
        "--method",
        required=True,
        choices=["ld", "cbd", "random", "llamaprob_join", "same_model_ld", "llama_ld"],
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--base-preds", type=Path, required=True, help="Predictions for evaluated base")
    ap.add_argument("--pbce", type=Path, default=None)
    ap.add_argument("--other-base-preds", type=Path, nargs=2, default=None, help="For CBD")
    ap.add_argument("--llamaprob", type=Path, default=None)
    ap.add_argument("--hard-labels", type=Path, default=None, help="JSONL with hard label field")
    ap.add_argument("--hard-label-field", type=str, default="hard_label")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=2026092800)
    args = ap.parse_args()

    gold = {str(r["example_id"]): r for r in csv.DictReader(args.manifest.open())}
    base = load_jsonl(args.base_preds)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.method == "random":
        rng = np.random.default_rng(args.seed)
        with args.out.open("w", encoding="utf-8") as f:
            for eid in gold:
                if eid not in base:
                    continue
                f.write(
                    json.dumps(
                        {
                            "example_id": eid,
                            "risk": float(rng.random()),
                            "method": "Random",
                            "seed": args.seed,
                        }
                    )
                    + "\n"
                )
        print(f"wrote Random risks → {args.out}")
        return 0

    if args.method == "ld":
        if not args.pbce:
            raise SystemExit("--pbce required for LD")
        pbce = load_jsonl(args.pbce)
        with args.out.open("w", encoding="utf-8") as f:
            for eid, g in gold.items():
                if eid not in base or eid not in pbce:
                    continue
                yhat = base[eid].get("base_pred")
                ev = pbce[eid].get("evidence")
                if not yhat or not ev:
                    continue
                labels = tuple(g.get("label_space", "FAVOR|AGAINST|NONE").split("|"))
                f.write(
                    json.dumps(
                        {
                            "example_id": eid,
                            "risk": evidence_label_disagreement(ev, yhat, labels),
                            "method": "LD",
                        }
                    )
                    + "\n"
                )
        return 0

    if args.method == "cbd":
        if not args.other_base_preds or len(args.other_base_preds) != 2:
            raise SystemExit("CBD requires --other-base-preds PATH1 PATH2")
        o1 = load_jsonl(args.other_base_preds[0])
        o2 = load_jsonl(args.other_base_preds[1])
        with args.out.open("w", encoding="utf-8") as f:
            for eid in gold:
                if eid not in base or eid not in o1 or eid not in o2:
                    continue
                yhat = base[eid].get("base_pred")
                a = o1[eid].get("base_pred")
                b = o2[eid].get("base_pred")
                if not yhat or not a or not b:
                    continue
                f.write(
                    json.dumps(
                        {
                            "example_id": eid,
                            "risk": cross_base_disagreement(yhat, a, b),
                            "method": "CBD",
                        }
                    )
                    + "\n"
                )
        return 0

    if args.method == "llamaprob_join":
        if not args.llamaprob:
            raise SystemExit("--llamaprob required")
        lp = load_jsonl(args.llamaprob)
        with args.out.open("w", encoding="utf-8") as f:
            for eid in gold:
                if eid not in base or eid not in lp:
                    continue
                yhat = base[eid].get("base_pred")
                P = lp[eid].get("P")
                if not yhat or not P:
                    continue
                f.write(
                    json.dumps(
                        {
                            "example_id": eid,
                            "risk": llamaprob_risk(P, yhat),
                            "method": "LlamaProb",
                        }
                    )
                    + "\n"
                )
        return 0

    if args.method in ("same_model_ld", "llama_ld"):
        if not args.hard_labels:
            raise SystemExit("--hard-labels JSONL required")
        hard = load_jsonl(args.hard_labels)
        fn = same_model_label_disagreement if args.method == "same_model_ld" else llama_label_disagreement
        name = "SameModelLD" if args.method == "same_model_ld" else "LlamaLD"
        with args.out.open("w", encoding="utf-8") as f:
            for eid in gold:
                if eid not in base or eid not in hard:
                    continue
                yhat = base[eid].get("base_pred")
                h = hard[eid].get(args.hard_label_field) or hard[eid].get("base_pred")
                if not yhat or not h:
                    continue
                f.write(json.dumps({"example_id": eid, "risk": fn(h, yhat), "method": name}) + "\n")
        return 0

    raise SystemExit(f"unhandled method {args.method}")


if __name__ == "__main__":
    raise SystemExit(main())
