#!/usr/bin/env python3
"""Run prediction-blind B-CARE Fixed-Llama evidence (OpenRouter).

Evidence is generated once per unique text and reusable across bases.
Risk R = A − S is joined later in evaluate.py using each base's ŷ.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from bcare.parse_evidence import parse_evidence_scores  # noqa: E402
from bcare.verifier.openrouter import chat_completion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--mode", choices=["pbce", "pace", "edcv"], default="pbce")
    ap.add_argument("--base-preds", type=Path, default=None, help="Required for pace/edcv")
    ap.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-retries-parse", type=int, default=2)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.models_config.read_text())["verifier_openrouter"]
    model = cfg["model"]
    temp = float(cfg["temperature"])

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    pred_by_id = {}
    if args.mode in ("pace", "edcv"):
        if args.base_preds is None or not args.base_preds.exists():
            raise SystemExit("--base-preds JSONL required for pace/edcv")
        for line in args.base_preds.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                pred_by_id[str(rec["example_id"])] = rec.get("base_pred")

    if args.mode == "pbce":
        # Choose ternary/binary prompt from first row
        kind = rows[0].get("prompt_kind", "ternary") if rows else "ternary"
        prompt_path = ROOT / ("prompts/bcare_pbce_binary.txt" if kind == "binary" else "prompts/bcare_pbce_ternary.txt")
        max_tokens = int(cfg["max_tokens_evidence"])
        system = None
    elif args.mode == "pace":
        kind = rows[0].get("prompt_kind", "ternary") if rows else "ternary"
        prompt_path = ROOT / ("prompts/pace_binary.txt" if kind == "binary" else "prompts/pace_ternary.txt")
        max_tokens = int(cfg["max_tokens_evidence"])
        system = None
    else:
        kind = rows[0].get("prompt_kind", "ternary") if rows else "ternary"
        prompt_path = ROOT / ("prompts/edcv_binary.txt" if kind == "binary" else "prompts/edcv_ternary.txt")
        max_tokens = int(cfg["max_tokens_edcv"])
        system = "Respond with exactly one integer character: 0, 1, 2, or 3. Do not write any other text."

    tmpl = prompt_path.read_text(encoding="utf-8")
    prompt_sha = hashlib.sha256(tmpl.encode()).hexdigest()
    labels = ("FAVOR", "AGAINST") if kind == "binary" else ("FAVOR", "AGAINST", "NONE")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(str(json.loads(line)["example_id"]))

    n_new = 0
    with args.out.open("a", encoding="utf-8") as f:
        for r in rows:
            eid = str(r["example_id"])
            if eid in done:
                continue
            user = tmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
            if args.mode in ("pace", "edcv"):
                yhat = pred_by_id.get(eid)
                if not yhat:
                    print(f"skip {eid}: missing base_pred", flush=True)
                    continue
                user = user.replace("{base_prediction}", str(yhat))
                # EDCV ternary template uses {base_prediction} via "Predicted stance" field name
                if "{base_prediction}" not in tmpl and "Predicted stance" in tmpl:
                    pass
                if args.mode == "edcv" and "{base_prediction}" not in tmpl:
                    # edcv_ternary uses {base_prediction} in our packaged file as predicted stance placeholder
                    user = tmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
                    if "{base_prediction}" in tmpl:
                        user = user.replace("{base_prediction}", str(yhat))
                    else:
                        # fallback for older wording
                        user = user.replace("{base_prediction}", str(yhat))

            raw = None
            parsed = None
            for attempt in range(args.max_retries_parse + 1):
                resp = chat_completion(
                    model=model,
                    user=user,
                    temperature=temp,
                    max_tokens=max_tokens,
                    system=system,
                )
                raw = resp["content"]
                if args.mode == "edcv":
                    from bcare.parse_evidence import parse_edcv_correctness

                    c = parse_edcv_correctness(raw)
                    if c is not None:
                        parsed = {"C": c, "EDCV_Risk": 3 - c}
                        break
                else:
                    ev = parse_evidence_scores(raw, labels)
                    if ev is not None:
                        parsed = ev
                        break
            rec = {
                "example_id": r["example_id"],
                "dataset": r.get("dataset"),
                "target": r["target"],
                "mode": args.mode,
                "prompt_sha256": prompt_sha,
                "model": model,
                "temperature": temp,
                "max_tokens": max_tokens,
                "raw_output": raw,
                "parse_ok": parsed is not None,
                "evidence": parsed if args.mode != "edcv" else None,
                "edcv": parsed if args.mode == "edcv" else None,
                "prediction_blind": args.mode == "pbce",
            }
            if args.mode in ("pace", "edcv"):
                rec["base_prediction"] = pred_by_id.get(eid)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 10 == 0:
                print(f"{args.mode}: wrote {n_new}", flush=True)
    print(f"done mode={args.mode} new={n_new} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
