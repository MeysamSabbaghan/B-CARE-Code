#!/usr/bin/env python3
"""Run greedy base-model stance prediction for one (dataset, base) setting.

Requires: torch, transformers, HF access, prepared manifest CSV.
Decoding: do_sample=false, max_new_tokens=16, seed=2026092701 (+ example index).
Config temperature 0.2 is NOT applied under greedy decoding.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from bcare.models.load import load_base_model_and_tokenizer  # noqa: E402
from bcare.parse_evidence import parse_closed_label  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--base", choices=["Phi", "Mistral", "Gemma"], required=True)
    ap.add_argument("--models-config", type=Path, default=ROOT / "configs/models.yaml")
    ap.add_argument("--prompt-ternary", type=Path, default=ROOT / "prompts/base_ternary.txt")
    ap.add_argument("--prompt-binary", type=Path, default=ROOT / "prompts/base_binary.txt")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--device-map", type=str, default="auto")
    ap.add_argument(
        "--model-local-path",
        type=Path,
        default=None,
        help="Optional local snapshot directory (overrides HF id / BCARE_MODEL_PATH_*).",
    )
    ap.add_argument(
        "--local-files-only",
        action="store_true",
        help="Refuse Hub downloads (same as BCARE_LOCAL_FILES_ONLY=1).",
    )
    args = ap.parse_args()

    try:
        import torch
    except ImportError as e:
        raise SystemExit(
            "Missing torch/transformers. Install optional infer deps:\n"
            "  pip install 'bcare[infer]'   # or see requirements-infer.txt"
        ) from e

    cfg = yaml.safe_load(args.models_config.read_text())
    mcfg = dict(cfg["bases"][args.base])
    if args.local_files_only:
        mcfg["local_files_only"] = True
    dec = cfg["decoding"]["base"]
    tern = args.prompt_ternary.read_text(encoding="utf-8")
    bina = args.prompt_binary.read_text(encoding="utf-8")

    rows = list(csv.DictReader(args.manifest.open(encoding="utf-8")))
    if args.max_examples is not None:
        rows = rows[: args.max_examples]
    if not rows:
        raise SystemExit("empty manifest")

    model_id = mcfg["hf_model_id"]
    revision = mcfg.get("revision")
    print(f"loading {model_id} revision={revision} class={mcfg.get('model_class')}", flush=True)
    try:
        model, tok = load_base_model_and_tokenizer(
            mcfg,
            device_map=args.device_map,
            base_name=args.base,
            local_path=args.model_local_path,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing torch/transformers. Install optional infer deps:\n"
            "  pip install 'bcare[infer]'   # or see requirements-infer.txt"
        ) from e
    device = next(model.parameters()).device

    def chat_ids(user: str) -> list[int]:
        ids = tok.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=True,
            add_generation_prompt=True,
        )
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return [int(x) for x in ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_path = args.out
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(str(json.loads(line)["example_id"]))

    seed_base = int(dec["seed"])
    max_new = int(dec["max_new_tokens"])
    n_new = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            eid = str(r["example_id"])
            if eid in done:
                continue
            labels = tuple(r.get("label_space", "FAVOR|AGAINST|NONE").split("|"))
            tmpl = bina if r.get("prompt_kind") == "binary" else tern
            user = tmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
            seed = seed_base + i
            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            prefix = chat_ids(user)
            inp = torch.tensor([prefix], device=device)
            # Greedy: do_sample=False → temperature NOT passed
            with torch.no_grad():
                out = model.generate(
                    inp,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            new_ids = out[0, len(prefix) :].tolist()
            raw = tok.decode(new_ids, skip_special_tokens=True)
            pred, status = parse_closed_label(raw, labels)
            gold = r["gold"]
            error = int(pred != gold) if pred is not None else 1
            rec = {
                "example_id": r["example_id"],
                "dataset": r.get("dataset"),
                "target": r["target"],
                "gold": gold,
                "base": args.base,
                "base_pred": pred,
                "parse_status": status,
                "raw": raw,
                "error": error,
                "model_id": model_id,
                "revision": revision,
                "do_sample": False,
                "max_new_tokens": max_new,
                "seed": seed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_new += 1
            if n_new % 20 == 0:
                print(f"wrote {n_new} new predictions", flush=True)
    print(f"done base={args.base} new={n_new} out={out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
