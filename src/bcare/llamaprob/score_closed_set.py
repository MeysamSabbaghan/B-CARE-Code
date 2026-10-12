#!/usr/bin/env python3
"""Corrected teacher-forced closed-set scoring — Llama-3.3-70B-Instruct.

BF16 weights, FP32 log_softmax, flat/uniform abort. No generation.
Writes to --out-dir only (never overwrites quarantined INVALID scores).

UNIFORM_LOGPROB_ABORT requires near-uniform *logits* (low std/range), not merely
a single-token LP near -log|V|. Peaked rows can assign ~1/|V| to a non-argmax
label (SemEval ID 294 NONE) without being the BF16 flat-logit failure mode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_REV = "6f6073b423013f6a7d4d9f39144961bfbfbc386b"
SCORING_VERSION = "corrected_fp32_log_softmax_v1_1"
# Healthy Llama-3.3 rows here have std≈2.2–2.4 and range≈40+. Near-flat failure
# rows have std≪1 and tiny range; keep headroom above FLAT_LOGITS (1e-3).
UNIFORM_DEGENERATE_STD = 0.5
UNIFORM_DEGENERATE_RANGE = 2.0


def log(msg: str) -> None:
    print(msg, flush=True)


def as_id_list(ids):
    if isinstance(ids, dict) or hasattr(ids, "keys"):
        try:
            ids = ids["input_ids"]
        except Exception:
            pass
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(x) for x in ids]


def softmax_closed(logscores: dict[str, float]) -> dict[str, float]:
    m = max(logscores.values())
    exps = {k: math.exp(v - m) for k, v in logscores.items()}
    z = sum(exps.values())
    return {k: float(exps[k] / z) for k in logscores}


def validate_closed_set(probs: dict[str, float], logscores: dict[str, float]) -> dict:
    s = sum(probs.values())
    re = softmax_closed(logscores)
    max_re_err = max(abs(probs[k] - re[k]) for k in probs)
    return {
        "sum_P": s,
        "sum_P_ok": abs(s - 1.0) < 1e-6,
        "all_finite": all(math.isfinite(v) for v in probs.values())
        and all(math.isfinite(v) for v in logscores.values()),
        "softmax_recompute_max_abs_err": max_re_err,
        "softmax_recompute_ok": max_re_err < 1e-9,
        "n_labels": len(probs),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--prompt-ternary", type=Path, required=True)
    ap.add_argument("--prompt-binary", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-memory-per-gpu", type=str, default=None)
    ap.add_argument("--max-memory-cpu", type=str, default=None)
    ap.add_argument("--offload-folder", type=Path, default=None)
    ap.add_argument("--tag", type=str, default=SCORING_VERSION)
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    raw_path = out / "per_example_closed_set.jsonl"

    # Refuse writing into the quarantined original directory by name
    if out.name == "continuous_llama" and "corrected" not in str(out):
        raise SystemExit(
            "REFUSE: out-dir looks like the quarantined continuous_llama root; "
            "use a corrected_* subdirectory"
        )

    tern = args.prompt_ternary.read_text(encoding="utf-8")
    bina = args.prompt_binary.read_text(encoding="utf-8")
    tern_sha = hashlib.sha256(tern.encode()).hexdigest()
    bina_sha = hashlib.sha256(bina.encode()).hexdigest()

    rows = list(csv.DictReader(args.manifest.open()))
    if args.max_examples is not None:
        rows = rows[: int(args.max_examples)]
    assert rows, "empty manifest"

    done = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done.add((rec["dataset"], str(rec["example_id"])))

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        tok_path = Path.home() / ".cache/huggingface/token"
        if tok_path.exists():
            token = tok_path.read_text().strip()
    if not token:
        raise SystemExit("BLOCKED_BY_HF_ACCESS: no token")

    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    log(
        f"loading {MODEL_ID}@{MODEL_REV} bf16 device_map=auto "
        f"scoring={args.tag} n_gpu={n_gpu}"
    )
    tok = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REV, token=token, trust_remote_code=False
    )
    load_kwargs: dict = {
        "revision": MODEL_REV,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "token": token,
        "trust_remote_code": False,
    }
    max_memory = None
    if args.max_memory_per_gpu or args.max_memory_cpu:
        max_memory = {}
        if args.max_memory_per_gpu:
            for i in range(n_gpu):
                max_memory[i] = args.max_memory_per_gpu
        if args.max_memory_cpu:
            max_memory["cpu"] = args.max_memory_cpu
        load_kwargs["max_memory"] = max_memory
        log(f"max_memory={max_memory}")
    if args.offload_folder is not None:
        args.offload_folder.mkdir(parents=True, exist_ok=True)
        load_kwargs["offload_folder"] = str(args.offload_folder)
        load_kwargs["offload_state_dict"] = True
        log(f"offload_folder={args.offload_folder}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **load_kwargs)
    model.eval()
    device = torch.device("cuda:0") if n_gpu > 0 else next(model.parameters()).device

    def chat_ids(user_text: str) -> list[int]:
        messages = [{"role": "user", "content": user_text}]
        ids = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        return as_id_list(ids)

    def teacher_forced_logprob(prefix_ids: list[int], cont_text: str) -> dict:
        cont_ids = tok.encode(cont_text, add_special_tokens=False)
        full = prefix_ids + cont_ids
        inp = torch.tensor([full], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=inp).logits[0]
        start = len(prefix_ids) - 1
        token_lps = []
        row_stats: list[tuple[float, float]] = []
        vocab = int(logits.shape[-1])
        uniform = -math.log(vocab)
        for i, tid in enumerate(cont_ids):
            row = logits[start + i].float()
            std = float(row.std().item())
            rng = float((row.max() - row.min()).item())
            row_stats.append((std, rng))
            if std < 1e-3:
                raise RuntimeError(
                    f"FLAT_LOGITS_ABORT pos={start + i} std={std} range={rng}"
                )
            lp = float(torch.log_softmax(row, dim=-1)[int(tid)].item())
            token_lps.append(lp)
        # Abort only when LPs look uniform *and* the scored rows lack peaked
        # structure (true flat/near-flat failure). Do not abort a healthy peaked
        # distribution that happens to assign ≈1/|V| to one continuation token.
        if cont_ids and all(abs(x - uniform) < 0.05 for x in token_lps):
            degenerate = any(
                std < UNIFORM_DEGENERATE_STD or rng < UNIFORM_DEGENERATE_RANGE
                for std, rng in row_stats
            )
            if degenerate:
                raise RuntimeError(
                    f"UNIFORM_LOGPROB_ABORT continuation={cont_text!r} "
                    f"lps={token_lps} row_stats={row_stats}"
                )
        return {
            "continuation": cont_text,
            "token_ids": cont_ids,
            "token_logprobs": token_lps,
            "seq_logprob": float(sum(token_lps)),
        }

    t0 = time.time()
    n_new = 0
    validation_fail = 0
    for r in rows:
        key = (r["dataset"], str(r["example_id"]))
        if key in done:
            continue
        labels = tuple(r["label_space"].split("|"))
        tmpl = tern if r["prompt_kind"] == "ternary" else bina
        user = tmpl.replace("{target}", r["target"]).replace("{text}", r["text"])
        prefix = chat_ids(user)
        logscores = {}
        detail = {}
        for lab in labels:
            sc = teacher_forced_logprob(prefix, lab)
            logscores[lab] = sc["seq_logprob"]
            detail[lab] = sc
        probs = softmax_closed(logscores)
        v = validate_closed_set(probs, logscores)
        if not (v["sum_P_ok"] and v["all_finite"] and v["softmax_recompute_ok"]):
            validation_fail += 1
            raise RuntimeError(f"CLOSED_SET_VALIDATION_FAIL {key} {v}")
        # Reject length-prior signature (uniform −11.75 * ntok)
        if all(
            abs(logscores[lab] - (-11.75 * len(detail[lab]["token_ids"]))) < 0.25
            for lab in labels
        ):
            raise RuntimeError(f"LENGTH_PRIOR_SIGNATURE_ABORT {key} {logscores}")

        rec = {
            "dataset": r["dataset"],
            "uid": r["uid"],
            "example_id": r["example_id"],
            "target": r["target"],
            "text_sha256": r["text_sha256"],
            "gold": r["gold"],
            "label_space": list(labels),
            "prompt_kind": r["prompt_kind"],
            "prompt_sha256": tern_sha if r["prompt_kind"] == "ternary" else bina_sha,
            "model_id": MODEL_ID,
            "revision": MODEL_REV,
            "dtype": "bfloat16",
            "log_softmax_dtype": "float32",
            "quantization": None,
            "scoring": "teacher_forced_sequence_logprob_closed_set_softmax",
            "scoring_version": args.tag,
            "logscores": logscores,
            "P": probs,
            "closed_set_validation": v,
            "token_detail": detail,
            "gold_used_in_inference": False,
            "shared_across_bases": ["Phi", "Mistral", "Gemma"],
        }
        with raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
        if n_new % 10 == 0 or n_new == 1:
            log(f"scored_new={n_new} last={key} P={probs} wall_s={time.time() - t0:.1f}")

    n_total = sum(1 for line in raw_path.read_text().splitlines() if line.strip())
    wall = time.time() - t0
    summary = {
        "model_id": MODEL_ID,
        "revision": MODEL_REV,
        "scoring_version": args.tag,
        "log_softmax_dtype": "float32",
        "n_manifest": len(rows),
        "n_scored": n_total,
        "n_new_this_run": n_new,
        "complete": n_total == len(rows),
        "wall_s": wall,
        "sec_per_new_example": (wall / n_new) if n_new else None,
        "prompt_ternary_sha256": tern_sha,
        "prompt_binary_sha256": bina_sha,
        "gpu_names": [
            torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
        ]
        if torch.cuda.is_available()
        else [],
        "peak_memory_gib": (
            round(torch.cuda.max_memory_allocated() / (1024**3), 3)
            if torch.cuda.is_available()
            else None
        ),
        "max_memory": max_memory,
        "quantization": None,
        "method": "teacher_forced_closed_set_bf16_weights_fp32_log_softmax",
        "validation_fail": validation_fail,
        "quarantine_note": "Does not overwrite continuous_llama/per_example_closed_set.jsonl (INVALID_PENDING_AUDIT)",
    }
    (out / "run_complete.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(f"COMPLETE {summary}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
