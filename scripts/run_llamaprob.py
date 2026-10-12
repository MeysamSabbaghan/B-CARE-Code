#!/usr/bin/env python3
"""Run self-hosted LlamaProb closed-set scoring (NOT OpenRouter).

Wraps the validated teacher-forced FP32 log_softmax scorer:
  src/bcare/llamaprob/score_closed_set.py

Risk at evaluation time: LlamaProb = 1 − P(base_pred), with the same P vector
reused across Phi/Mistral/Gemma (join in evaluate.py).

Requires: multi-GPU Hugging Face access to Llama-3.3-70B-Instruct @ pinned revision.
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCORER = ROOT / "src" / "bcare" / "llamaprob" / "score_closed_set.py"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Self-hosted LlamaProb scorer (teacher-forced closed-set)"
    )
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-memory-per-gpu", type=str, default=None)
    ap.add_argument("--max-memory-cpu", type=str, default=None)
    ap.add_argument("--offload-folder", type=Path, default=None)
    ap.add_argument(
        "--prompt-ternary",
        type=Path,
        default=ROOT / "prompts/llamaprob_ternary.txt",
    )
    ap.add_argument(
        "--prompt-binary",
        type=Path,
        default=ROOT / "prompts/llamaprob_binary.txt",
    )
    args = ap.parse_args()

    if not SCORER.exists():
        raise SystemExit(f"Missing LlamaProb scorer: {SCORER}")

    argv = [
        str(SCORER),
        "--manifest",
        str(args.manifest),
        "--prompt-ternary",
        str(args.prompt_ternary),
        "--prompt-binary",
        str(args.prompt_binary),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.max_examples is not None:
        argv += ["--max-examples", str(args.max_examples)]
    if args.max_memory_per_gpu:
        argv += ["--max-memory-per-gpu", args.max_memory_per_gpu]
    if args.max_memory_cpu:
        argv += ["--max-memory-cpu", args.max_memory_cpu]
    if args.offload_folder is not None:
        argv += ["--offload-folder", str(args.offload_folder)]

    sys.argv = argv
    runpy.run_path(str(SCORER), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
