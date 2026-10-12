# Ablations

`scripts/run_ablations.py` computes from frozen PBCE evidence + base ŷ:

- Comparative \(A-S\)
- Support-only \(-S\)
- Alternative-only \(A\)
- Evidence-share / softmax-support / lexicographic
- Evidence–label disagreement (LD)

Evaluate each risk column with `scripts/evaluate.py --risk-jsonl ...` after extracting a `risk` field, or extend the ablation writer.
