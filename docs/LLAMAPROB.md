# LlamaProb (self-hosted closed-set probability)

## Implementation

- Executable entry: `scripts/run_llamaprob.py`
- Scorer: `src/bcare/llamaprob/score_closed_set.py`
- Join to base predictions: `scripts/run_baselines.py --method llamaprob_join`
- Risk: \(1 - P(\hat y)\) after closed-set softmax over stance labels

## Scorer identity

| Check | Value |
|---|---|
| SHA-256 of scorer | `e87ef227d1d73d92c881d921c972182748ad8ac0c01ba8e51540120c51ec41a1` |
| Checkpoint | `meta-llama/Llama-3.3-70B-Instruct` @ `6f6073b423013f6a7d4d9f39144961bfbfbc386b` |
| Scoring | teacher-forced sequence logprobs; `corrected_fp32_log_softmax` |
| Prompting | frozen base ternary/binary stance prompts |
| Join | per-example `risk = 1 - P[base_pred]` |

## Hardware

Typically 2×A100-80GB (or equivalent) with `device_map=auto`. See
`configs/models.yaml` → `llamaprob_self_hosted`.
