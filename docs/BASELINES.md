# Baselines & controls

All baselines have runnable CLI entry points (not library-only).

| Method | Entry point | Needs |
|---|---|---|
| MSP / PE / DSV | `scripts/run_local_baseline_suite.py` | GPU |
| LE / VR | suite `--enable-sampling` | GPU |
| Corrected \(p(\mathrm{True})\) | suite `--enable-ptrue --train-csv TRAIN.csv` | GPU + TRAIN split |
| SE / EVSE | suite `--enable-se-evse --enable-sampling` | GPU + DeBERTa-MNLI |
| Same-Model Evidence | suite `--enable-same-model-evidence` | GPU |
| LD / Same-Model LD / Llama LD / CBD / Random | `scripts/run_baselines.py --method …` | offline after preds |
| LlamaProb join | `scripts/run_baselines.py --method llamaprob_join` | after LlamaProb scoring |
| EDCV / PACE | `scripts/run_bcare.py --mode edcv\|pace` | OpenRouter |
| B-CARE PBCE | `scripts/run_bcare.py --mode pbce` | OpenRouter |
| Component ablations (\(A{-}S\), \(-S\), \(A\), share, softmax, lexico, LD) | `scripts/run_ablations.py` | offline after PBCE |

## \(p(\mathrm{True})\) settings

- Few-shot pool: **TRAIN only** (`--train-csv`)
- Default: 20 few-shot, 10 brainstorm samples, seed `2026092817`
- Prompt: jlko `Question` / `Brainstormed Answers` / `A) True` / `B) False` / `The possible answer is:`
- Scoring: raw completion tokenize(`prompt + ' A'`); risk = jlko `p_false_fixed = 1 - P(' A')`
- No chat-template wrapping for the True/False score step

## SE / EVSE settings

- Sampling: \(M{=}5\), \(T{=}1.0\), `top_p=0.9`, seed base `42`
- NLI: `microsoft/deberta-v2-xlarge-mnli` bidirectional entailment clustering
- SE: jlko `logsumexp_by_id` + `predictive_entropy_rao`
- EVSE: `baselines/evse_wrapper.compute_evse` on the same samples
