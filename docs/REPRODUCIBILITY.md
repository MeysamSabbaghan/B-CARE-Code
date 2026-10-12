# Reproducibility

## Offline package checks (no GPU / API / datasets)

```bash
python scripts/verify_release.py
pytest -q
```

These checks validate parsers, risk formulas, evaluation metrics (midrank AUROC, tie-aware AUPRC, ErrorCapture with `round(n·q)`), prompt hashes, script syntax, LlamaProb scorer identity constants, anonymity scanners, and synthetic end-to-end joins.

## Implementation coverage (runtime requires hardware / API / data)

- Local HF bases (Phi / Mistral / Gemma) and the local baseline suite
- Corrected \(p(\mathrm{True})\), SE, EVSE (need GPU + TRAIN CSV + NLI)
- Fixed-Llama PBCE / PACE / EDCV (need OpenRouter)
- LlamaProb closed-set scoring (need multi-GPU HF weights)

## Full pipeline prerequisites

1. **Datasets** — obtain four evaluation sets; place under `data/raw/` per `configs/datasets.yaml`. VaccineStance has no verified public mirror in this release.
2. **GPU** — bases + local suite; multi-GPU for LlamaProb.
3. **OpenRouter** — Fixed-Llama PBCE / PACE / EDCV (`OPENROUTER_API_KEY`).
4. **HF token** — gated models as required.

## Known limits

- OpenRouter model revision string is unknowable via the API.
- VaccineStance redistribution is not packaged.
- Gemma is pinned to revision `005ad340…` with Gemma3 loader classes.
- See `SOURCE_REPRO_AUDIT.md` for per-method source mapping and status.
