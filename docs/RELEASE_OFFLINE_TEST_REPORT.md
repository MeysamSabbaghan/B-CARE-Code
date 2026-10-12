# Offline test report (packaging machine)

Date: 2026-10-12. Python 3.9.6. No GPU / OpenRouter / raw datasets used.

| Component | Result | Evidence |
|---|---|---|
| `scripts/verify_release.py` | **PASS** | Syntax, prompt SHA, branding, LlamaProb hash, suite wiring markers |
| `pytest` (14 tests) | **PASS** | Core formulas + midrank/EC ports + synthetic e2e smoke + all CLI `--help` |
| Placeholder scan (suite notes) | **PASS** | Real `p_true_risk` / SE / EVSE computation present |
| Dataset prepare on real CSVs | **NOT RUN** | Raw datasets not present in anonymous package |
| GPU base / local suite / LlamaProb | **NOT RUN** | Requires GPU + HF weights |
| OpenRouter PBCE/PACE/EDCV | **NOT RUN** | Requires API key |

## Re-run

```bash
python scripts/verify_release.py
pytest -q
```
