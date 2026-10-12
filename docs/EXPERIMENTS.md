# Experiments (12 settings)

For each of {SemEval, Vaccines, COVID, P-Stance} × {Phi, Mistral, Gemma}:

1. Prepare/validate TEST manifest (`scripts/prepare_data.py`, `validate_manifest.py`).
2. Base greedy predictions (`run_base_models.py`).
3. Local suite risks: MSP, PE, DSV, LE, VR; optional \(p(\mathrm{True})\), SE, EVSE, Same-Model evidence.
4. Fixed-Llama PBCE once per dataset (reusable across bases); PACE/EDCV per base.
5. Offline controls + ablations.
6. Optional LlamaProb scoring + join.
7. `evaluate.py` → metrics JSON; `generate_paper_outputs.py` aggregates.

Outputs belong under `outputs/` (gitignored).
