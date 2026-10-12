# B-CARE — Anonymous ARR Code Release

**B-CARE: Blind Classwise Assessment of Relative Evidence for Stance Error Ranking**

Anonymous companion code for ACL ARR. Pipelines cover **4 datasets × 3 base models (12 settings)**.

## Method (one line)

A prediction-blind verifier emits classwise ordinal evidence \(E_c\in\{0,1,2,3\}\). With base prediction \(\hat y\), B-CARE ranks error risk by \(R=A-S\) where \(S=E_{\hat y}\) and \(A=\max_{c\neq\hat y}E_c\).

## 12 settings

| Datasets | Bases |
|---|---|
| SemEval-2016, VaccineStance, COVID-19-Stance, P-Stance | Phi-4, Mistral-Small-24B-Instruct-2501, Gemma-3-27B |

## Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# Optional GPU / API inference:
pip install -r requirements-infer.txt
cp .env.example .env   # set OPENROUTER_API_KEY / HF_TOKEN as needed (placeholders only)
```

Python ≥3.9. Core deps: `numpy`, `scikit-learn`, `PyYAML`, `networkx`. Inference extras: `torch`, `transformers`, `accelerate`, `requests`, `python-dotenv`.

**Local model caches (no hard-coded machine paths):** set `HF_HOME` / `HUGGINGFACE_HUB_CACHE`, or `BCARE_MODEL_PATH_{PHI,MISTRAL,GEMMA}`, or pass `--model-local-path` / `--local-files-only` to GPU scripts. See `.env.example`.

## Dataset preparation

1. Obtain the four TEST sets (see `docs/DATASETS.md` and `configs/datasets.yaml`).
2. Place raw files under `data/raw/{semeval2016,vaccines,covid_19,PStance}/`.
3. Prepare and **strictly validate** manifests:

```bash
python scripts/prepare_data.py --raw-root data/raw --out-dir outputs/manifests --dataset all
python scripts/validate_manifest.py --manifest outputs/manifests/SemEval_test.csv --dataset-key semeval
```

Expected eval sizes: SemEval 1249, Vaccines 1000, COVID 800, P-Stance 2157. Invalid manifests fail hard.

**Limitation:** VaccineStance has no verified public redistributable URL in this package.

## Pipeline (reviewer workflow)

Outputs are JSONL/CSV under `outputs/` (metrics JSON, per-example risks).

```bash
# 1) Base predictions (GPU)
python scripts/run_base_models.py \
  --manifest outputs/manifests/SemEval_test.csv --base Phi \
  --out outputs/base/SemEval_Phi.jsonl

# 2) Local GPU suite (MSP/PE/DSV/LE/VR + optional p(True)/SE/EVSE)
python scripts/run_local_baseline_suite.py \
  --manifest outputs/manifests/SemEval_test.csv --base Phi \
  --enable-sampling --enable-same-model-evidence \
  --enable-ptrue --train-csv data/raw/semeval2016/train.csv \
  --enable-se-evse \
  --out outputs/baselines/SemEval_Phi_suite.jsonl

# 3) B-CARE Fixed-Llama PBCE (OpenRouter)
python scripts/run_bcare.py --mode pbce \
  --manifest outputs/manifests/SemEval_test.csv \
  --out outputs/bcare/SemEval_pbce.jsonl

# 4) PACE / EDCV (OpenRouter; require --base-preds)
python scripts/run_bcare.py --mode pace \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --out outputs/bcare/SemEval_Phi_pace.jsonl
python scripts/run_bcare.py --mode edcv \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --out outputs/bcare/SemEval_Phi_edcv.jsonl

# 5) Offline controls / ablations
python scripts/run_baselines.py --method ld \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --pbce outputs/bcare/SemEval_pbce.jsonl \
  --out outputs/controls/SemEval_Phi_ld.jsonl
python scripts/run_ablations.py \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --pbce outputs/bcare/SemEval_pbce.jsonl \
  --out outputs/ablations/SemEval_Phi.jsonl

# 6) LlamaProb (self-hosted multi-GPU) + join
python scripts/run_llamaprob.py \
  --manifest outputs/manifests/SemEval_test.csv \
  --out-dir outputs/llamaprob/SemEval
python scripts/run_baselines.py --method llamaprob_join \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --llamaprob outputs/llamaprob/SemEval/per_example_closed_set.jsonl \
  --out outputs/controls/SemEval_Phi_llamaprob.jsonl

# 7) Evaluate a suite risk field (writes metrics JSON + per-example CSV)
python scripts/evaluate.py \
  --manifest outputs/manifests/SemEval_test.csv \
  --base-preds outputs/base/SemEval_Phi.jsonl \
  --suite-jsonl outputs/baselines/SemEval_Phi_suite.jsonl \
  --risk-field p_true_risk \
  --method-name pTrue \
  --out-dir outputs/metrics/SemEval_Phi/pTrue

# 8) Aggregate locally computed metrics
python scripts/generate_paper_outputs.py \
  --metrics-root outputs/metrics \
  --out-csv outputs/tables/summary_auroc.csv
```

## Executable entry points

| Method | Entry point |
|---|---|
| MSP, PE, DSV, LE, VR, Same-Model Evidence | `scripts/run_local_baseline_suite.py` |
| Corrected \(p(\mathrm{True})\) | suite `--enable-ptrue --train-csv …` |
| SE, EVSE | suite `--enable-se-evse --enable-sampling` |
| B-CARE PBCE / PACE / EDCV | `scripts/run_bcare.py --mode {pbce,pace,edcv}` |
| LD, CBD, Random, LlamaProb join, LD variants | `scripts/run_baselines.py --method …` |
| Component ablations | `scripts/run_ablations.py` |
| LlamaProb scoring | `scripts/run_llamaprob.py` |
| Metrics + bootstrap | `scripts/evaluate.py` |

## Offline checks (no GPU/API/datasets)

```bash
python scripts/verify_release.py
pytest -q
```

## Runtime prerequisites

- Model weights, GPU hardware, and (for LlamaProb) multi-GPU hosts for generative methods.
- OpenRouter API key for Fixed-Llama PBCE / PACE / EDCV (`OPENROUTER_API_KEY`).
- Raw datasets obtained by the reviewer; VaccineStance availability is limited.
- OpenRouter Fixed-Llama revision is unknowable via the API.

## Docs

`docs/SOURCE_REPRO_AUDIT.md`, `docs/REPRODUCIBILITY.md`, `docs/DATASETS.md`, `docs/LLAMAPROB.md`, `docs/MODELS.md`.

## License

MIT (anonymous authorship for double-blind review). Datasets/models retain original licenses.
