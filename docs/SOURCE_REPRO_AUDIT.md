# Source-to-release audit (public summary)

**Scope:** Compare this anonymous release against the original experimental implementation (local baseline suite modules, shared formula library, frozen prompts, and the self-hosted LlamaProb scorer).  
**Not:** GPU/API reruns.  
**Research tree:** left unchanged outside this package.

## How to read statuses

| Label | Meaning |
|---|---|
| **VERIFIED** | Source↔release algorithm/settings match with concrete evidence (hashes and/or formula identity tests). Offline path exercised. |
| **PARTIALLY VERIFIED** | Source comparison completed and matches (or was corrected to match); **GPU/API runtime not executed** in this packaging pass. |
| **NOT VERIFIED** | Original source unavailable or comparison incomplete. |
| **MISMATCH** | Substantive discrepancy remains. |

**Important:** Passing imports, source hashes, or synthetic tests does **not** mean GPU/API runs were executed. Generative methods remain **PARTIALLY VERIFIED** until run with the required models/API/data.

---

## Verified offline (formulas / metrics / joins)

| Method | Original source (neutral) | Release | Settings | Implementation | Tests | Class | Remaining issues |
|---|---|---|---|---|---|---|---|
| LD (Evidence–label) | shared formula library (unique-argmax) | `evidence_label_disagreement` + `run_baselines.py --method ld` | — | Formula match | Synthetic e2e | **VERIFIED** | — |
| Same-Model LD / Llama LD | hard≠ŷ | `run_baselines.py` | — | Formula match | Unit | **VERIFIED** | — |
| CBD | 0/1/2/3 via n_match | `cross_base_disagreement` | — | Formula match | Synthetic e2e | **VERIFIED** | — |
| Ablations A−S/−S/A/share/softmax/lexico | component-ablation experimental script | `run_ablations.py` + `risk.py` | — | Match; share ΣE=0 → `1−1/\|C\|` | Unit + smoke | **VERIFIED** | — |
| AUROC | midrank Mann–Whitney in shared formula library | `metrics.midrank_auroc` / `auroc` | — | Midrank port | Unit | **VERIFIED** | — |
| AUPRC | tie-aware PR integration in shared formula library | `metrics.auprc` | — | Ported | Unit | **VERIFIED** | — |
| ErrorCapture | expected-tie with `k=round(n·q)` | `error_capture_expected_tie` | — | Ported | Unit | **VERIFIED** | — |

## Partially verified (source match; runtime not executed)

| Method | Original source (neutral) | Release | Settings | Implementation | Tests | Class | Remaining issues |
|---|---|---|---|---|---|---|---|
| MSP | experimental suite `msp_pe.py` + teacher-forced **sum** logprob | `src/baselines/msp_pe.py` + `run_local_baseline_suite.py` | Match (no length-norm) | Ternary order preserved; binary uses `label_space` keys | Unit + verify | **PARTIALLY VERIFIED** | No GPU regen |
| PE | same as MSP | same | Match | Shared `compute_msp_pe` | Unit | **PARTIALLY VERIFIED** | No GPU |
| Corrected p(True) | `p_true_official.py` + YAML 20/10/`1−P(A)` + TRAIN few-shot | identical module; suite `--enable-ptrue` | Match | Module byte-identical; jlko raw score wired | Unit | **PARTIALLY VERIFIED** | No GPU + TRAIN CSV |
| DSV | `dsv.py` + True/False prompt | identical | Match | Byte-identical | Unit | **PARTIALLY VERIFIED** | No GPU |
| LabelEntropy / VR | `label_entropy_vr.py` | identical | M=5,T=1,top_p=0.9 | Byte-identical | Unit | **PARTIALLY VERIFIED** | No GPU sampling |
| Semantic Entropy | official SE module + DeBERTa-MNLI | identical + suite NLI pass | Match | Byte-identical | Unit synthetic | **PARTIALLY VERIFIED** | No GPU+NLI |
| EVSE | EVSE wrapper + core | identical; `networkx` required | Match | Core byte-identical | Import/`--help` | **PARTIALLY VERIFIED** | No GPU+NLI |
| Same-Model Evidence | suite V2 + PBCE prompt + A−S | suite flag + `evidencerisk_as` | Match | Formula + prompt SHA match | Unit | **PARTIALLY VERIFIED** | No GPU |
| B-CARE / PBCE | Fixed-Llama OpenRouter PBCE; R=A−S | `run_bcare.py --mode pbce` | T=0,max_tokens=80 | Prompt SHA + formula match | Unit; CLI help | **PARTIALLY VERIFIED** | No OpenRouter |
| PACE | prediction-aware prompt; A−S | `run_bcare.py --mode pace` | Match | Prompt SHA + formula | CLI help | **PARTIALLY VERIFIED** | No API |
| EDCV | correctness prompt; Risk=3−C | `run_bcare.py --mode edcv` | max_tokens=16 | Match | Unit | **PARTIALLY VERIFIED** | No API |
| LlamaProb | self-hosted closed-set scorer @ pinned revision | identical scorer + `run_llamaprob.py` + join `1−P(ŷ)` | Match | Byte-identical scorer SHA `e87ef227…` | Hash test | **PARTIALLY VERIFIED** | No multi-GPU |
| Random | seeded RNG | `run_baselines.py --method random` | Default seed `2026092800` | Formula OK | Unit | **PARTIALLY VERIFIED** | — |
| AURC / E-AURC | experimental suite lexsort AURC | identical lexsort excess | — | Match suite | Unit | **PARTIALLY VERIFIED** | Some diagnostics use expected-tie AURC |
| Bootstrap | paired B=10000 | `paired_bootstrap_delta_auroc` | B=10000 default | Match policy | Unit B=200 | **PARTIALLY VERIFIED** | Seeds via CLI |
| Bases Phi/Mistral/Gemma | experimental YAML pins + Gemma3 classes | `models.yaml` + `bcare.models.load` | Match; Mistral `eager` | Match | Config verify | **PARTIALLY VERIFIED** | No GPU load |
| Datasets (4) / 12 settings | frozen eval n + targets | `datasets.yaml` + `validate_manifest` | Match | Strict validation | Unit + config | **PARTIALLY VERIFIED** | Raw CSVs not shipped |

**MISMATCH:** none remaining after packaging fixes.  
**NOT VERIFIED:** none for methods with available original sources.

---

## Not executed in this packaging pass

- Full GPU evaluation of bases / local suite / LlamaProb  
- OpenRouter PBCE / PACE / EDCV calls  
- Real-dataset end-to-end reproduction  

## Offline checks (executed on packaging machine)

| Check | Result |
|---|---|
| `verify_release.py` | **PASS** |
| `pytest` | **PASS** |
| CLI `--help` smoke | **PASS** |
| Prompt SHA (base/PBCE) | **PASS** |
| LlamaProb scorer SHA | **PASS** |
| GPU / OpenRouter / real datasets | **NOT RUN** |

## Remaining external limitations

1. GPU + Hugging Face weights required for bases, suite, SE/EVSE NLI, LlamaProb.  
2. OpenRouter API key required for Fixed-Llama PBCE/PACE/EDCV; API revision string unknowable.  
3. Local raw datasets required; VaccineStance has no verified public redistributable URL in this package.
