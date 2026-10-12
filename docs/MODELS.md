# Models & decoding

See `configs/models.yaml`.

**Bases:** Phi-4 (`microsoft/phi-4` @ `2db69c1c…`), Mistral-Small-24B-Instruct-2501 @ `9527884b…` (`attn_implementation: eager`), Gemma-3-27B-IT @ `005ad340…` via `Gemma3ForConditionalGeneration` + `Gemma3Processor`.

**Base decoding:** greedy `do_sample=false`, `max_new_tokens=16`, seed `2026092701`. Config `temperature=0.2` is unused under greedy decoding.

**Fixed-Llama (OpenRouter PBCE/PACE):** `meta-llama/llama-3.3-70b-instruct`, `temperature=0.0`, `max_tokens=80` (EDCV: 16).

**LlamaProb (self-hosted):** `meta-llama/Llama-3.3-70B-Instruct` @ `6f6073b4…`, BF16 weights, FP32 log_softmax, teacher-forced closed-set scores; risk `1-P(ŷ)`.

**Sampling baselines:** M=5, T=1.0, top_p=0.9, max_new_tokens=8, seed 42.
