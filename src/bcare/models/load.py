"""Canonical HF model loading matching the original experimental suite provenance."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


_BASE_ENV_KEYS = {
    "Phi": "BCARE_MODEL_PATH_PHI",
    "Mistral": "BCARE_MODEL_PATH_MISTRAL",
    "Gemma": "BCARE_MODEL_PATH_GEMMA",
}


def resolve_model_source(
    mcfg: dict[str, Any],
    *,
    base_name: str | None = None,
    local_path: str | Path | None = None,
) -> str:
    """Resolve HF id or local directory without hardcoding machine paths.

    Precedence: explicit ``local_path`` → env ``BCARE_MODEL_PATH_*`` →
    ``mcfg['local_path']`` → ``mcfg['hf_model_id']``.
    """
    if local_path:
        p = Path(local_path).expanduser()
        if not p.exists():
            raise SystemExit(f"REFUSE: --model-local-path does not exist: {p}")
        return str(p)
    if base_name:
        env_key = _BASE_ENV_KEYS.get(base_name)
        if env_key and os.environ.get(env_key):
            p = Path(os.environ[env_key]).expanduser()
            if not p.exists():
                raise SystemExit(f"REFUSE: {env_key} path does not exist: {p}")
            return str(p)
    if mcfg.get("local_path"):
        p = Path(str(mcfg["local_path"])).expanduser()
        if not p.exists():
            raise SystemExit(f"REFUSE: models.yaml local_path does not exist: {p}")
        return str(p)
    return str(mcfg["hf_model_id"])


def _local_files_only(mcfg: dict[str, Any]) -> bool:
    env = os.environ.get("BCARE_LOCAL_FILES_ONLY", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    if env in {"0", "false", "no"}:
        return False
    return bool(mcfg.get("local_files_only", False))


def load_base_model_and_tokenizer(
    mcfg: dict[str, Any],
    *,
    device_map: str = "auto",
    base_name: str | None = None,
    local_path: str | Path | None = None,
):
    """Load Phi/Mistral/Gemma with the same classes/pins as the experimental suites.

    - Phi / Mistral: AutoModelForCausalLM + AutoTokenizer
    - Mistral: attn_implementation='eager' (SDPA nan mitigation from original suite)
    - Gemma: Gemma3ForConditionalGeneration + Gemma3Processor (NOT AutoCausalLM)

    Local caches: set ``HF_HOME`` / ``HUGGINGFACE_HUB_CACHE``, or pass
    ``--model-local-path`` / ``BCARE_MODEL_PATH_{PHI,MISTRAL,GEMMA}``.
    Use ``BCARE_LOCAL_FILES_ONLY=1`` to refuse Hub downloads.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = resolve_model_source(mcfg, base_name=base_name, local_path=local_path)
    revision = mcfg.get("revision")
    # Local directories already pin a snapshot; revision is only for Hub ids.
    use_revision = revision if not Path(model_id).exists() else None
    dtype = getattr(torch, mcfg.get("dtype", "bfloat16"))
    model_class = mcfg.get("model_class") or "AutoModelForCausalLM"
    processor_class = mcfg.get("processor_class")
    local_only = _local_files_only(mcfg)

    if model_class == "Gemma3ForConditionalGeneration" or (
        processor_class == "Gemma3Processor"
    ):
        from transformers import Gemma3ForConditionalGeneration, Gemma3Processor

        if use_revision is None and not Path(model_id).exists():
            raise SystemExit(
                "REFUSE: Gemma revision must be pinned "
                "(expected 005ad3404e59d6023443cb575daa05336842228a)"
            )
        proc_kw: dict[str, Any] = {
            "trust_remote_code": False,
            "local_files_only": local_only,
        }
        load_kw: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": device_map,
            "trust_remote_code": False,
            "local_files_only": local_only,
        }
        if use_revision:
            proc_kw["revision"] = use_revision
            load_kw["revision"] = use_revision
        processor = Gemma3Processor.from_pretrained(model_id, **proc_kw)
        model = Gemma3ForConditionalGeneration.from_pretrained(model_id, **load_kw)
        model.eval()
        # Processor owns tokenizer + chat template
        tok = processor.tokenizer
        tok._bcare_processor = processor  # type: ignore[attr-defined]
        return model, tok

    tok_kw: dict[str, Any] = {
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    load_kw = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": False,
        "local_files_only": local_only,
    }
    if use_revision:
        tok_kw["revision"] = use_revision
        load_kw["revision"] = use_revision
    if mcfg.get("attn_implementation"):
        load_kw["attn_implementation"] = mcfg["attn_implementation"]

    tok = AutoTokenizer.from_pretrained(model_id, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    model.eval()
    return model, tok
