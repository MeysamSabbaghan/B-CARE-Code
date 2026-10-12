#!/usr/bin/env python3
"""Sanity checks for the anonymous B-CARE release (no GPU, no results required)."""
from __future__ import annotations

import ast
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REQUIRED_SCRIPTS = [
    "prepare_data.py",
    "run_base_models.py",
    "run_bcare.py",
    "run_baselines.py",
    "run_local_baseline_suite.py",
    "run_ablations.py",
    "run_llamaprob.py",
    "evaluate.py",
    "generate_paper_outputs.py",
    "validate_manifest.py",
]
LLAMAPROB_SHA256 = "e87ef227d1d73d92c881d921c972182748ad8ac0c01ba8e51540120c51ec41a1"

PROMPT_SHA = {
    "base_ternary.txt": "9f176afe4516771789a3706861bcfd2d71a77b8766b53f2e8aa0e93d3c151825",
    "bcare_pbce_ternary.txt": "3598de6b1347260d5b05dc14c001b52055393f1c6dfc5042b044d2eefeb82a00",
}

# Precomputed SHA-256 of a forbidden legacy project brand token (plaintext not stored).
_LEGACY_BRAND_SHA = "33bf0f7aaf36353d35ee307b628b9abe01112ce17aa8619e58f056db6a88fe2c"


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _forbidden_internal_markers() -> list[str]:
    # Decoded at runtime so the release tree does not contain the plaintext markers.
    return [
        bytes.fromhex("62656f636174").decode("ascii"),  # cluster tree prefix
        bytes.fromhex("72756e706f64").decode("ascii"),  # remote GPU vendor shorthand
    ]


def _scan_anonymity(errors: list[str]) -> None:
    """Fail on absolute personal paths, internal infra names, emails, live secrets."""
    home = "Users"
    path_re = re.compile(rf"(/{home}/[^\s\"']+|/home/[^\s\"']+|/scratch/[^\s\"']+)")
    email_re = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(edu|com|org|net)\b")
    secret_re = re.compile(
        r"(sk-or-[A-Za-z0-9]{10,}|hf_[A-Za-z0-9]{20,}|OPENROUTER_API_KEY\s*=\s*(?!REPLACE_ME)[^\s#]+)"
    )
    markers = _forbidden_internal_markers()
    skip_suffix = {".pyc", ".png", ".jpg", ".pdf"}
    self_path = Path(__file__).resolve()

    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() in skip_suffix:
            continue
        if any(x in p.parts for x in (".venv", ".pytest_cache")):
            continue
        if p.name.endswith(".egg-info") or "egg-info" in p.parts:
            continue
        if p.resolve() == self_path:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if path_re.search(text):
            errors.append(f"absolute path in {p.relative_to(ROOT)}")
        for m in email_re.finditer(text):
            addr = m.group(0).lower()
            if addr.endswith("@openrouter.ai") or "example.com" in addr:
                continue
            errors.append(f"email-like string in {p.relative_to(ROOT)}")
            break
        if secret_re.search(text):
            errors.append(f"secret-like string in {p.relative_to(ROOT)}")
        low = text.lower()
        for marker in markers:
            if marker.lower() in low:
                errors.append(f"internal infrastructure reference in {p.relative_to(ROOT)}")
                break
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{5,}", text):
            if _sha256_text(token) == _LEGACY_BRAND_SHA:
                errors.append(f"legacy project branding token in {p.relative_to(ROOT)}")
                break


def main() -> int:
    errors = []
    for pat in ("results/**/*.csv", "**/TABLE_*.csv", "**/AGGREGATE_*.csv"):
        hits = list(ROOT.glob(pat))
        if hits:
            errors.append(f"forbidden result artifacts: {hits}")
    if (ROOT / "results").exists() and any((ROOT / "results").rglob("*")):
        files = [p for p in (ROOT / "results").rglob("*") if p.is_file()]
        if files:
            errors.append(f"results/ must be empty of experimental files: {files[:5]}")

    for name in REQUIRED_SCRIPTS:
        p = ROOT / "scripts" / name
        if not p.exists():
            errors.append(f"missing script {name}")
        else:
            try:
                ast.parse(p.read_text(encoding="utf-8"))
            except SyntaxError as e:
                errors.append(f"syntax error {name}: {e}")

    for p in (ROOT / "src").rglob("*.py"):
        try:
            ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError as e:
            errors.append(f"syntax error {p}: {e}")

    for name, digest in PROMPT_SHA.items():
        got = hashlib.sha256((ROOT / "prompts" / name).read_bytes()).hexdigest()
        if got != digest:
            errors.append(f"prompt hash mismatch {name}: {got} != {digest}")

    _scan_anonymity(errors)

    from bcare.risk import evidencerisk_as, llamaprob_risk  # noqa: F401
    from bcare.evaluation.metrics import auroc, error_capture_expected_tie  # noqa: F401
    from baselines.msp_pe import compute_msp_pe  # noqa: F401
    import yaml

    models = yaml.safe_load((ROOT / "configs/models.yaml").read_text())
    gemma_rev = models["bases"]["Gemma"].get("revision")
    if gemma_rev != "005ad3404e59d6023443cb575daa05336842228a":
        errors.append(f"Gemma revision not pinned to experimental value: {gemma_rev}")
    if models["bases"]["Gemma"].get("model_class") != "Gemma3ForConditionalGeneration":
        errors.append("Gemma must use Gemma3ForConditionalGeneration")
    if models["bases"]["Mistral"].get("attn_implementation") != "eager":
        errors.append("Mistral must set attn_implementation=eager")
    if models["bases"]["Phi"].get("revision") != "2db69c1c3e91a05d2c64a3185acfbaf36f744e25":
        errors.append("Phi revision not pinned to experimental value")
    if models["bases"]["Mistral"].get("revision") != "9527884be6e5616bdd54de542f9ae13384489724":
        errors.append("Mistral revision not pinned to experimental value")
    pt_risk = models.get("decoding", {}).get("p_true", {}).get("risk")
    if pt_risk != "1 - exp(log_p_A)":
        errors.append(f"p_true.risk must match original suite risk_definition: got {pt_risk!r}")
    ver = models.get("verifier_openrouter", {}).get("model")
    if ver != "meta-llama/llama-3.3-70b-instruct":
        errors.append("OpenRouter verifier model id mismatch")
    lp = models.get("llamaprob_self_hosted", {})
    if lp.get("hf_model_id") != "meta-llama/Llama-3.3-70B-Instruct":
        errors.append("LlamaProb self-hosted model id mismatch")
    if lp.get("revision") != "6f6073b423013f6a7d4d9f39144961bfbfbc386b":
        errors.append("LlamaProb revision mismatch")
    covid_targets = yaml.safe_load((ROOT / "configs/datasets.yaml").read_text())["covid"].get(
        "expected_targets", []
    )
    if "Face Masks" not in covid_targets:
        errors.append("COVID expected_targets must use dataset string 'Face Masks' etc.")

    llamaprob = ROOT / "src/bcare/llamaprob/score_closed_set.py"
    if not llamaprob.exists():
        errors.append("missing LlamaProb scorer")
    else:
        got = hashlib.sha256(llamaprob.read_bytes()).hexdigest()
        if got != LLAMAPROB_SHA256:
            errors.append(f"LlamaProb scorer hash mismatch: {got}")
        text = llamaprob.read_text(encoding="utf-8")
        for needle in (
            "meta-llama/Llama-3.3-70B-Instruct",
            "6f6073b423013f6a7d4d9f39144961bfbfbc386b",
            "corrected_fp32_log_softmax",
        ):
            if needle not in text:
                errors.append(f"LlamaProb missing marker: {needle}")

    suite = (ROOT / "scripts/run_local_baseline_suite.py").read_text(encoding="utf-8")
    for needle in ("p_true_risk", "SemanticEntropy", "jlko_get_p_true_logprob", "compute_evse"):
        if needle not in suite:
            errors.append(f"local suite missing real pipeline marker: {needle}")
    if "explanatory note" in suite.lower() or "write a note" in suite.lower():
        errors.append("local suite still appears to write placeholder notes")

    if errors:
        print("FAIL")
        for e in errors:
            print(" -", e)
        return 1
    print("PASS: release sanity checks OK (12-setting B-CARE package, no results tree)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
