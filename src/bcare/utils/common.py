"""Shared helpers for the B-CARE experimental suite."""
from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

LABELS = ("FAVOR", "AGAINST", "NONE")

# Frozen Base prompt SHA (canonical no-hint)
BASE_PROMPT_SHA256 = "9f176afe4516771789a3706861bcfd2d71a77b8766b53f2e8aa0e93d3c151825"
VERIFIER_SHA256 = "3598de6b1347260d5b05dc14c001b52055393f1c6dfc5042b044d2eefeb82a00"
DETECTOR_SHA256 = "99fc4ce5026f58bbee50a2863130adb8116af503df2eb3cf3029afefa15fbf7d"
MANIFEST_SHA256 = "6f12604c6d06ceef2825758ba278f54ecc0a47307e1f28bf478514f99f15a4f5"
PHI_REVISION = "2db69c1c3e91a05d2c64a3185acfbaf36f744e25"
MISTRAL_REVISION = "9527884be6e5616bdd54de542f9ae13384489724"
GEMMA_REVISION = "005ad3404e59d6023443cb575daa05336842228a"
NLI_SE_EVSE_ID = "microsoft/deberta-v2-xlarge-mnli"
EXPECTED_REVISIONS = {
    "microsoft/phi-4": PHI_REVISION,
    "mistralai/Mistral-Small-24B-Instruct-2501": MISTRAL_REVISION,
    "google/gemma-3-27b-it": GEMMA_REVISION,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x)))))


def parse_label(raw: str | None) -> tuple[str | None, str]:
    if raw is None:
        return None, "fail_empty"
    text = str(raw).strip()
    if not text:
        return None, "fail_empty"
    cleaned = text.strip().strip("`\"'.").strip().upper()
    if cleaned in LABELS:
        return cleaned, "ok_exact"
    hits = [
        m.group(1).upper()
        for m in re.finditer(r"\b(FAVOR|AGAINST|NONE)\b", text, flags=re.IGNORECASE)
    ]
    uniq: list[str] = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    if len(uniq) == 1:
        return uniq[0], "ok_extracted"
    if len(uniq) > 1:
        return None, "fail_ambiguous"
    return None, "fail_no_label"


def parse_evidence_scores(raw: str | None) -> tuple[dict[str, int] | None, str]:
    import json

    if raw is None:
        return None, "empty"
    text = str(raw).strip()
    if not text:
        return None, "empty"
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", text, flags=re.S)
        if not m:
            return None, "no_json"
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None, "json_parse_fail"
    if not isinstance(obj, dict):
        return None, "not_dict"
    norm: dict[str, Any] = {}
    for k, v in obj.items():
        kk = str(k).strip().upper()
        if kk in LABELS:
            norm[kk] = v
        elif kk.startswith("E_") and kk[2:] in LABELS:
            norm[kk[2:]] = v
    if set(norm.keys()) != set(LABELS):
        return None, f"missing_keys:{sorted(norm.keys())}"
    out: dict[str, int] = {}
    for lab in LABELS:
        try:
            val = int(norm[lab])
        except Exception:
            return None, f"non_int:{lab}"
        if val not in (0, 1, 2, 3):
            return None, f"out_of_range:{lab}={val}"
        out[lab] = val
    return out, "ok"


def render_template(tmpl: str, target: str, text: str) -> str:
    return tmpl.replace("{target}", target).replace("{text}", text)
