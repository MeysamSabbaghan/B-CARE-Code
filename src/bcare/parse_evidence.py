"""Parse ordinal evidence JSON and EDCV integer outputs.

Faithful to paper-primary parsers in lib_formulas.py:
  - JSON object extracted with {[^{}]*} (non-greedy nested-free)
  - Ternary requires FAVOR/AGAINST/NONE ∈ {0,1,2,3}
  - Binary requires FAVOR/AGAINST only; presence of NONE ⇒ reject
"""
from __future__ import annotations

import json
import re


_JSON_OBJ = re.compile(r"\{[^{}]*\}", flags=re.S)


def parse_evidence_scores(raw: str, labels=("FAVOR", "AGAINST", "NONE")):
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    m = _JSON_OBJ.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    labs = tuple(labels)
    # Binary fidelity: reject if NONE present when not in label space
    if "NONE" not in labs and "NONE" in obj:
        return None
    out = {}
    for lab in labs:
        if lab not in obj:
            return None
        try:
            v = int(obj[lab])
        except (TypeError, ValueError):
            return None
        if v not in (0, 1, 2, 3):
            return None
        out[lab] = v
    return out


def parse_edcv_correctness(raw: str):
    if raw is None:
        return None
    text = str(raw).strip()
    if re.fullmatch(r"[0-3]", text):
        return int(text)
    m = re.search(r"(?<!\d)([0-3])(?!\d)", text)
    return int(m.group(1)) if m else None


def parse_closed_label(raw: str, labels=("FAVOR", "AGAINST", "NONE")):
    if raw is None:
        return None, "fail_empty"
    text = str(raw).strip()
    if not text:
        return None, "fail_empty"
    cleaned = text.strip().strip("`\"'.").strip().upper()
    if cleaned in labels:
        return cleaned, "ok_exact"
    hits = [
        m.group(1).upper()
        for m in re.finditer(r"\b(" + "|".join(labels) + r")\b", text, flags=re.I)
    ]
    uniq = []
    for h in hits:
        if h not in uniq:
            uniq.append(h)
    if len(uniq) == 1:
        return uniq[0], "ok_extracted"
    if len(uniq) > 1:
        return None, "fail_ambiguous"
    return None, "fail_no_label"
