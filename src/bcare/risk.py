"""B-CARE and control risk scores (higher = more likely error)."""
from __future__ import annotations

import math
from typing import Mapping, Sequence

TERNARY = ("FAVOR", "AGAINST", "NONE")
BINARY = ("FAVOR", "AGAINST")


def support_and_alternative(
    evidence: Mapping[str, float], yhat: str, labels: Sequence[str] = TERNARY
) -> tuple[float, float]:
    labs = tuple(labels)
    if yhat not in labs:
        raise KeyError(f"yhat={yhat!r} not in labels={list(labs)}")
    if yhat not in evidence:
        raise KeyError(f"yhat={yhat!r} not in evidence={list(evidence)}")
    s = float(evidence[yhat])
    a = max(float(evidence[c]) for c in labs if c != yhat)
    return s, a


def evidencerisk_as(evidence, yhat, labels=TERNARY) -> float:
    s, a = support_and_alternative(evidence, yhat, labels)
    return a - s


def support_only_risk(evidence, yhat, labels=TERNARY) -> float:
    s, _ = support_and_alternative(evidence, yhat, labels)
    return -s


def alternative_only_risk(evidence, yhat, labels=TERNARY) -> float:
    _, a = support_and_alternative(evidence, yhat, labels)
    return a


def evidence_share_risk(evidence, yhat, labels=TERNARY) -> float:
    """Risk = 1 − E_ŷ/ΣE; if ΣE=0 return 1−1/|C| (paper ablation script)."""
    labs = tuple(labels)
    if yhat not in labs:
        raise KeyError(f"yhat={yhat!r} not in labels={list(labs)}")
    total = sum(float(evidence[c]) for c in labs)
    nlab = len(labs)
    if total <= 0:
        return 1.0 - (1.0 / nlab)
    return 1.0 - float(evidence[yhat]) / total


def softmax_support_risk(evidence, yhat, labels=TERNARY, temperature: float = 1.0) -> float:
    labs = tuple(labels)
    if yhat not in labs:
        raise KeyError(f"yhat={yhat!r} not in labels={list(labs)}")
    t = float(temperature)
    xs = [float(evidence[c]) / t for c in labs]
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    z = sum(exps)
    p = {c: e / z for c, e in zip(labs, exps)}
    return 1.0 - p[yhat]


def lexicographic_risk(evidence, yhat, labels=TERNARY, a_stride: int = 4) -> float:
    s, a = support_and_alternative(evidence, yhat, labels)
    return (-s) * float(a_stride) + a


def evidence_label_disagreement(evidence, yhat, labels=TERNARY) -> int:
    """LD: 1 unless yhat is unique argmax (ties ⇒ disagree)."""
    labs = tuple(labels)
    if yhat not in labs:
        return 1
    vals = {c: float(evidence[c]) for c in labs}
    mx = max(vals.values())
    argmaxes = [c for c, v in vals.items() if v == mx]
    if len(argmaxes) != 1:
        return 1
    return 0 if argmaxes[0] == yhat else 1


def same_model_label_disagreement(same_model_hard_label: str, yhat: str) -> int:
    """Binary disagreement between same-model hard stance and base ŷ."""
    return int(str(same_model_hard_label) != str(yhat))


def llama_label_disagreement(llama_hard_label: str, yhat: str) -> int:
    """Binary disagreement between Fixed-Llama hard stance and base ŷ."""
    return int(str(llama_hard_label) != str(yhat))


def cross_base_disagreement(yhat: str, other1: str, other2: str) -> int:
    n_match = int(other1 == yhat) + int(other2 == yhat)
    if n_match == 2:
        return 0
    if n_match == 1:
        return 1
    if other1 == other2:
        return 2
    return 3


def edcv_risk(correctness: int) -> int:
    c = int(correctness)
    if c < 0 or c > 3:
        raise ValueError(f"EDCV correctness out of range: {c}")
    return 3 - c


def llamaprob_risk(probs: Mapping[str, float], yhat: str) -> float:
    """LlamaProb risk = 1 − P(ŷ) over closed-set probabilities."""
    if yhat not in probs:
        raise KeyError(f"yhat {yhat!r} missing from LlamaProb vector")
    return 1.0 - float(probs[yhat])
