"""Frozen evidence-risk formulas used by B-CARE (A − S)."""
from __future__ import annotations

from typing import Mapping, Sequence

LABELS_TERNARY = ("FAVOR", "AGAINST", "NONE")
LABELS_BINARY = ("FAVOR", "AGAINST")


def evidence_risk(
    scores: Mapping[str, float],
    yhat: str,
    labels: Sequence[str] = LABELS_TERNARY,
) -> float:
    """EvidenceRisk = max_{c≠ŷ} E_c − E_ŷ (higher ⇒ more error risk)."""
    labs = tuple(labels)
    if yhat not in labs:
        raise KeyError(f"yhat={yhat!r} not in labels={list(labs)}")
    e_y = float(scores[yhat])
    others = [float(scores[lab]) for lab in labs if lab != yhat]
    return float(max(others) - e_y)
