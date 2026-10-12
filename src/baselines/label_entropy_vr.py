"""Label Entropy and Variation Ratio from M sampled stance labels."""
from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from utils.common import LABELS


def label_entropy(labels: Sequence[str | None], M: int | None = None) -> dict:
    """H = -sum (n_c/n_valid) log(n_c/n_valid) over observed labels (natural log).

    Failed parses excluded from frequency; if none valid → nan.
    """
    parsed = [x for x in labels if x in LABELS]
    n = len(parsed)
    if n == 0:
        return {"LabelEntropy": float("nan"), "n_valid": 0, "counts": {}}
    cnt = Counter(parsed)
    h = 0.0
    for c in LABELS:
        if cnt[c] > 0:
            p = cnt[c] / n
            h -= p * math.log(p)
    return {
        "LabelEntropy": float(h),
        "n_valid": n,
        "counts": {k: int(v) for k, v in cnt.items()},
        "M": M if M is not None else n,
    }


def variation_ratio(labels: Sequence[str | None], M: int) -> dict:
    """VR = 1 - max_c count(c) / M   (Freeman / Gal dropout literature).

    Denominator is M (planned samples), matching common MC dropout VR.
    """
    parsed = [x for x in labels if x in LABELS]
    if not parsed:
        return {"VariationRatio": float("nan"), "n_valid": 0, "mode_count": 0}
    cnt = Counter(parsed)
    mode = max(cnt.values())
    return {
        "VariationRatio": float(1.0 - (mode / float(M))),
        "n_valid": len(parsed),
        "mode_count": int(mode),
        "M": M,
    }
