"""Paper-primary ranking metrics.

AUROC: midrank Mann–Whitney (identical to the primary table scripts in
`lib_formulas.midrank_auroc`). sklearn.roc_auc_score is numerically equivalent
for binary labels but is NOT the paper source implementation.

ErrorCapture: expected-tie with budget k = max(1, round(n·q)) matching
`lib_formulas.error_capture`.

AUPRC: tie-aware step integration matching `lib_formulas.auprc`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


def midrank_auroc(y: Sequence[int], s: Sequence[float]) -> float:
    """Exact port of paper-primary midrank AUROC."""
    y = list(y)
    s = list(s)
    n, npos = len(y), int(sum(y))
    nneg = n - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    order = sorted(range(n), key=lambda i: s[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        avg = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return float(
        (sum(ranks[i] for i in range(n) if y[i] == 1) - npos * (npos + 1) / 2.0)
        / (npos * nneg)
    )


def auroc(errors, risk) -> float:
    """Paper-primary AUROC (midrank)."""
    errors = np.asarray(errors, dtype=int)
    risk = np.asarray(risk, dtype=float)
    return midrank_auroc(errors.tolist(), risk.tolist())


def auprc(errors, risk) -> float:
    """Paper-primary AUPRC (tie-aware PR curve integration)."""
    y = [int(v) for v in np.asarray(errors).tolist()]
    s = [float(v) for v in np.asarray(risk, dtype=float).tolist()]
    npos = sum(y)
    if npos == 0:
        return float("nan")
    order = sorted(range(len(y)), key=lambda i: -s[i])
    tp = fp = 0.0
    points = [(0.0, 1.0)]
    for i in order:
        if y[i] == 1:
            tp += 1
        else:
            fp += 1
        points.append((tp / npos, tp / max(tp + fp, 1.0)))
    area = 0.0
    for (r0, p0), (r1, p1) in zip(points, points[1:]):
        area += (r1 - r0) * 0.5 * (p0 + p1)
    return float(area)


def error_capture_expected_tie(errors, risk, q: float) -> float:
    """Expected-tie ErrorCapture with k = round(n·q) (paper-primary)."""
    y = [int(v) for v in np.asarray(errors).tolist()]
    s = [float(v) for v in np.asarray(risk, dtype=float).tolist()]
    n, nerr = len(y), sum(y)
    if nerr == 0:
        return float("nan")
    kk = int(round(n * float(q)))
    if kk <= 0:
        return 0.0
    if kk >= n:
        return 1.0
    order = sorted(range(n), key=lambda i: -s[i])
    sel = 0.0
    rem = kk
    i = 0
    while i < n and rem > 0:
        j = i
        while j < n and s[order[j]] == s[order[i]]:
            j += 1
        g = j - i
        ey = sum(y[order[k]] for k in range(i, j))
        if g <= rem:
            sel += ey
            rem -= g
        else:
            sel += rem * (ey / g)
            rem = 0
        i = j
    return float(sel / nerr)


def paired_bootstrap_delta_auroc(
    errors, risk_a, risk_b, *, B: int = 10000, seed: int = 0
) -> dict:
    """Paired bootstrap of AUROC(A) − AUROC(B) on identical rows (B=10000 default)."""
    rng = np.random.default_rng(seed)
    errors = np.asarray(errors, dtype=int)
    risk_a = np.asarray(risk_a, dtype=float)
    risk_b = np.asarray(risk_b, dtype=float)
    n = len(errors)
    point = auroc(errors, risk_a) - auroc(errors, risk_b)
    deltas = []
    for _ in range(int(B)):
        idx = rng.integers(0, n, size=n)
        d = auroc(errors[idx], risk_a[idx]) - auroc(errors[idx], risk_b[idx])
        if np.isfinite(d):
            deltas.append(d)
    arr = np.asarray(deltas, dtype=float)
    lo, hi = (
        np.quantile(arr, [0.025, 0.975]) if len(arr) else (float("nan"), float("nan"))
    )
    return {
        "delta_auroc": float(point),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0)
        if np.isfinite(lo) and np.isfinite(hi)
        else False,
        "B_ok": int(len(arr)),
        "B": int(B),
        "seed": int(seed),
        "auroc_impl": "midrank",
    }


def full_ranking_metrics(errors, risk, q_list=(0.10, 0.20, 0.30)) -> dict:
    out = {
        "AUROC": auroc(errors, risk),
        "AUPRC": auprc(errors, risk),
        "AUROC_impl": "midrank",
        "ErrorCapture_impl": "expected_tie_round_nq",
        "n": int(len(errors)),
        "n_errors": int(np.asarray(errors).sum()),
    }
    for q in q_list:
        out[f"ErrorCapture@{int(q * 100)}"] = error_capture_expected_tie(errors, risk, q)
    return out
