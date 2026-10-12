"""AUROC / AUPRC / ErrorCapture / Enrichment / Random."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

# Predeclared before looking at results
RANDOM_SEEDS = list(range(2026092800, 2026092800 + 50))  # 50 fixed seeds
BUDGETS = (0.10, 0.20, 0.30)


def auroc(errors: np.ndarray, risk: np.ndarray) -> float:
    errors = np.asarray(errors)
    risk = np.asarray(risk, dtype=float)
    if len(np.unique(errors)) < 2:
        return float("nan")
    return float(roc_auc_score(errors, risk))


def auprc(errors: np.ndarray, risk: np.ndarray) -> float:
    return float(average_precision_score(np.asarray(errors), np.asarray(risk, dtype=float)))


def _topk_k(n: int, q: float) -> int:
    """Predeclared rounding: k = max(1, int(round(N * q)))."""
    return max(1, int(round(n * q)))


def error_capture_enrichment(
    errors: np.ndarray, risk: np.ndarray, row_ids: np.ndarray, q: float
) -> dict[str, float]:
    errors = np.asarray(errors)
    risk = np.asarray(risk, dtype=float)
    row_ids = np.asarray(row_ids)
    n = len(errors)
    k = _topk_k(n, q)
    order = np.lexsort((row_ids, -risk))  # highest risk first
    top = order[:k]
    n_err = int(errors.sum())
    err_top = int(errors[top].sum())
    ec = err_top / n_err if n_err else float("nan")
    enrich = ec / q if q > 0 else float("nan")
    return {
        "q": q,
        "k": k,
        "ErrorCapture": float(ec),
        "Enrichment": float(enrich),
        "errors_in_top": err_top,
        "n_errors": n_err,
    }


def random_risk_metrics(errors: np.ndarray, row_ids: np.ndarray) -> dict:
    """Mean±SD over predeclared RANDOM_SEEDS."""
    errors = np.asarray(errors)
    row_ids = np.asarray(row_ids)
    aurocs, auprcs, aurcs = [], [], []
    ecs = {q: [] for q in BUDGETS}
    from metrics.aurc import aurc_from_risk

    for s in RANDOM_SEEDS:
        rng = np.random.default_rng(s)
        risk = rng.random(len(errors))
        aurocs.append(auroc(errors, risk))
        auprcs.append(auprc(errors, risk))
        aurcs.append(aurc_from_risk(errors, risk, row_ids))
        for q in BUDGETS:
            ecs[q].append(error_capture_enrichment(errors, risk, row_ids, q)["ErrorCapture"])
    out = {
        "n_perms": len(RANDOM_SEEDS),
        "seeds": RANDOM_SEEDS,
        "AUROC_mean": float(np.mean(aurocs)),
        "AUROC_std": float(np.std(aurocs)),
        "AUPRC_mean": float(np.mean(auprcs)),
        "AUPRC_std": float(np.std(auprcs)),
        "AURC_mean": float(np.mean(aurcs)),
        "AURC_std": float(np.std(aurcs)),
        "budgets": {},
    }
    for q in BUDGETS:
        arr = np.asarray(ecs[q], dtype=float)
        out["budgets"][f"EC@{int(q*100)}"] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    return out
