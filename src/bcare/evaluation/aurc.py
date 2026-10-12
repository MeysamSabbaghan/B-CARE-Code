"""AURC / E-AURC (source-faithful excess vs ORACLE).

Primary literature:
  Geifman & El-Yaniv, Selective Classification for Deep Neural Networks (ICML 2017)
  Geifman et al. / Jaeger et al.: Excess-AURC = AURC_method - AURC_oracle

Oracle ranking: sort so all correct examples are kept before any errors
(equivalently: ascending risk where risk=0 for correct, risk=1 for error).

E-AURC MUST NOT subtract a random baseline (that bug produced negatives).
"""
from __future__ import annotations

import numpy as np


def aurc_from_errors(errors_sorted_keep_first: np.ndarray) -> float:
    """AURC given errors already ordered by increasing risk (lowest risk kept first)."""
    errors = np.asarray(errors_sorted_keep_first, dtype=float)
    n = len(errors)
    if n == 0:
        return float("nan")
    cum = np.cumsum(errors)
    ks = np.arange(1, n + 1, dtype=float)
    return float(np.mean(cum / ks))


def aurc_from_risk(
    errors: np.ndarray, risk: np.ndarray, row_ids: np.ndarray | None = None
) -> float:
    errors = np.asarray(errors, dtype=float)
    risk = np.asarray(risk, dtype=float)
    if row_ids is None:
        row_ids = np.arange(len(errors))
    order = np.lexsort((np.asarray(row_ids), risk))  # ascending risk
    return aurc_from_errors(errors[order])


def oracle_aurc(errors: np.ndarray) -> float:
    """Optimal selective ranking: keep all correct first, then errors."""
    errors = np.asarray(errors, dtype=float)
    # risk: 0 for correct, 1 for error → sort ascending puts correct first
    risk = errors.copy()
    row_ids = np.arange(len(errors))
    return aurc_from_risk(errors, risk, row_ids)


def worst_aurc(errors: np.ndarray) -> float:
    """Anti-oracle: keep all errors first."""
    errors = np.asarray(errors, dtype=float)
    risk = 1.0 - errors
    row_ids = np.arange(len(errors))
    return aurc_from_risk(errors, risk, row_ids)


def excess_aurc(
    errors: np.ndarray, risk: np.ndarray, row_ids: np.ndarray | None = None
) -> dict[str, float]:
    """Standard E-AURC = AURC_method - AURC_oracle (>= 0 up to numerics)."""
    a = aurc_from_risk(errors, risk, row_ids)
    o = oracle_aurc(errors)
    return {
        "AURC": a,
        "AURC_oracle": o,
        "E_AURC": float(a - o),
        "AURC_worst": worst_aurc(errors),
    }
