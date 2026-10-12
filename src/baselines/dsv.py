"""Direct Self-Verification (DSV) — LEGACY custom method (formerly mislabeled p(True)).

PRESERVE UNCHANGED from semeval_phi_local_full_run1.

Definition: model directly evaluates whether its own predicted stance is correct
via a zero-shot True/False prompt; risk = 1 - renormalized P(True).

This is NOT official Kadavath/jlko p(True). Never label this as p(True).
"""
from __future__ import annotations

import math

from utils.common import logsumexp

DSV_PROMPT_TEMPLATE = (
    "Question: What is the stance of the text toward '{target}'?\n"
    "Text: {text}\n"
    "Proposed answer: {cand}\n"
    "Is the proposed answer correct? Answer True or False."
)

DSV_NAME = "Direct Self-Verification (DSV)"
DSV_INTERNAL = "direct_self_verification"


def build_dsv_prompt(target: str, text: str, candidate: str) -> str:
    return DSV_PROMPT_TEMPLATE.format(target=target, text=text, cand=candidate)


def dsv_from_true_false_logprobs(lp_true: float, lp_false: float) -> dict:
    z = logsumexp([lp_true, lp_false])
    p_true = float(math.exp(lp_true - z))
    p_false = float(math.exp(lp_false - z))
    return {
        "method": DSV_INTERNAL,
        "display_name": DSV_NAME,
        "p_true": p_true,
        "p_false": p_false,
        "risk": float(1.0 - p_true),
        "normalization": "softmax_over_True_False_strings",
        "official_p_true": False,
    }
