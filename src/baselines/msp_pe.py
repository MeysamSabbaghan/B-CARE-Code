"""MSP and Predictive Entropy for multi-token stance verbalizers.

Source notes
------------
Hendrycks & Gimpel (ICLR 2017): MSP = max_c p(c|x) for a discriminative softmax.

For autoregressive LMs without a class head, the source-supported adaptation used
in Farquhar et al. / jlko semantic_uncertainty sequence likelihoods is:

  score(c|x) = sum_t log p(token_t | prefix, tokens_<t)   # NO length normalization

Official jlko code aggregates token log-likelihoods by summation (see model.predict
transition scores; EVSE mass_assignment explicitly: "no length-norm").

We therefore:
  1) teacher-force each label string FAVOR / AGAINST / NONE
  2) score_c = sum token logprobs (not length-normalized — do NOT invent a fix)
  3) p(c|x) = softmax_c(score_c) over the closed 3-class set
  4) MSP_confidence = max p(c); MSP_risk = 1 - MSP_confidence
  5) PE = -sum p log p  (natural log; class entropy, not vocab entropy)

Unequal tokenization (FAVOR=3, AGAINST=2, NONE=1) is an inherent property of these
verbalizers under the Phi tokenizer; we document it rather than invent length-norm.
"""
from __future__ import annotations

import math
from typing import Callable

from utils.common import LABELS, logsumexp


def class_probs_from_seq_logprobs(seq_logprobs: dict[str, float]) -> dict[str, float]:
    # Preserve original ternary iteration order when the closed set is FAVOR|AGAINST|NONE.
    # For binary P-Stance, use the provided keys (scientific closed-set over label_space).
    if set(seq_logprobs.keys()) == set(LABELS):
        labs = LABELS
    else:
        labs = tuple(seq_logprobs.keys())
    lps = [float(seq_logprobs[lab]) for lab in labs]
    z = logsumexp(lps)
    probs = {lab: float(math.exp(seq_logprobs[lab] - z)) for lab in labs}
    s = sum(probs.values())
    if abs(s - 1.0) > 1e-5:
        probs = {k: v / s for k, v in probs.items()}
    return probs


def msp_risk(probs: dict[str, float]) -> float:
    return float(1.0 - max(probs.values()))


def predictive_entropy(probs: dict[str, float], *, eps: float = 0.0) -> float:
    h = 0.0
    for p in probs.values():
        if p > eps:
            h -= p * math.log(p)
    return float(h)


def compute_msp_pe(seq_logprobs: dict[str, float]) -> dict:
    probs = class_probs_from_seq_logprobs(seq_logprobs)
    return {
        "class_probs": probs,
        "MSP_confidence": float(max(probs.values())),
        "MSP_risk": msp_risk(probs),
        "PredictiveEntropy": predictive_entropy(probs),
        "sum_probs": float(sum(probs.values())),
        "length_normalization": False,
        "score_definition": "sum_token_logprob_then_closed_set_softmax",
    }
