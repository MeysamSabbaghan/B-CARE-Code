"""Official Farquhar / jlko Semantic Entropy (bidirectional entailment).

Sources:
  Farquhar et al., Nature 2024
  github.com/jlko/semantic_uncertainty
    semantic_entropy.py :: get_semantic_ids, logsumexp_by_id, predictive_entropy_rao
  NLI: microsoft/deberta-v2-xlarge-mnli (EntailmentDeberta)

NOT EVSE hierarchy clustering.

Stance note: generations are constrained stance labels. Bidirectional entailment
will typically place distinct labels in distinct clusters (contradiction), so SE
often degenerates toward likelihood-weighted label entropy. That degeneracy is
reported, not papered over. Method remains named Semantic Entropy when using
official clustering+NLI+entropy equations on the sampled strings.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def get_semantic_ids(
    strings_list: list[str],
    check_implication: Callable[[str, str], tuple[int, float]],
    strict_entailment: bool = False,
) -> list[int]:
    """Faithful port of jlko get_semantic_ids (bidirectional equivalence)."""

    def are_equivalent(text1: str, text2: str) -> bool:
        implication_1, _ = check_implication(text1, text2)
        implication_2, _ = check_implication(text2, text1)
        assert implication_1 in (0, 1, 2) and implication_2 in (0, 1, 2)
        if strict_entailment:
            return implication_1 == 2 and implication_2 == 2
        implications = [implication_1, implication_2]
        return (0 not in implications) and (implications != [1, 1])

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1
    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(
    semantic_ids: Sequence[int], log_likelihoods: Sequence[float], agg: str = "sum_normalized"
) -> tuple[list[int], list[float]]:
    """Faithful port of jlko logsumexp_by_id."""
    unique_ids = sorted(set(semantic_ids))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []
    log_likelihoods = list(log_likelihoods)
    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = np.array([log_likelihoods[i] for i in id_indices], dtype=float)
        if agg == "sum_normalized":
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = float(np.log(np.sum(np.exp(log_lik_norm))))
        else:
            raise ValueError(agg)
        log_likelihood_per_semantic_id.append(logsumexp_value)
    return unique_ids, log_likelihood_per_semantic_id


def predictive_entropy_rao(log_probs: Sequence[float]) -> float:
    """Faithful jlko predictive_entropy_rao: -sum exp(lp)*lp."""
    log_probs = np.asarray(log_probs, dtype=float)
    return float(-np.sum(np.exp(log_probs) * log_probs))


def cluster_assignment_entropy(semantic_ids: Sequence[int]) -> float:
    """Discrete SE without token likelihoods (jlko)."""
    n = len(semantic_ids)
    counts = np.bincount(np.asarray(semantic_ids, dtype=int))
    probabilities = counts / n
    assert np.isclose(probabilities.sum(), 1.0)
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def semantic_entropy_from_samples(
    responses: list[str],
    seq_logprobs: list[float],
    check_implication: Callable[[str, str], tuple[int, float]],
    nli_context_prefix: str,
) -> dict:
    """Compute official SE.

    NLI strings = f\"{nli_context_prefix} {response}\" matching jlko habit of
    comparing answer strings (we include question context for stance disambiguation).
    """
    assert len(responses) == len(seq_logprobs)
    nli_strings = [f"{nli_context_prefix} {r}" for r in responses]
    sem_ids = get_semantic_ids(nli_strings, check_implication, strict_entailment=False)
    # log_likelihoods in jlko are typically mean token loglik or sum — SE paper uses
    # sequence log-likelihoods normalized in logsumexp_by_id across generations.
    _, ll_per_id = logsumexp_by_id(sem_ids, seq_logprobs, agg="sum_normalized")
    se = predictive_entropy_rao(ll_per_id)
    cae = cluster_assignment_entropy(sem_ids)
    n_clusters = len(set(sem_ids))
    n_unique_resp = len(set(responses))
    return {
        "SemanticEntropy": float(se),
        "cluster_assignment_entropy": float(cae),
        "semantic_ids": list(sem_ids),
        "n_clusters": n_clusters,
        "n_unique_responses": n_unique_resp,
        "degeneracy_toward_label_entropy": bool(
            n_unique_resp <= 3 and n_clusters == n_unique_resp
        ),
        "nli_model": "microsoft/deberta-v2-xlarge-mnli",
        "clustering": "jlko_bidirectional_entailment",
        "length_normalization": False,
    }
