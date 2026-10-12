"""Evidence-theory utils for EVSE (adapted from lucieK-J/EvidentialSemanticEntropy)."""

from __future__ import annotations

import math


def my_bin2dec(bin_vect: list[int]) -> int:
    bin_str = "".join(str(item) for item in reversed(bin_vect))
    return int(bin_str, 2) if bin_str else 0


def vector_to_set_expression(binary_vector: list[int]) -> str:
    if all(x == 1 for x in binary_vector):
        return r"\Theta"
    elements = [rf"\theta_{{{index + 1}}}" for index, value in enumerate(binary_vector) if value == 1]
    return r"\{" + ", ".join(elements) + r"\}"


def calculate_pignistic_probability(F: list[int], M: list[float], frame_size: int) -> list[float]:
    bet_p = [0.0] * frame_size
    for focal_set, mass in zip(F, M):
        if mass <= 0:
            continue
        binary_representation = bin(focal_set)[2:].zfill(frame_size)
        subset = {i for i, bit in enumerate(reversed(binary_representation)) if bit == "1"}
        subset_size = len(subset)
        if subset_size <= 0:
            continue
        contribute = mass / subset_size
        for element in subset:
            bet_p[element] += contribute
    return bet_p


def jousselme_entropy(F: list[int], M: list[float], frame_size: int) -> float:
    """Ambiguity measure / Jousselme entropy of the bba (bits, log2) — paper Eq. 8."""
    bet_p = calculate_pignistic_probability(F, M, frame_size)
    entropy = 0.0
    for prob in bet_p:
        if prob > 0:
            entropy += prob * math.log(1.0 / prob, 2)
    return float(entropy)
