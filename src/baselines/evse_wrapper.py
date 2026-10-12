"""EVSE using official lucieK-J hierarchy + Jousselme, with official NLI checkpoint.

Sources:
  Kunitomo-Jacquin et al., Evidential Semantic Entropy (EACL 2026)
  github.com/lucieK-J/EvidentialSemanticEntropy @ ba2aefe
  Vendored SE entailment uses microsoft/deberta-v2-xlarge-mnli
  (NOT deberta-large-mnli).

Our previous run used deberta-large-mnli — that was a deviation; corrected
config uses deberta-v2-xlarge-mnli. If VRAM is tight: unload Phi, run NLI
sequentially (never silently substitute a smaller NLI).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable

# Local adapted core (same math as official); NLI checkpoint is external.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from evse_core import (  # noqa: E402
    all_entailments,
    build_frame_of_discernment,
    build_hierarchy,
    jousselme_entropy,
    mass_assignment,
)

OFFICIAL_NLI = "microsoft/deberta-v2-xlarge-mnli"


def compute_evse(
    responses: list[str],
    log_liks_with_eos: list[list[float]],
    check_implication: Callable[[str, str], tuple[int, float]],
    nli_strings: list[str] | None = None,
) -> dict:
    unique_responses = list(dict.fromkeys(responses))
    strings = nli_strings if nli_strings is not None else unique_responses
    # Map unique response order for entailment matrix on unique set
    # Official operates on unique answer strings present.
    uniq_nli = []
    seen = set()
    for r, s in zip(responses, strings if nli_strings else responses):
        if r not in seen:
            seen.add(r)
            uniq_nli.append(s if nli_strings else r)
    # Align: unique_responses order
    uniq_nli = []
    for ur in unique_responses:
        # find first matching nli string
        if nli_strings is None:
            uniq_nli.append(ur)
        else:
            for r, s in zip(responses, nli_strings):
                if r == ur:
                    uniq_nli.append(s)
                    break
    ent_matrix = all_entailments(uniq_nli, check_implication)
    hierarchy_info = build_hierarchy(ent_matrix, unique_responses)
    hierarchy_info["unique_responses"] = unique_responses
    discernment_info = build_frame_of_discernment(hierarchy_info)
    mass = mass_assignment(log_liks_with_eos, responses, hierarchy_info, discernment_info)
    evse = float(jousselme_entropy(mass["F"], mass["M"], discernment_info["Omega_size"]))
    return {
        "EVSE": evse,
        "m_theta": mass["m_theta"],
        "cluster_belonging": hierarchy_info["cluster_belonging"],
        "unique_responses": unique_responses,
        "frame_omega_size": discernment_info["Omega_size"],
        "nli_model": OFFICIAL_NLI,
        "mass": {k: mass[k] for k in ("F", "M", "m_theta", "observed_mass")},
    }
