"""Synthetic offline tests (NOT experimental results)."""
from __future__ import annotations

import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from baselines.dsv import build_dsv_prompt, dsv_from_true_false_logprobs  # noqa: E402
from baselines.label_entropy_vr import label_entropy, variation_ratio  # noqa: E402
from baselines.msp_pe import compute_msp_pe  # noqa: E402
from baselines.p_true_official import (  # noqa: E402
    build_few_shot_block,
    build_p_true_query_prompt,
    score_p_true_from_logprob_A,
    stance_question,
)
from baselines.semantic_entropy_official import (  # noqa: E402
    get_semantic_ids,
    predictive_entropy_rao,
    semantic_entropy_from_samples,
)
from metrics.aurc import excess_aurc, oracle_aurc  # noqa: E402
from bcare.data.validate import ManifestValidationError, validate_manifest  # noqa: E402
from bcare.evaluation.metrics import (  # noqa: E402
    auprc,
    auroc,
    error_capture_expected_tie,
    full_ranking_metrics,
    midrank_auroc,
    paired_bootstrap_delta_auroc,
)
from bcare.parse_evidence import parse_closed_label, parse_edcv_correctness, parse_evidence_scores  # noqa: E402
from bcare.risk import (  # noqa: E402
    alternative_only_risk,
    cross_base_disagreement,
    edcv_risk,
    evidence_label_disagreement,
    evidence_share_risk,
    evidencerisk_as,
    lexicographic_risk,
    llama_label_disagreement,
    llamaprob_risk,
    same_model_label_disagreement,
    softmax_support_risk,
    support_only_risk,
)


def test_evidencerisk_components_and_ld():
    e = {"FAVOR": 3, "AGAINST": 0, "NONE": 1}
    assert evidencerisk_as(e, "FAVOR") == -2
    assert support_only_risk(e, "FAVOR") == -3
    assert alternative_only_risk(e, "FAVOR") == 1
    assert evidence_share_risk(e, "FAVOR") == 1 - 3 / 4
    # Paper ablation: ΣE=0 → 1 − 1/|C|
    z = {"FAVOR": 0, "AGAINST": 0, "NONE": 0}
    assert abs(evidence_share_risk(z, "FAVOR") - (1.0 - 1.0 / 3)) < 1e-12
    assert 0 <= softmax_support_risk(e, "FAVOR") <= 1
    assert lexicographic_risk(e, "FAVOR") == (-3) * 4 + 1
    assert evidence_label_disagreement(e, "FAVOR") == 0
    assert evidence_label_disagreement({"FAVOR": 2, "AGAINST": 2, "NONE": 0}, "FAVOR") == 1
    with pytest.raises(KeyError):
        evidencerisk_as(e, "NOTALABEL")


def test_disagreement_controls():
    assert cross_base_disagreement("FAVOR", "FAVOR", "FAVOR") == 0
    assert cross_base_disagreement("FAVOR", "NONE", "AGAINST") == 3
    assert same_model_label_disagreement("NONE", "FAVOR") == 1
    assert llama_label_disagreement("FAVOR", "FAVOR") == 0
    assert edcv_risk(3) == 0
    assert abs(llamaprob_risk({"FAVOR": 0.8, "AGAINST": 0.2}, "FAVOR") - 0.2) < 1e-12


def test_parse_evidence_edcv_labels():
    assert parse_evidence_scores('{"FAVOR":1,"AGAINST":2,"NONE":0}')["AGAINST"] == 2
    assert parse_evidence_scores('{"FAVOR":9,"AGAINST":0,"NONE":0}') is None
    # Binary: NONE present ⇒ reject (lib_formulas fidelity)
    assert parse_evidence_scores('{"FAVOR":1,"AGAINST":2,"NONE":0}', ("FAVOR", "AGAINST")) is None
    assert parse_evidence_scores('{"FAVOR":1,"AGAINST":2}', ("FAVOR", "AGAINST"))["FAVOR"] == 1
    assert parse_edcv_correctness("2") == 2
    assert parse_closed_label("FAVOR", ("FAVOR", "AGAINST"))[1] == "ok_exact"


def test_msp_pe_dsv_le_vr():
    out = compute_msp_pe({"FAVOR": -5.0, "AGAINST": -1.0, "NONE": -4.0})
    assert abs(out["sum_probs"] - 1.0) < 1e-9
    # Binary closed set (P-Stance)
    out_b = compute_msp_pe({"FAVOR": -5.0, "AGAINST": -1.0})
    assert abs(out_b["sum_probs"] - 1.0) < 1e-9
    assert set(out_b["class_probs"]) == {"FAVOR", "AGAINST"}
    dsv = dsv_from_true_false_logprobs(-0.1, -2.0)
    assert dsv["official_p_true"] is False
    assert "True or False" in build_dsv_prompt("T", "x", "NONE")
    le = label_entropy(["NONE", "NONE", "FAVOR", "FAVOR", "FAVOR"], M=5)
    vr = variation_ratio(["NONE", "NONE", "FAVOR", "FAVOR", "FAVOR"], M=5)
    assert le["LabelEntropy"] > 0
    assert abs(vr["VariationRatio"] - (1 - 3 / 5)) < 1e-12


def test_ptrue_prompt_and_jlko_risk():
    q = stance_question("Atheism") + " Text: hello"
    prompt = build_p_true_query_prompt(q, "NONE", ["FAVOR", "NONE"], few_shot_prompt="")
    assert "Brainstormed Answers:" in prompt
    assert prompt.endswith("The possible answer is:")
    block = build_few_shot_block(stance_question("Atheism"), ["NONE", "FAVOR"], "NONE", True)
    assert block.endswith(" A")
    scored = score_p_true_from_logprob_A(np.log(0.75), np.log(0.25))
    assert scored["official_p_true"] is True
    assert abs(scored["risk"] - 0.25) < 1e-9
    assert "p_false_fixed" in scored["risk_definition"]


def test_se_clustering_synthetic():
    def check(a, b):
        # treat identical strings as entailment both ways
        return (2, 1.0) if a.split()[-1] == b.split()[-1] else (0, 1.0)

    responses = ["FAVOR", "FAVOR", "AGAINST"]
    seq = [-0.1, -0.2, -1.0]
    out = semantic_entropy_from_samples(responses, seq, check, nli_context_prefix="Q")
    assert "SemanticEntropy" in out
    assert out["n_clusters"] >= 1
    ids = get_semantic_ids(["a FAVOR", "a FAVOR", "a AGAINST"], check)
    assert ids[0] == ids[1]
    assert predictive_entropy_rao([-0.5, -1.0]) >= 0


def test_auroc_auprc_ec_aurc_bootstrap():
    errors = np.array([1, 0, 1, 0, 1, 0])
    risk = np.array([3.0, 0.0, 2.0, 1.0, 2.5, 0.5])
    assert auroc(errors, risk) > 0.5
    assert abs(auroc(errors, risk) - midrank_auroc(errors.tolist(), risk.tolist())) < 1e-15
    assert 0 < auprc(errors, risk) <= 1
    assert 0 <= error_capture_expected_tie(errors, risk, 0.5) <= 1
    m = full_ranking_metrics(errors, risk)
    assert m["AUROC_impl"] == "midrank"
    assert m["ErrorCapture_impl"] == "expected_tie_round_nq"
    # Paper EC: k = round(n*q); n=3,q=1/3 → k=1; top risk tied → 0.5 of the error
    ec = error_capture_expected_tie(
        np.array([1, 0, 0]), np.array([5.0, 5.0, 0.0]), 1 / 3
    )
    assert abs(ec - 0.5) < 1e-12
    # SemEval-sized budget rounding: round(1249*0.2)=250
    assert int(round(1249 * 0.2)) == 250
    # Oracle ranking: risk equals the error indicator → E-AURC ≈ 0
    ex = excess_aurc(errors, errors.astype(float), np.arange(len(errors)))
    assert abs(ex["E_AURC"]) < 1e-12
    assert abs(ex["AURC"] - oracle_aurc(errors)) < 1e-12
    boot = paired_bootstrap_delta_auroc(errors, risk, -risk, B=200, seed=0)
    assert "ci95_low" in boot and boot["B_ok"] > 0
    assert boot["auroc_impl"] == "midrank"


def test_manifest_validation_pass_and_fail():
    fixture = ROOT / "tests/fixtures/synthetic_manifest.csv"
    info = validate_manifest(fixture, expected_n=2, expected_dataset="SemEval")
    assert info["ok"] is True
    with pytest.raises(ManifestValidationError):
        validate_manifest(fixture, expected_n=999)


def test_fixture_marked_synthetic():
    data = json.loads((ROOT / "tests/fixtures/synthetic_example.json").read_text())
    assert data["synthetic"] is True


def test_llamaprob_scorer_present_and_constants():
    p = ROOT / "src/bcare/llamaprob/score_closed_set.py"
    text = p.read_text(encoding="utf-8")
    assert "meta-llama/Llama-3.3-70B-Instruct" in text
    assert "6f6073b423013f6a7d4d9f39144961bfbfbc386b" in text
    assert "corrected_fp32_log_softmax" in text
    assert "teacher_forced" in text


def test_per_example_join_evidencerisk_and_llamaprob():
    """Synthetic join: gold × base × PBCE/LlamaProb → aligned risk vectors."""
    gold = {
        "1": {"gold": "FAVOR", "label_space": "FAVOR|AGAINST|NONE"},
        "2": {"gold": "NONE", "label_space": "FAVOR|AGAINST|NONE"},
    }
    base = {"1": {"base_pred": "FAVOR"}, "2": {"base_pred": "AGAINST"}}
    pbce = {
        "1": {"evidence": {"FAVOR": 3, "AGAINST": 0, "NONE": 1}},
        "2": {"evidence": {"FAVOR": 1, "AGAINST": 2, "NONE": 0}},
    }
    lp = {
        "1": {"P": {"FAVOR": 0.7, "AGAINST": 0.2, "NONE": 0.1}},
        "2": {"P": {"FAVOR": 0.1, "AGAINST": 0.6, "NONE": 0.3}},
    }
    errors, er_risks, lp_risks = [], [], []
    for eid, g in gold.items():
        yhat = base[eid]["base_pred"]
        labels = tuple(g["label_space"].split("|"))
        errors.append(int(yhat != g["gold"]))
        er_risks.append(evidencerisk_as(pbce[eid]["evidence"], yhat, labels))
        lp_risks.append(llamaprob_risk(lp[eid]["P"], yhat))
    assert len(errors) == 2
    assert er_risks[0] == -2
    assert abs(lp_risks[0] - 0.3) < 1e-12
    m = full_ranking_metrics(np.asarray(errors), np.asarray(er_risks))
    assert "AUROC" in m


def test_suite_script_wires_real_ptrue_se_evse():
    text = (ROOT / "scripts/run_local_baseline_suite.py").read_text(encoding="utf-8")
    assert "jlko_get_p_true_logprob" in text
    assert 'rec["p_true_risk"]' in text or "rec[\"p_true_risk\"]" in text
    assert "semantic_entropy_from_samples" in text
    assert "compute_evse" in text
    assert "explanatory note" not in text.lower()
