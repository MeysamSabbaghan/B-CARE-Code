"""Synthetic end-to-end smoke (no GPU/API): prepare→risks→metrics join."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcare.evaluation.metrics import full_ranking_metrics  # noqa: E402
from bcare.risk import (  # noqa: E402
    cross_base_disagreement,
    evidencerisk_as,
    llamaprob_risk,
)


def test_cli_scripts_exist_and_parse():
    scripts = [
        "prepare_data.py",
        "validate_manifest.py",
        "run_base_models.py",
        "run_local_baseline_suite.py",
        "run_bcare.py",
        "run_baselines.py",
        "run_ablations.py",
        "run_llamaprob.py",
        "evaluate.py",
        "generate_paper_outputs.py",
        "verify_release.py",
    ]
    for name in scripts:
        p = ROOT / "scripts" / name
        assert p.exists(), name
        # --help should exit 0 without GPU
        r = subprocess.run(
            [sys.executable, str(p), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, f"{name} --help failed: {r.stderr}"


def test_synthetic_join_pipeline_smoke(tmp_path: Path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "example_id",
                "uid",
                "dataset",
                "target",
                "text",
                "text_sha256",
                "gold",
                "label_space",
                "prompt_kind",
            ],
        )
        w.writeheader()
        for i, (gold, yhat, ev, P) in enumerate(
            [
                ("FAVOR", "FAVOR", {"FAVOR": 3, "AGAINST": 0, "NONE": 1}, {"FAVOR": 0.8, "AGAINST": 0.1, "NONE": 0.1}),
                ("NONE", "AGAINST", {"FAVOR": 1, "AGAINST": 2, "NONE": 0}, {"FAVOR": 0.2, "AGAINST": 0.5, "NONE": 0.3}),
                ("AGAINST", "AGAINST", {"FAVOR": 0, "AGAINST": 3, "NONE": 1}, {"FAVOR": 0.1, "AGAINST": 0.7, "NONE": 0.2}),
            ]
        ):
            w.writerow(
                {
                    "example_id": str(i),
                    "uid": str(i),
                    "dataset": "SemEval",
                    "target": "Atheism",
                    "text": f"synthetic {i}",
                    "text_sha256": "a" * 64,
                    "gold": gold,
                    "label_space": "FAVOR|AGAINST|NONE",
                    "prompt_kind": "ternary",
                }
            )

    # Simulate outputs
    base = {
        "0": {"base_pred": "FAVOR"},
        "1": {"base_pred": "AGAINST"},
        "2": {"base_pred": "AGAINST"},
    }
    pbce = {
        "0": {"evidence": {"FAVOR": 3, "AGAINST": 0, "NONE": 1}},
        "1": {"evidence": {"FAVOR": 1, "AGAINST": 2, "NONE": 0}},
        "2": {"evidence": {"FAVOR": 0, "AGAINST": 3, "NONE": 1}},
    }
    lp = {
        "0": {"P": {"FAVOR": 0.8, "AGAINST": 0.1, "NONE": 0.1}},
        "1": {"P": {"FAVOR": 0.2, "AGAINST": 0.5, "NONE": 0.3}},
        "2": {"P": {"FAVOR": 0.1, "AGAINST": 0.7, "NONE": 0.2}},
    }
    other = {"0": "FAVOR", "1": "NONE", "2": "AGAINST"}
    other2 = {"0": "FAVOR", "1": "FAVOR", "2": "NONE"}

    gold = {str(r["example_id"]): r for r in csv.DictReader(manifest.open())}
    errors, er, lpr, cbd = [], [], [], []
    for eid, g in gold.items():
        yhat = base[eid]["base_pred"]
        labels = tuple(g["label_space"].split("|"))
        errors.append(int(yhat != g["gold"]))
        er.append(evidencerisk_as(pbce[eid]["evidence"], yhat, labels))
        lpr.append(llamaprob_risk(lp[eid]["P"], yhat))
        cbd.append(cross_base_disagreement(yhat, other[eid], other2[eid]))

    m_er = full_ranking_metrics(np.asarray(errors), np.asarray(er))
    m_lp = full_ranking_metrics(np.asarray(errors), np.asarray(lpr))
    assert m_er["n"] == 3
    assert "AUROC" in m_er and "ErrorCapture@20" in m_er
    assert "AUROC" in m_lp
    assert all(isinstance(x, int) for x in cbd)

    # Ablation CLI offline
    base_p = tmp_path / "base.jsonl"
    pbce_p = tmp_path / "pbce.jsonl"
    with base_p.open("w") as f:
        for eid, r in base.items():
            f.write(json.dumps({"example_id": eid, **r}) + "\n")
    with pbce_p.open("w") as f:
        for eid, r in pbce.items():
            f.write(json.dumps({"example_id": eid, **r}) + "\n")
    out = tmp_path / "abl.jsonl"
    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_ablations.py"),
            "--manifest",
            str(manifest),
            "--base-preds",
            str(base_p),
            "--pbce",
            str(pbce_p),
            "--out",
            str(out),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(lines) == 3
    assert "ComparativeRisk" in lines[0]
    assert "EvidenceShareRisk" in lines[0]
