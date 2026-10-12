#!/usr/bin/env python3
"""Strict CLI validation for a prepared B-CARE manifest."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bcare.data.validate import ManifestValidationError, validate_manifest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/datasets.yaml")
    ap.add_argument(
        "--dataset-key",
        choices=["semeval", "vaccines", "covid", "pstance"],
        default=None,
        help="If set, enforce eval_n / labels / display name from configs/datasets.yaml",
    )
    ap.add_argument("--enforce-sha256", action="store_true")
    args = ap.parse_args()
    kw = {}
    if args.dataset_key:
        cfg = yaml.safe_load(args.config.read_text())[args.dataset_key]
        display = {
            "semeval": "SemEval",
            "vaccines": "Vaccines",
            "covid": "COVID",
            "pstance": "P-Stance",
        }[args.dataset_key]
        kw = {
            "expected_n": cfg["eval_n"],
            "expected_dataset": display,
            "expected_labels": list(cfg["labels"]),
            "expected_targets": cfg.get("expected_targets"),
            "expected_prompt_kind": cfg.get("prompt_kind"),
            "expected_sha256": cfg.get("manifest_sha256") if args.enforce_sha256 else None,
        }
    try:
        info = validate_manifest(args.manifest, **kw)
    except ManifestValidationError as e:
        print(f"FAIL: {e}")
        return 1
    print("PASS")
    print(json.dumps(info, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
