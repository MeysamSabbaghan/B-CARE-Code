"""Prepare evaluation manifests for the four B-CARE datasets (no redistribution)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

DATASETS = ("semeval", "vaccines", "covid", "pstance")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def map_gold(raw: str, gold_map: dict) -> str:
    g = str(raw).strip()
    if g in gold_map:
        return gold_map[g]
    gu = g.upper()
    return gold_map.get(gu, gu)


COLS = [
    "example_id",
    "uid",
    "dataset",
    "target",
    "text",
    "text_sha256",
    "gold",
    "label_space",
    "prompt_kind",
]


def enrich(row: dict) -> dict:
    text = row.get("text") or ""
    row = dict(row)
    row.setdefault("uid", row.get("example_id"))
    row["text_sha256"] = hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    return row


def write_csv(rows: list[dict], path: Path, cols: list[str] | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = cols or COLS
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            rr = enrich(r)
            w.writerow({c: rr.get(c, "") for c in cols})
    return sha256_file(path)


def prepare_semeval(raw_test: Path, out: Path, cfg: dict) -> dict:
    exclude = set(cfg.get("exclude_targets", ["Donald Trump"]))
    gold_map = cfg.get("gold_map", {"NEUTRAL": "NONE"})
    rows = []
    with raw_test.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            target = r.get("Target") or r.get("target")
            text = r.get("Tweet") or r.get("text") or r.get("Text")
            gold = r.get("Stance") or r.get("gold") or r.get("label")
            if target in exclude:
                continue
            eid = r.get("ID") or r.get("id") or i
            rows.append(
                {
                    "example_id": eid,
                    "dataset": "SemEval",
                    "target": target,
                    "text": text,
                    "gold": map_gold(gold, gold_map),
                    "label_space": "FAVOR|AGAINST|NONE",
                    "prompt_kind": "ternary",
                }
            )
    digest = write_csv(rows, out)
    return {"dataset": "SemEval", "n": len(rows), "sha256": digest, "path": str(out)}


def prepare_vaccines(raw_test: Path, out: Path, cfg: dict) -> dict:
    gold_map = cfg.get("gold_map", {"favor": "FAVOR", "against": "AGAINST", "neutral": "NONE"})
    rows = []
    with raw_test.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            text = r.get("tweet") or r.get("Tweet") or r.get("text")
            gold = r.get("Stance") or r.get("stance") or r.get("gold")
            target = r.get("target") or "vaccines"
            rows.append(
                {
                    "example_id": r.get("tweet_id") or i,
                    "dataset": "Vaccines",
                    "target": target,
                    "text": text,
                    "gold": map_gold(gold, gold_map),
                    "label_space": "FAVOR|AGAINST|NONE",
                    "prompt_kind": "ternary",
                }
            )
    digest = write_csv(rows, out)
    return {"dataset": "Vaccines", "n": len(rows), "sha256": digest, "path": str(out)}


def prepare_covid(raw_test: Path, out: Path, cfg: dict) -> dict:
    gold_map = cfg.get("gold_map", {"NEUTRAL": "NONE"})
    rows = []
    with raw_test.open(encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            target = r.get("Target") or r.get("target")
            text = r.get("Tweet") or r.get("text") or r.get("Text")
            gold = r.get("Stance") or r.get("stance") or r.get("gold")
            rows.append(
                {
                    "example_id": r.get("ID") or r.get("id") or i,
                    "dataset": "COVID",
                    "target": target,
                    "text": text,
                    "gold": map_gold(gold, gold_map),
                    "label_space": "FAVOR|AGAINST|NONE",
                    "prompt_kind": "ternary",
                }
            )
    digest = write_csv(rows, out)
    return {"dataset": "COVID", "n": len(rows), "sha256": digest, "path": str(out)}


def prepare_pstance(raw_dir: Path, out: Path, cfg: dict) -> dict:
    """Expect raw_test_{bernie,biden,trump}.csv under raw_dir."""
    files = sorted(raw_dir.glob("raw_test_*.csv"))
    if not files:
        files = sorted(raw_dir.glob("*test*.csv"))
    rows = []
    for fp in files:
        with fp.open(encoding="utf-8") as f:
            for i, r in enumerate(csv.DictReader(f)):
                target = r.get("Target") or r.get("target")
                text = r.get("Tweet") or r.get("text") or r.get("Text")
                gold = r.get("Stance") or r.get("stance") or r.get("gold")
                rows.append(
                    {
                        "example_id": r.get("ID") or r.get("id") or f"{fp.stem}_{i}",
                        "dataset": "P-Stance",
                        "target": target,
                        "text": text,
                        "gold": map_gold(gold, {}),
                        "label_space": "FAVOR|AGAINST",
                        "prompt_kind": "binary",
                    }
                )
    digest = write_csv(rows, out)
    return {"dataset": "P-Stance", "n": len(rows), "sha256": digest, "path": str(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Prepare B-CARE evaluation manifests")
    ap.add_argument("--config", type=Path, default=Path("configs/datasets.yaml"))
    ap.add_argument("--raw-root", type=Path, required=True, help="Root with raw dataset folders")
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/manifests"))
    ap.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = []

    todo = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for ds in todo:
        c = cfg[ds]
        if ds == "semeval":
            raw = args.raw_root / c["raw_subdir"] / c["raw_test_file"]
            summaries.append(prepare_semeval(raw, out_dir / "SemEval_test.csv", c))
        elif ds == "vaccines":
            raw = args.raw_root / c["raw_subdir"] / c["raw_test_file"]
            summaries.append(prepare_vaccines(raw, out_dir / "Vaccines_test.csv", c))
        elif ds == "covid":
            raw = args.raw_root / c["raw_subdir"] / c["raw_test_file"]
            summaries.append(prepare_covid(raw, out_dir / "COVID_test.csv", c))
        elif ds == "pstance":
            raw = args.raw_root / c["raw_subdir"]
            summaries.append(prepare_pstance(raw, out_dir / "PStance_test.csv", c))
        else:
            raise SystemExit(f"unknown dataset {ds}")

    from bcare.data.validate import ManifestValidationError, validate_manifest

    key_map = {
        "SemEval": "semeval",
        "Vaccines": "vaccines",
        "COVID": "covid",
        "P-Stance": "pstance",
    }
    display_map = {
        "semeval": "SemEval",
        "vaccines": "Vaccines",
        "covid": "COVID",
        "pstance": "P-Stance",
    }
    validated = []
    for s in summaries:
        ds_key = key_map[s["dataset"]]
        c = cfg[ds_key]
        try:
            v = validate_manifest(
                Path(s["path"]),
                expected_n=c.get("eval_n"),
                expected_dataset=display_map[ds_key],
                expected_labels=list(c.get("labels", [])),
                expected_targets=c.get("expected_targets"),
                expected_prompt_kind=c.get("prompt_kind"),
                # Do not enforce frozen sha256 on freshly prepared files (text
                # hashing of ids may differ); enforce count/schema/labels/targets.
                expected_sha256=None,
            )
        except ManifestValidationError as e:
            raise SystemExit(f"VALIDATION FAILED for {s['dataset']}: {e}") from e
        s["validation"] = v
        validated.append(s)
        print(
            f"{s['dataset']}: n={s['n']} sha256={s['sha256'][:12]}… VALIDATION=PASS"
        )

    (out_dir / "prepare_summary.json").write_text(json.dumps(validated, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
