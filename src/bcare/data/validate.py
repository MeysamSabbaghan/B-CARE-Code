"""Strict validation of prepared B-CARE manifests."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


REQUIRED_COLS = [
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


class ManifestValidationError(ValueError):
    pass


def validate_manifest(
    path: Path,
    *,
    expected_n: int | None = None,
    expected_dataset: str | None = None,
    expected_labels: list[str] | None = None,
    expected_targets: list[str] | None = None,
    expected_prompt_kind: str | None = None,
    expected_sha256: str | None = None,
    require_unique_ids: bool = True,
    allow_empty_text: bool = False,
    require_hex_sha256: bool = True,
) -> dict:
    """Validate a prepared TEST manifest. Raises ManifestValidationError on failure."""
    if not path.exists():
        raise ManifestValidationError(f"manifest not found: {path}")
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ManifestValidationError("manifest has no header")
        missing = [c for c in REQUIRED_COLS if c not in reader.fieldnames]
        if missing:
            raise ManifestValidationError(f"missing required columns: {missing}")
        rows = list(reader)

    if expected_n is not None and len(rows) != int(expected_n):
        raise ManifestValidationError(
            f"expected n={expected_n}, got n={len(rows)} in {path}"
        )
    if not rows:
        raise ManifestValidationError(f"empty manifest: {path}")

    ids = [str(r["example_id"]) for r in rows]
    if require_unique_ids and len(ids) != len(set(ids)):
        dup = [i for i, c in Counter(ids).items() if c > 1][:5]
        raise ManifestValidationError(f"duplicate example_id values (e.g. {dup})")

    datasets = {r["dataset"] for r in rows}
    if expected_dataset and datasets != {expected_dataset}:
        raise ManifestValidationError(
            f"expected dataset={expected_dataset!r}, found {datasets}"
        )

    found_targets = {str(r["target"]).strip() for r in rows}
    if expected_targets is not None:
        exp = set(expected_targets)
        if found_targets != exp:
            raise ManifestValidationError(
                f"target set mismatch: expected {sorted(exp)}, found {sorted(found_targets)}"
            )

    for i, r in enumerate(rows):
        for c in REQUIRED_COLS:
            if r.get(c) is None or str(r.get(c)).strip() == "":
                if c == "text" and allow_empty_text:
                    continue
                raise ManifestValidationError(f"row {i}: empty/missing field {c}")
        labels = tuple(str(r["label_space"]).split("|"))
        if expected_labels and list(labels) != list(expected_labels):
            raise ManifestValidationError(
                f"row {i}: label_space {labels} != expected {expected_labels}"
            )
        gold = str(r["gold"]).strip()
        if gold not in labels:
            raise ManifestValidationError(
                f"row {i}: gold {gold!r} not in label_space {labels}"
            )
        pk = str(r["prompt_kind"]).strip()
        if pk not in ("ternary", "binary"):
            raise ManifestValidationError(f"row {i}: invalid prompt_kind {pk!r}")
        if expected_prompt_kind and pk != expected_prompt_kind:
            raise ManifestValidationError(
                f"row {i}: prompt_kind {pk!r} != expected {expected_prompt_kind!r}"
            )
        if pk == "binary" and set(labels) != {"FAVOR", "AGAINST"}:
            raise ManifestValidationError(
                f"row {i}: binary prompt_kind requires FAVOR|AGAINST"
            )
        if pk == "ternary" and set(labels) != {"FAVOR", "AGAINST", "NONE"}:
            raise ManifestValidationError(
                f"row {i}: ternary prompt_kind requires FAVOR|AGAINST|NONE"
            )
        sha = str(r["text_sha256"]).strip().lower()
        if len(sha) != 64:
            raise ManifestValidationError(f"row {i}: text_sha256 must be 64 hex chars")
        if require_hex_sha256 and any(ch not in "0123456789abcdef" for ch in sha):
            raise ManifestValidationError(f"row {i}: text_sha256 must be hex")

    if expected_sha256:
        from bcare.data.prepare import sha256_file

        got = sha256_file(path)
        if got != expected_sha256:
            raise ManifestValidationError(
                f"manifest sha256 mismatch: got {got}, expected {expected_sha256}"
            )

    return {
        "path": str(path),
        "n": len(rows),
        "dataset": next(iter(datasets)),
        "n_targets": len(found_targets),
        "targets": sorted(found_targets),
        "gold_counts": dict(Counter(r["gold"] for r in rows)),
        "ok": True,
    }
