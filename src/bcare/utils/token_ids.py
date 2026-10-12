"""HF tokenizer return-type normalization (runtime compatibility only)."""
from __future__ import annotations

from typing import Any


def as_id_list(ids: Any) -> list[int]:
    """Normalize HF BatchEncoding / Mapping / tensor / nested list to list[int].

    Runtime compatibility only: extract integer token IDs. Does not alter prompts.
    """
    if ids is None:
        raise ValueError("as_id_list: got None")

    # BatchEncoding / Mapping with input_ids (do NOT iterate dict keys)
    if hasattr(ids, "input_ids") and not isinstance(ids, (list, tuple)):
        try:
            ids = ids["input_ids"]
        except Exception:
            ids = getattr(ids, "input_ids")
    if isinstance(ids, dict):
        if "input_ids" not in ids:
            raise ValueError(f"as_id_list: dict missing input_ids, keys={list(ids.keys())}")
        ids = ids["input_ids"]

    if hasattr(ids, "detach"):
        ids = ids.detach().cpu()
    if hasattr(ids, "tolist"):
        ids = ids.tolist()

    # Nested batch: allow exactly one sequence
    if isinstance(ids, list) and ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError(
                f"as_id_list: refused to flatten multi-example batch of size {len(ids)}"
            )
        ids = list(ids[0])

    if not isinstance(ids, (list, tuple)):
        raise TypeError(f"as_id_list: unsupported type {type(ids)}")

    out: list[int] = []
    for x in ids:
        if isinstance(x, (list, tuple)):
            raise ValueError(
                "as_id_list: still nested after unwrap; refusing silent flatten"
            )
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise TypeError(f"as_id_list: non-integer token id {x!r} type={type(x)}")
        out.append(int(x))
    if not out:
        raise ValueError("as_id_list: empty token id list")
    return out
