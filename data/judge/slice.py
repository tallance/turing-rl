"""Deterministic hash slicing for judge-training data.

Membership is a pure function of ``(user_id, post_id, target_idx)``: no seed, no
ordering dependence, and no dependence on how many rows precede a given row. Future
alternating iterations can therefore draw disjoint slices from the same pool without
storing or recomputing anything.

The triple is hashed rather than ``user_id`` alone because users hold varying row
counts, so a 10% user slice would be lumpy. The corpus is already partitioned by user
one level up, in ``data/prism/split_data.py``.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


def slice_key(user_id: Any, post_id: Any, target_idx: Any) -> str:
    """Canonical identity string for one target turn."""
    return f"{user_id}::{post_id}::{target_idx}"


def slice_fraction(user_id: Any, post_id: Any, target_idx: Any) -> float:
    """Map a row identity to a stable ``u`` in ``[0.0, 1.0)``."""
    digest = hashlib.blake2b(
        slice_key(user_id, post_id, target_idx).encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "big") / 2**64


def _validate_bounds(lo: float, hi: float) -> None:
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"slice bounds must satisfy 0 <= lo < hi <= 1, got lo={lo} hi={hi}")


def in_slice(user_id: Any, post_id: Any, target_idx: Any, *, lo: float, hi: float) -> bool:
    """True when this row's hash falls in the half-open interval ``[lo, hi)``."""
    _validate_bounds(lo, hi)
    return lo <= slice_fraction(user_id, post_id, target_idx) < hi


def select_slice(
    df: pd.DataFrame, *, lo: float, hi: float, limit: int | None = None
) -> pd.DataFrame:
    """Return rows whose ``extra_info`` identity hashes into ``[lo, hi)``.

    Rows come back in ascending hash order, so ``limit`` truncates deterministically
    instead of depending on the input row order.
    """
    _validate_bounds(lo, hi)
    if df.empty:
        return df.copy()

    fractions: list[float] = []
    for extra in df["extra_info"]:
        if not isinstance(extra, dict):
            raise TypeError(f"extra_info must be a dict, got {type(extra)!r}")
        fractions.append(
            slice_fraction(extra.get("user_id"), extra.get("post_id"), extra.get("target_idx"))
        )

    out = df.assign(_slice_u=fractions)
    out = out.loc[(out["_slice_u"] >= lo) & (out["_slice_u"] < hi)]
    out = out.sort_values("_slice_u", kind="stable")
    if limit is not None:
        out = out.head(limit)
    return out.drop(columns=["_slice_u"]).reset_index(drop=True)
