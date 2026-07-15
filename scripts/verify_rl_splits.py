# scripts/verify_rl_splits.py
"""PRISM RL split-integrity / no-leakage guard.

Standing guard that the PRISM ``full_s42_history_sft40_grpo60_test10`` splits are
leakage-free (disjoint user sets across SFT / GRPO / test) and match the paper's
Table 3 counts. Runs both as a pytest (``tests/test_split_integrity.py``, which
SKIPs when the parquets are absent, i.e. locally) and as a CLI preflight gate to
be run on the cluster before any train/eval.

Split layout (produced by ``data/prism/split_data.py``)
-------------------------------------------------------
Given ``--base data/prism/full_s42_history_sft40_grpo60_test10`` the parquets are:

    <base>/grpo/train.parquet   # GRPO training rows
    <base>/grpo/val.parquet     # GRPO validation rows
    <base>/test.parquet         # held-out test rows (disjoint users)
    <base>/sft/train.parquet    # SFT rows (ASSUMED path, see below)

**ASSUMED SFT split path:** ``<base>/sft/train.parquet``.
This is confirmed against ``data/prism/split_data.py``, whose ``_write_sft_dir``
writes the SFT rows to ``<output_dir>/sft/train.parquet``. There is NO local PRISM
data on the Mac working copy, so this cannot be verified locally; the loader tries
``<base>/sft/train.parquet`` first and falls back gracefully (reporting the SFT
overlaps as ``None`` = "not checked") if that file is absent. If the cluster run
finds the SFT rows live elsewhere in this lineage, pass ``--sft <path>`` to point
at the real file and correct this docstring.

user_id extraction
------------------
Each split row carries an ``extra_info`` field. ``split_data.py`` stores it as a
dict with a ``user_id`` key, but parquet round-tripping / different producers can
surface it as a dict, a JSON string, or a numpy/pandas-wrapped object. ``_user_id``
below is a robust extractor: it JSON-parses strings, unwraps numpy scalars/arrays
and objects exposing ``item()``/attributes, then reads ``user_id`` (falling back to
``user`` / ``uid`` only if ``user_id`` is absent).

Return shape (``check_splits``)
-------------------------------
    {
      "counts": {"grpo/train": (n_rows, n_users),
                 "grpo/val":   (n_rows, n_users),
                 "test":       (n_rows, n_users)},
      "overlaps": {"train_test": int, "val_test": int,
                   "sft_grpo": int|None, "sft_test": int|None},
    }

``sft_grpo`` / ``sft_test`` are ``None`` when the SFT split file was not found
(warn, don't hard-fail); a present-but-overlapping SFT is a hard failure.

CLI (preflight gate)
--------------------
    python scripts/verify_rl_splits.py --base <base> [--sft <path>] [--overfit <path>]

Prints a JSON report and exits non-zero if any test overlap > 0, any (present)
SFT overlap > 0, or any count mismatch vs the expected Table 3 counts.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Optional

import pandas as pd

# Expected paper Table 3 counts: split name -> (n_rows, n_users).
EXPECTED_COUNTS: dict[str, tuple[int, int]] = {
    "grpo/train": (4174, 696),
    "grpo/val": (705, 696),
    "test": (880, 128),
}


def _coerce_to_obj(extra: Any) -> Any:
    """Best-effort unwrap of an ``extra_info`` cell into a plain dict.

    Handles: dict (returned as-is), JSON string (parsed), bytes (decoded then
    parsed), numpy/pandas scalar or 0-d array wrappers (``.item()``), and objects
    that expose ``user_id`` etc. as attributes (returned unchanged for attribute
    access downstream).
    """
    # Unwrap numpy scalars / 0-d arrays / anything with a scalar .item().
    if hasattr(extra, "item") and not isinstance(extra, (dict, str, bytes)):
        try:
            extra = extra.item()
        except (ValueError, TypeError):
            pass
    if isinstance(extra, bytes):
        try:
            extra = extra.decode("utf-8")
        except UnicodeDecodeError:
            return extra
    if isinstance(extra, str):
        try:
            return json.loads(extra)
        except (json.JSONDecodeError, ValueError):
            return extra
    return extra


def _user_id(extra: Any) -> str:
    """Extract the per-row user id from an ``extra_info`` cell.

    Robust to dict / JSON-string / numpy-or-pandas-object encodings. Prefers the
    ``user_id`` key; falls back to ``user`` / ``uid`` only if ``user_id`` absent.
    """
    obj = _coerce_to_obj(extra)

    if isinstance(obj, dict):
        for key in ("user_id", "user", "uid"):
            if key in obj and obj[key] is not None:
                return str(obj[key])
        raise KeyError(f"no user_id/user/uid key in extra_info dict: {sorted(obj)!r}")

    # Object with attributes (e.g. a namedtuple/struct-like row).
    for key in ("user_id", "user", "uid"):
        val = getattr(obj, key, None)
        if val is not None:
            return str(val)

    raise TypeError(
        f"cannot extract user_id from extra_info of type {type(extra)!r}: {extra!r}"
    )


def _user_ids(df: pd.DataFrame) -> set[str]:
    if "extra_info" not in df.columns:
        raise KeyError(f"parquet has no 'extra_info' column; columns={list(df.columns)}")
    return {_user_id(v) for v in df["extra_info"].tolist()}


def _rows_and_users(path: str) -> tuple[int, set[str]]:
    df = pd.read_parquet(path)
    return int(len(df)), _user_ids(df)


def _resolve_sft_path(base: str, sft_path: Optional[str]) -> Optional[str]:
    """Return the SFT train parquet path, or None if not found.

    Explicit ``sft_path`` wins; otherwise try the assumed ``<base>/sft/train.parquet``.
    """
    if sft_path:
        return sft_path if os.path.exists(sft_path) else None
    candidate = os.path.join(base, "sft", "train.parquet")
    return candidate if os.path.exists(candidate) else None


def check_splits(base: str, sft_path: Optional[str] = None) -> dict[str, Any]:
    """Compute per-split (rows, distinct-users) counts and cross-split user overlaps.

    See the module docstring for the SFT path assumption and return shape.
    ``sft_grpo`` / ``sft_test`` are ``None`` when the SFT split is not found.
    """
    gt_rows, gt_users = _rows_and_users(os.path.join(base, "grpo", "train.parquet"))
    gv_rows, gv_users = _rows_and_users(os.path.join(base, "grpo", "val.parquet"))
    te_rows, te_users = _rows_and_users(os.path.join(base, "test.parquet"))

    resolved_sft = _resolve_sft_path(base, sft_path)
    if resolved_sft is not None:
        _, sft_users = _rows_and_users(resolved_sft)
        sft_grpo: Optional[int] = len(sft_users & (gt_users | gv_users))
        sft_test: Optional[int] = len(sft_users & te_users)
    else:
        sft_grpo = None
        sft_test = None

    return {
        "counts": {
            "grpo/train": (gt_rows, len(gt_users)),
            "grpo/val": (gv_rows, len(gv_users)),
            "test": (te_rows, len(te_users)),
        },
        "overlaps": {
            "train_test": len(gt_users & te_users),
            "val_test": len(gv_users & te_users),
            "sft_grpo": sft_grpo,
            "sft_test": sft_test,
        },
        "sft_path": resolved_sft,
    }


def check_overfit(base: str, overfit_path: str) -> dict[str, Any]:
    """Sanity-check an overfit subset against the GRPO train split and test set.

    Returns ``{"subset_of_train": bool, "overfit_test_overlap": int}``:
      * ``subset_of_train`` — every overfit row's user id is in grpo/train users
        (user membership; the overfit subset is built from grpo train rows).
      * ``overfit_test_overlap`` — number of overfit users that also appear in test
        (must be 0).
    """
    _, gt_users = _rows_and_users(os.path.join(base, "grpo", "train.parquet"))
    _, te_users = _rows_and_users(os.path.join(base, "test.parquet"))
    _, of_users = _rows_and_users(overfit_path)
    return {
        "subset_of_train": of_users.issubset(gt_users),
        "overfit_test_overlap": len(of_users & te_users),
    }


def _evaluate(report: dict[str, Any], overfit: Optional[dict[str, Any]]) -> tuple[bool, list[str]]:
    """Return (ok, problems). ``ok`` is False if any hard-fail condition is hit."""
    problems: list[str] = []
    counts = report["counts"]
    overlaps = report["overlaps"]

    # Count mismatch vs expected Table 3 counts.
    for name, expected in EXPECTED_COUNTS.items():
        actual = tuple(counts.get(name, ()))
        if actual != expected:
            problems.append(f"count mismatch {name}: expected {expected}, got {actual}")

    # Any test overlap must be 0.
    for key in ("train_test", "val_test"):
        if overlaps.get(key):
            problems.append(f"{key} overlap = {overlaps[key]} (must be 0)")

    # SFT overlaps: None = not checked (warn only); present-but-nonzero = hard fail.
    for key in ("sft_grpo", "sft_test"):
        val = overlaps.get(key)
        if val is None:
            problems.append(f"WARN: {key} not checked (SFT split not found)")
        elif val:
            problems.append(f"{key} overlap = {val} (must be 0)")

    if overfit is not None:
        if not overfit["subset_of_train"]:
            problems.append("overfit rows are NOT a subset of grpo/train users")
        if overfit["overfit_test_overlap"]:
            problems.append(
                f"overfit users overlap test = {overfit['overfit_test_overlap']} (must be 0)"
            )

    hard_fail = any(not p.startswith("WARN:") for p in problems)
    return (not hard_fail), problems


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PRISM RL split-integrity / no-leakage preflight gate.")
    ap.add_argument("--base", required=True, help="Split base dir (contains grpo/, sft/, test.parquet).")
    ap.add_argument("--sft", default=None, help="Explicit SFT train parquet path (overrides <base>/sft/train.parquet).")
    ap.add_argument("--overfit", default=None, help="Optional overfit subset parquet to sanity-check.")
    a = ap.parse_args(argv)

    report = check_splits(a.base, sft_path=a.sft)
    overfit = check_overfit(a.base, a.overfit) if a.overfit else None
    if overfit is not None:
        report["overfit"] = overfit

    ok, problems = _evaluate(report, overfit)
    report["problems"] = problems
    report["passed"] = ok
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
