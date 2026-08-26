"""Pre-run gates for the single-token judge comparison.

Both guard against a result that looks fine and is wrong: training on the users you
evaluate on, and evaluating on a pair set other than the one the reused cells used.

  * ``check_no_user_overlap`` -- the CE training pairs (built by
    ``scripts/build_judge_train_pairs.py``) must share zero ``user_id``s with the
    evaluation pairs. Any overlap means the trained cell's reported accuracy is inflated
    by memorised users, not held-out generalisation.
  * ``check_sha256`` -- the evaluation parquet actually scored must match the exact file
    whose checksum is recorded in the design, so every reused number in the final table
    was computed against the same 880 pairs.

Run before any GPU work:

    python scripts/judge_ce_guards.py \\
        --train-pairs <ce_train_pairs.parquet> \\
        --eval-pairs <eval_pairs.parquet> \\
        --expected-sha256 <sha256 recorded in the design doc> \\
        --out <run_root>/split_guard.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


class LeakageError(Exception):
    """CE training users overlap the eval users."""


def check_no_user_overlap(train_users: set, eval_users: set) -> dict:
    overlap = sorted(str(u) for u in (set(train_users) & set(eval_users)))
    if overlap:
        raise LeakageError(
            f"{len(overlap)} user(s) appear in both CE training and eval: "
            f"{overlap[:10]}{'...' if len(overlap) > 10 else ''}"
        )
    return {
        "n_train_users": len(train_users),
        "n_eval_users": len(eval_users),
        "overlap": overlap,
    }


def check_sha256(path: Path, expected: str) -> str:
    actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: {actual} != {expected}")
    return actual


def write_split_guard(out: Path, payload: dict) -> None:
    Path(out).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def load_user_ids(parquet: Path) -> set:
    """Pull the ``user_id`` set out of a judge-pairs parquet's ``extra_info`` column."""
    df = pd.read_parquet(parquet, columns=["extra_info"])
    return {str(extra["user_id"]) for extra in df["extra_info"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Leakage and pair-set identity gates for the judge CE comparison"
    )
    parser.add_argument("--train-pairs", required=True, type=Path)
    parser.add_argument("--eval-pairs", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        train_users = load_user_ids(args.train_pairs)
        eval_users = load_user_ids(args.eval_pairs)
        overlap_report = check_no_user_overlap(train_users, eval_users)
        eval_sha256 = check_sha256(args.eval_pairs, args.expected_sha256)
    except (LeakageError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    payload = {
        "status": "pass",
        "train_pairs": str(Path(args.train_pairs).resolve()),
        "eval_pairs": str(Path(args.eval_pairs).resolve()),
        "eval_sha256": eval_sha256,
        **overlap_report,
    }
    write_split_guard(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
