#!/usr/bin/env python
"""Convert a judge-training pair parquet into the JSONL lora_sft.py consumes.

The judge parquet already holds the rendered prompt and an "A"/"B" label, so this is a
reshape, not a transformation. Keeping it a separate script (rather than a flag on the
pair builder) means the veRL-shaped parquet stays the single source of pairs for both
the GRPO arms and the CE arm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_ce_records(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict("records"):
        label = row["reward_model"]["ground_truth"]
        if label not in ("A", "B"):
            raise ValueError(f"unexpected label {label!r} in {row['extra_info']}")
        records.append({
            "messages": [
                {"role": "user", "content": row["prompt"][0]["content"]},
                {"role": "assistant", "content": label},
            ],
            "pair_id": row["extra_info"]["pair_id"],
            # human_a/human_b is a perfect proxy for the label by construction (see
            # build_judge_train_pairs.py). It is inert for lora_sft.py, which drops every
            # column but "messages" -- but anything reading this JSONL directly must never
            # feed "order" to a model.
            "order": row["extra_info"]["order"],
            "user_id": row["extra_info"]["user_id"],
        })
    return records


def split_by_user(records: list[dict], *, val_frac: float) -> tuple[list[dict], list[dict]]:
    """Hold out whole USERS, not rows. Both orders of a pair, and every pair from one user,
    must land on the same side: splitting by row would put one order in train and the
    other in val, leaking the answer and turning the val score into a memorisation score."""
    if not (0.0 <= val_frac < 1.0):
        raise ValueError(f"--val-frac must be in [0, 1), got {val_frac!r}")
    users = sorted({r["user_id"] for r in records})
    # max(1, ...) is a floor, not an off-switch: even val_frac=0.0 holds out exactly one
    # user, so --val-out always yields a non-empty val set when given (see --val-frac help).
    n_val = max(1, int(len(users) * val_frac))
    val_users = set(users[:n_val])
    train = [r for r in records if r["user_id"] not in val_users]
    val = [r for r in records if r["user_id"] in val_users]
    if not train or not val:
        # Training on nothing, or measuring convergence on nothing, are both unrecoverable
        # and must fail loudly rather than silently write an empty JSONL -- this is exactly
        # what a small --limit smoke run (few users) can hit.
        raise ValueError(
            f"split_by_user produced an empty side: {len(users)} user(s), "
            f"val_frac={val_frac} -> {len(val_users)} val user(s), "
            f"{len(train)} train row(s), {len(val)} val row(s). Need at least 2 users."
        )
    return train, val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-out", default=None,
                    help="If set, hold out whole users into this file for early stopping.")
    ap.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of users held out for val, in [0, 1). NOTE: 0.0 is not \"no split\" "
             "-- a floor always holds out at least one user, so --val-out never writes an "
             "empty file when given.",
    )
    args = ap.parse_args()

    records = build_ce_records(pd.read_parquet(args.pairs))

    if args.val_out:
        train, val = split_by_user(records, val_frac=args.val_frac)
        Path(args.val_out).write_text("".join(json.dumps(r) + "\n" for r in val))
        val_users = {r["user_id"] for r in val}
        print(f"val: {len(val)} rows / {len(val_users)} users -> {args.val_out}")
    else:
        train = records

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in train))
    print(f"train: {len(train)} rows -> {args.out}")


if __name__ == "__main__":
    main()
