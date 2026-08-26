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

from shared.judge_prompts import TURING_SINGLE_TOKEN_PROMPT  # noqa: E402

PROMPT_STYLES = ("full", "single_token")

# Everything the single-token template emits after its last placeholder: the watchlist
# closer, the "## Output Format" heading and the single-letter instruction. Derived from
# the template rather than spelled out so it cannot drift away from what is rendered.
SINGLE_TOKEN_PROMPT_SUFFIX = TURING_SINGLE_TOKEN_PROMPT.rsplit("}", 1)[1]


def meta_path_for(pairs_path: str | Path) -> Path:
    """The sidecar build_judge_train_pairs.py writes next to its parquet."""
    return Path(pairs_path).with_suffix(".meta.json")


def recorded_prompt_style(pairs_path: str | Path) -> str | None:
    """prompt_style from the sibling .meta.json, or None if there is no sidecar.

    A sidecar that exists but predates the prompt_style field means "full", matching the
    backfill in merge_judge_comparison.py.
    """
    meta = meta_path_for(pairs_path)
    if not meta.exists():
        return None
    return json.loads(meta.read_text()).get("prompt_style", "full")


def check_prompt_style(records: list[dict], pairs_path: str | Path, *, expected: str) -> None:
    """Refuse to build a CE dataset from pairs rendered in the wrong prompt style.

    Nothing downstream catches this. A full-schema parquet yields 20k-char rubric prompts
    that train fine and produce a sane-looking val curve, and the resulting discriminator
    is then served against ~900-char single-token prompts. The bad number reads as "the
    single-token protocol does not work" -- corrupting the conclusion in exactly the
    direction the experiment is testing.

    Two independent checks, because either one alone can be defeated: the sidecar records
    the style but can go missing, and the prompt text is always present but only tells us
    about the single-token style.
    """
    if expected not in PROMPT_STYLES:
        raise ValueError(f"--expect-prompt-style must be one of {list(PROMPT_STYLES)}, "
                         f"got {expected!r}")

    recorded = recorded_prompt_style(pairs_path)
    if recorded is not None and recorded != expected:
        raise ValueError(
            f"prompt style mismatch: {meta_path_for(pairs_path)} records "
            f"prompt_style={recorded!r}, but --expect-prompt-style is {expected!r}. "
            f"Rebuild the pairs with --prompt-style {expected}, or pass "
            f"--expect-prompt-style {recorded} if {recorded!r} is what you meant."
        )
    if recorded is None:
        print(f"WARNING: no {meta_path_for(pairs_path)}; cannot confirm prompt_style "
              f"from provenance, falling back to the prompt-text check.")

    if expected != "single_token":
        return
    for i, rec in enumerate(records):
        prompt = rec["messages"][0]["content"]
        if not prompt.endswith(SINGLE_TOKEN_PROMPT_SUFFIX):
            raise ValueError(
                f"--expect-prompt-style single_token, but record {i} "
                f"(pair_id={rec.get('pair_id')!r}) does not end with the single-letter "
                f"instruction. Expected suffix {SINGLE_TOKEN_PROMPT_SUFFIX!r}; "
                f"prompt ends {prompt[-len(SINGLE_TOKEN_PROMPT_SUFFIX):]!r}."
            )


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
    ap.add_argument(
        "--expect-prompt-style", choices=list(PROMPT_STYLES), default="single_token",
        help="Prompt style the --pairs parquet must have been built with. Checked against "
             "the sibling .meta.json and, for single_token, against the rendered prompt "
             "text. Not a conversion: mismatches fail rather than re-render.",
    )
    args = ap.parse_args()

    records = build_ce_records(pd.read_parquet(args.pairs))
    check_prompt_style(records, args.pairs, expected=args.expect_prompt_style)

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
