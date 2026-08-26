#!/usr/bin/env python3
"""Assert that an eval parquet really is the split the caller claims it is.

WHY THIS EXISTS
---------------
The eval chain (launch_test_eval.sh -> generator_infer.sh -> build_pairs.sh) used to hardcode
``data/prism/.../test.parquet``, so "held-out eval" was true by construction. Once the parquet
became overridable (EVAL_PARQUET), that guarantee disappeared: a run could point at
``grpo/train.parquet``, produce a very nice curve, and be written up as test-set generalisation.
Nothing downstream would notice -- the generator, pair builder, judge and summarizer are all
split-agnostic.

So the split becomes an ASSERTED property of every run, checked before any GPU time is spent and
recorded in the artifacts.

WHY AN EXPECTATION, NOT AN --allow-overlap BYPASS
-------------------------------------------------
A bypass flag only protects the held-out case. The moment a deliberate train-set eval sets it, the
check is off for that run too, and pointing the train arm at the val parquet passes silently. An
expectation flag instead makes every arm carry a positive assertion:

  --expect heldout  (default)  eval users must be DISJOINT from sft/train + grpo/train + grpo/val
  --expect train               eval rows must be a SUBSET of grpo/train
  --expect val                 eval rows must be a SUBSET of grpo/val
  --expect any                 escape hatch; reports what it found and passes loudly

Held-out is checked at USER level (the split was built by user, so a shared user is leakage even
with disjoint rows). train/val are checked at ROW level (user, post_id, target_idx), because both
draw from the same 696 users and only the rows distinguish them.

Usage:
  python scripts/check_eval_split.py --eval_parquet data/prism/<ds>/test.parquet
  python scripts/check_eval_split.py --eval_parquet <...>/grpo/train_used2048.parquet \
      --expect train --out_json <eval_root>/split_guard.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# Reference splits, relative to the dataset root that holds split_metadata.json.
SFT_TRAIN = Path("sft") / "train.parquet"
GRPO_TRAIN = Path("grpo") / "train.parquet"
GRPO_VAL = Path("grpo") / "val.parquet"
HELDOUT = Path("test.parquet")

# Splits whose users must never appear in a held-out eval.
TRAINING_CORPORA = {"sft_train": SFT_TRAIN, "grpo_train": GRPO_TRAIN, "grpo_val": GRPO_VAL}

EXPECT_SUBSET_OF = {"train": ("grpo_train", GRPO_TRAIN), "val": ("grpo_val", GRPO_VAL)}


def find_split_root(eval_parquet: Path) -> Path:
    """Walk up to the nearest dataset root (the dir holding split_metadata.json).

    Deriving the references from the eval parquet's own location means the guard cannot be
    silently pointed at a DIFFERENT dataset build's splits -- which would make it pass while
    comparing against the wrong users entirely.
    """
    for candidate in [eval_parquet.parent, *eval_parquet.parents]:
        if (candidate / "split_metadata.json").is_file():
            return candidate
    raise SystemExit(
        f"FAIL: no split_metadata.json found at or above {eval_parquet}. The guard cannot "
        "identify this parquet's dataset build. Pass --split_root explicitly if the layout differs."
    )


def load_keys(parquet: Path) -> tuple[set[str], set[tuple[str, str, str]]]:
    """Return (user_ids, (user_id, post_id, target_idx) keys) for a GRPO-format parquet."""
    if not parquet.is_file():
        raise SystemExit(f"FAIL: reference split missing: {parquet}")
    # extra_info alone; the prompt columns are large and irrelevant here.
    df = pd.read_parquet(parquet, columns=["extra_info"])
    users: set[str] = set()
    keys: set[tuple[str, str, str]] = set()
    for extra in df["extra_info"]:
        user_id = str(extra["user_id"])
        users.add(user_id)
        keys.add((user_id, str(extra["post_id"]), str(extra["target_idx"])))
    return users, keys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check(eval_parquet: Path, expect: str, split_root: Path) -> tuple[bool, dict[str, Any]]:
    eval_users, eval_keys = load_keys(eval_parquet)
    report: dict[str, Any] = {
        "eval_parquet": str(eval_parquet.resolve()),
        "eval_sha256": sha256_file(eval_parquet),
        "eval_rows": len(eval_keys),
        "eval_users": len(eval_users),
        "expect": expect,
        "split_root": str(split_root.resolve()),
        "overlap": {},
    }

    # Always report every overlap, whatever the expectation -- the numbers are the evidence, and
    # a future reader of split_guard.json should not have to re-run anything to see them.
    for name, rel in {**TRAINING_CORPORA, "heldout_test": HELDOUT}.items():
        path = split_root / rel
        if not path.is_file():
            report["overlap"][name] = {"error": "missing"}
            continue
        ref_users, ref_keys = load_keys(path)
        report["overlap"][name] = {
            "ref_rows": len(ref_keys),
            "ref_users": len(ref_users),
            "shared_users": len(eval_users & ref_users),
            "shared_rows": len(eval_keys & ref_keys),
            "eval_rows_not_in_ref": len(eval_keys - ref_keys),
        }

    problems: list[str] = []
    if expect == "heldout":
        for name in TRAINING_CORPORA:
            shared = report["overlap"][name].get("shared_users")
            if shared is None:
                problems.append(f"{name}: reference split missing, cannot prove held-out")
            elif shared:
                problems.append(
                    f"{name}: {shared} eval users also appear in {name} "
                    f"({report['overlap'][name]['shared_rows']} identical rows). "
                    "This eval set is NOT held out."
                )
    elif expect in EXPECT_SUBSET_OF:
        name, _ = EXPECT_SUBSET_OF[expect]
        strays = report["overlap"][name].get("eval_rows_not_in_ref")
        if strays is None:
            problems.append(f"{name}: reference split missing, cannot prove subset")
        elif strays:
            problems.append(
                f"--expect {expect} requires every eval row to come from {name}, but "
                f"{strays}/{len(eval_keys)} rows are not in it. Wrong parquet for this arm?"
            )
    elif expect != "any":  # argparse restricts this, but fail loudly rather than pass silently
        problems.append(f"unknown expectation {expect!r}")

    report["problems"] = problems
    report["verdict"] = "PASS" if not problems else "FAIL"
    return not problems, report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--eval_parquet", required=True)
    ap.add_argument(
        "--expect",
        default="heldout",
        choices=["heldout", "train", "val", "any"],
        help="What this eval set is claimed to be. Default 'heldout' (zero training overlap).",
    )
    ap.add_argument(
        "--split_root",
        default=None,
        help="Dataset root holding split_metadata.json (default: nearest one above --eval_parquet)",
    )
    ap.add_argument("--out_json", default=None, help="Write the verdict here (e.g. <eval_root>/split_guard.json)")
    a = ap.parse_args()

    eval_parquet = Path(a.eval_parquet)
    if not eval_parquet.is_file():
        raise SystemExit(f"FAIL: --eval_parquet does not exist: {eval_parquet}")
    split_root = Path(a.split_root) if a.split_root else find_split_root(eval_parquet)

    ok, report = check(eval_parquet, a.expect, split_root)

    print(f"[split guard] {eval_parquet}")
    print(f"  expect={a.expect}  rows={report['eval_rows']}  users={report['eval_users']}")
    for name, ov in report["overlap"].items():
        if "error" in ov:
            print(f"  {name:13s}: MISSING")
            continue
        print(f"  {name:13s}: shared_users={ov['shared_users']:5d}  shared_rows={ov['shared_rows']:5d}")

    if a.out_json:
        out = Path(a.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"  verdict -> {out}")

    if a.expect == "any":
        print(
            "\n!! --expect any: the split was NOT verified. Whatever this run produces must not be "
            "described as held-out unless the overlap counts above are all zero.",
            file=sys.stderr,
        )

    if not ok:
        print("\nFAILED: eval data does not match the declared split:", file=sys.stderr)
        for p in report["problems"]:
            print(f"  - {p}", file=sys.stderr)
        print(
            "\nIf this is intentional, say so explicitly with --expect train|val (asserts the eval "
            "set really is that split) rather than disabling the check.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print(f"\nPASS: eval set is consistent with --expect {a.expect}")


if __name__ == "__main__":
    main()
