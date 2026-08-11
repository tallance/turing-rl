#!/usr/bin/env python3
"""Check that a GRPO run repeats the SAME samples every epoch, from its reward dump.

Complements scripts/build_eval_subsets.py. That script proves *which* prompts a run used
(the key SET matches the replayed seeded draw). This one proves *how often* each was used:
whether every sample recurs the same number of times, i.e. once per epoch with nothing
rotating in or out.

Why it matters for MODE=frac10ep10: the train loader sets drop_last=True
(verl/trainer/ppo/ray_trainer.py:409). If the subset size is not a multiple of
data.train_batch_size, each epoch silently drops the remainder -- and because the sampler
reshuffles per epoch, a DIFFERENT remainder each time. The key set still looks correct, so
build_eval_subsets.py still passes; only the per-key counts reveal the rotation. 384 = 6 x 64
divides exactly, so every count should be identical.

Safe to run mid-run: the dump is append-only and a partial trailing line is skipped. Run it
at an epoch boundary -- mid-epoch the counts are legitimately uneven because the epoch is
only part-way through.

Usage:
  verify_run_repetition.py --reward_dump <dir> [--expect-train 384] [--expect-val 352]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path


def counts_by_key(dump_dir: Path, split: str) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    files = sorted(glob.glob(str(dump_dir / "reward-*.jsonl")))
    if not files:
        raise SystemExit(f"FAIL: no reward-*.jsonl under {dump_dir}")
    for path in files:
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a partially-written trailing line on a live dump
                if row.get("split") != split:
                    continue
                counter[(str(row["user_id"]), str(row["post_id"]), str(row["target_idx"]))] += 1
    return counter


def report(split: str, counter: collections.Counter, expect_unique: int | None) -> bool:
    if not counter:
        print(f"[{split}] no rows yet")
        return True
    hist = collections.Counter(counter.values())
    uniq = len(counter)
    total = sum(counter.values())
    print(f"[{split}] {uniq} unique keys, {total} judged calls")
    for times, n in sorted(hist.items()):
        print(f"        {n} keys seen {times}x")

    ok = True
    if expect_unique is not None and uniq != expect_unique:
        print(f"        FAIL: expected {expect_unique} unique keys, got {uniq}")
        ok = False
    if len(hist) == 1:
        print(f"        OK: every key seen exactly {next(iter(hist))}x -- no rotation")
    else:
        # Uneven counts at an epoch boundary mean the sample set is not stable across epochs.
        lo, hi = min(hist), max(hist)
        print(
            f"        UNEVEN: counts range {lo}..{hi}. At an epoch boundary this means the "
            f"per-epoch sample set is rotating (drop_last on a non-multiple subset). "
            f"Mid-epoch, it is expected."
        )
        ok = False
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward_dump", required=True)
    ap.add_argument("--expect-train", type=int, default=None)
    ap.add_argument("--expect-val", type=int, default=None)
    ap.add_argument("--strict", action="store_true", help="exit non-zero on any problem")
    a = ap.parse_args()

    dump = Path(a.reward_dump)
    ok_train = report("train", counts_by_key(dump, "train"), a.expect_train)
    ok_val = report("val", counts_by_key(dump, "val"), a.expect_val)
    good = ok_train and ok_val
    print("\n" + ("PASS: sample set is stable across epochs" if good else "PROBLEM: see above"))
    return 0 if good or not a.strict else 1


if __name__ == "__main__":
    sys.exit(main())
