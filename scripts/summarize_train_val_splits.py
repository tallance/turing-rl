"""Aggregate a GRPO run's reward_dump into per-step TRAIN and VAL judge stats.

The reward dump records every judged pair the GRPO run produced, tagged
``split`` = train | val, but it carries **no step field** -- only a wall-clock
``ts``. Steps are recovered from the run's own structure:

  * VAL is scored in discrete passes (val_before_train, then every test_freq
    steps), so val rows gap-split cleanly into passes. Pass 0 is step 0 (before
    any optimisation); pass k is step k*steps_per_block.
  * TRAIN rollouts fall strictly between consecutive val passes, so the val
    pass boundaries partition them into blocks of steps_per_block steps each.

That partition is *verified*, not assumed: the script fails if train rows leak
outside the val-pass boundaries or if the blocks come out unequal. On the
9b_half_kl1e4_lr1e4_temp1 run this yields 5 val passes of 352 and 4 train
blocks of exactly 2048, with 0 rows before the first pass or after the last.

Train is reported per BLOCK, not per step. Within a block the rows are only
approximately step-ordered -- judge latency runs to tens of seconds against a
step time of ~640s, so rows bleed across step boundaries. The block boundaries
themselves are exact (a val pass creates a hard multi-thousand-second gap), so
block-level aggregation is the finest split the timestamps actually support.
Each block is labelled with its END step, which is the checkpoint that block of
training produced.

Usage (cluster; env turing-rl-train):
  python summarize_train_val_splits.py \
      --dump results/grpo/rl-generator/9b_half_kl1e4_lr1e4_temp1/reward_dump \
      --out_csv train_val_splits.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys


def load(dump_dir: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(dump_dir, "reward-*.jsonl"))):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def likerts(rows: list[dict]) -> list[int]:
    """Valid integer Likerts 1..7; parse-fails and 0 dropped.

    Matches the convention in summarize_test_eval.py so train/val numbers are
    computed the same way as the test numbers they get plotted against.
    """
    out = []
    for r in rows:
        v = r.get("turing_judge_score_raw")
        if v is None or int(round(float(v))) == 0:
            continue
        out.append(int(round(float(v))))
    return out


def stats(rows: list[dict]) -> dict:
    lk = likerts(rows)
    if not lk:
        raise SystemExit("FAIL: a bucket had no valid Likert ratings")
    return {
        "n_rows": len(rows),
        "n_likert": len(lk),
        "likert_mean": round(statistics.mean(lk), 4),
        "win_rate_ge5": round(sum(1 for v in lk if v >= 5) / len(lk), 4),
        "pct_7": round(100 * sum(1 for v in lk if v == 7) / len(lk), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--val_gap", type=float, default=400.0,
                    help="seconds; a larger inter-row gap starts a new val pass")
    ap.add_argument("--steps_per_block", type=int, default=8,
                    help="test_freq: GRPO steps between val passes")
    a = ap.parse_args()

    rows = load(a.dump)
    if not rows:
        raise SystemExit(f"FAIL: no reward-*.jsonl rows under {a.dump}")

    judges = {r.get("judge_model") for r in rows}
    if len(judges) != 1:
        raise SystemExit(f"FAIL: dump mixes judge models {judges}; stats would not be comparable")
    judge = judges.pop()

    val = sorted((r for r in rows if r.get("split") == "val"), key=lambda r: r["ts"])
    train = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["ts"])
    if not val or not train:
        raise SystemExit(f"FAIL: need both splits, got val={len(val)} train={len(train)}")

    # --- recover val passes by gap-splitting ---
    passes, cur = [], [val[0]]
    for prev, nxt in zip(val, val[1:]):
        if (nxt["ts"] - prev["ts"]) > a.val_gap:
            passes.append(cur)
            cur = []
        cur.append(nxt)
    passes.append(cur)

    sizes = {len(p) for p in passes}
    if len(sizes) != 1:
        raise SystemExit(f"FAIL: val passes have unequal sizes {[len(p) for p in passes]}; "
                         f"--val_gap={a.val_gap} is probably splitting in the wrong place")

    bounds = [p[0]["ts"] for p in passes]

    # --- partition train by those boundaries, and VERIFY the partition ---
    before = [r for r in train if r["ts"] < bounds[0]]
    after = [r for r in train if r["ts"] >= bounds[-1]]
    if before or after:
        raise SystemExit(f"FAIL: {len(before)} train rows precede the first val pass and "
                         f"{len(after)} follow the last; the val passes do not bracket training, "
                         f"so blocks cannot be mapped to steps")
    blocks = [[r for r in train if bounds[i] <= r["ts"] < bounds[i + 1]]
              for i in range(len(bounds) - 1)]
    bsizes = {len(b) for b in blocks}
    if len(bsizes) != 1:
        raise SystemExit(f"FAIL: train blocks are unequal {[len(b) for b in blocks]}; "
                         f"the step mapping would be wrong")

    out = []
    for i, p in enumerate(passes):
        out.append({"split": "val", "step": i * a.steps_per_block, **stats(p)})
    for i, b in enumerate(blocks):
        # Label the block with the step it ENDS at -- the checkpoint it produced.
        out.append({"split": "train", "step": (i + 1) * a.steps_per_block, **stats(b)})
    out.sort(key=lambda d: (d["split"], d["step"]))

    cols = ["split", "step", "n_rows", "n_likert", "likert_mean", "win_rate_ge5", "pct_7"]
    with open(a.out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out)

    print(f"judge: {judge}", file=sys.stderr)
    print(f"val: {len(passes)} passes of {sizes.copy().pop()}  |  "
          f"train: {len(blocks)} blocks of {bsizes.copy().pop()}", file=sys.stderr)
    print(f"wrote {a.out_csv}", file=sys.stderr)
    print(",".join(cols))
    for r in out:
        print(",".join(str(r[c]) for c in cols))


if __name__ == "__main__":
    main()
