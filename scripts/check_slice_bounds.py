"""Assert every turn in a parquet or reward dump hashes into a given slice.

The slice arithmetic itself is already covered by tests/test_judge_slice.py. This checks
the thing a unit test cannot: that a RUN actually used the sliced data it was configured
with. ``TRAIN_FILE`` reaches the trainer through sbatch ``--export=ALL`` and srun
inheritance, and if it were ever dropped the job would silently fall back to the full split
-- training on rows the judge was trained on, while every log, metric and curve still looks
healthy. Only the scored rows reveal that, so this runs against the reward dump too.

Checking ``lo <= u < hi`` subsumes disjointness from an adjacent slice: the judge holds
[0.0, 0.1), so proving every generator turn is in [0.1, 0.2) proves no turn is shared.

    python scripts/check_slice_bounds.py --rows <parquet|reward_dump_dir> --lo 0.1 --hi 0.2
    python scripts/check_slice_bounds.py --rows <reward_dump_dir> --split train --lo 0.1 --hi 0.2
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.judge.slice import slice_fraction, slice_key


def keys_from_parquet(path: str) -> list[str]:
    import pandas as pd

    df = pd.read_parquet(path)
    if "extra_info" not in df.columns:
        raise SystemExit(f"{path} has no extra_info column (cols: {list(df.columns)})")
    out = []
    for extra in df["extra_info"]:
        if not isinstance(extra, dict):
            raise SystemExit(f"extra_info must be a dict, got {type(extra)!r}")
        out.append(slice_key(extra.get("user_id"), extra.get("post_id"), extra.get("target_idx")))
    return out


def keys_from_dump(dump_dir: str, split: str | None) -> list[str]:
    """Turn keys from a reward dump, optionally restricted to one split.

    A dump mixes splits, and only the train rows come from the sliced file -- val is
    deliberately left unsliced. Without --split the check would fail on rows that are
    supposed to be out of range, so the filter is load-bearing, not cosmetic.
    """
    files = sorted(glob.glob(os.path.join(dump_dir, "reward-*.jsonl")))
    if not files:
        raise SystemExit(f"no reward-*.jsonl under {dump_dir}")
    out = []
    for path in files:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a dump being appended to can end mid-line
                if split is not None and row.get("split") != split:
                    continue
                out.append(slice_key(row.get("user_id"), row.get("post_id"), row.get("target_idx")))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", required=True, help="parquet file or reward_dump directory")
    parser.add_argument("--lo", type=float, required=True)
    parser.add_argument("--hi", type=float, required=True)
    parser.add_argument("--split", default=None,
                        help="reward-dump split to keep (e.g. train); ignored for parquet")
    args = parser.parse_args()

    if not 0.0 <= args.lo < args.hi <= 1.0:
        raise SystemExit(f"bounds must satisfy 0 <= lo < hi <= 1, got lo={args.lo} hi={args.hi}")

    if os.path.isdir(args.rows):
        keys = keys_from_dump(args.rows, args.split)
        source = f"reward dump {args.rows}" + (f" (split={args.split})" if args.split else "")
    else:
        keys = keys_from_parquet(args.rows)
        source = f"parquet {args.rows}"

    if not keys:
        raise SystemExit(f"no rows found in {source}")

    unique = sorted(set(keys))
    fractions = {k: slice_fraction(*k.split("::", 2)) for k in unique}
    offenders = [(k, u) for k, u in fractions.items() if not args.lo <= u < args.hi]

    lo_seen = min(fractions.values())
    hi_seen = max(fractions.values())
    print(f"{source}: {len(keys)} rows, {len(unique)} distinct turns")
    print(f"  observed hash range [{lo_seen:.4f}, {hi_seen:.4f}]  "
          f"required [{args.lo}, {args.hi})")

    if offenders:
        print(f"FAIL: {len(offenders)} turn(s) outside [{args.lo}, {args.hi})", file=sys.stderr)
        for k, u in offenders[:10]:
            print(f"  {k}  u={u:.6f}", file=sys.stderr)
        if len(offenders) > 10:
            print(f"  ... and {len(offenders) - 10} more", file=sys.stderr)
        raise SystemExit(1)

    print(f"OK: all {len(unique)} turns in [{args.lo}, {args.hi})")


if __name__ == "__main__":
    main()
