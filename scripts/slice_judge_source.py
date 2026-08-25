"""Write the hash slice of a source parquet to its own file, before generation runs.

``scripts/build_judge_train_pairs.py`` slices too, but it runs *after* generation. Handing
the full split to ``eval.generate_trained`` means generating k samples for every context in
the split and then discarding ~90% of them — on the train split that is ~16.7k generations
to keep ~1.7k, inside a single-GPU job whose output pickle is only written at the end.

``data.judge.slice.select_slice`` is a pure function of ``extra_info``, so moving it in
front of generation selects exactly the same rows. Pass the same ``--slice_lo/--slice_hi/
--limit`` to this script and to the builder: ``select_slice`` is idempotent on its own
output, so the builder's assertion that every sliced row has a generation still holds.

``--num_users`` on the generator is NOT a substitute: it truncates by user rather than by
hash, so the builder would then hit ``assert not missing``.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from data.judge.slice import select_slice


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-slice a judge source parquet")
    parser.add_argument("--source_parquet", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--slice_lo", type=float, default=0.0)
    parser.add_argument("--slice_hi", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    source = pd.read_parquet(args.source_parquet)
    sliced = select_slice(source, lo=args.slice_lo, hi=args.slice_hi, limit=args.limit)
    if sliced.empty:
        raise SystemExit(
            f"slice [{args.slice_lo}, {args.slice_hi}) of {args.source_parquet} is empty"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sliced.to_parquet(args.out, index=False)
    print(
        f"Wrote {len(sliced)} of {len(source)} rows -> {args.out} "
        f"(slice=[{args.slice_lo},{args.slice_hi}) limit={args.limit})"
    )


if __name__ == "__main__":
    main()
