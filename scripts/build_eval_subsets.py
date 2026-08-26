#!/usr/bin/env python3
"""Materialise the EXACT prompt sets a GRPO run trained and validated on.

WHY THIS IS NOT JUST "READ train.parquet"
-----------------------------------------
Job 13634 did not train on all of ``grpo/train.parquet`` (4174 rows). It passed
``data.train_max_samples=2048`` / ``data.val_max_samples=352``, and because ``data.shuffle=true``
veRL's RLHFDataset takes a SEEDED RANDOM subset, not the first N:

    verl/utils/dataset/rl_dataset.py::_read_files_and_tokenize
        rng = np.random.default_rng(seed)                     # data.seed = 42
        indices = rng.choice(total, size=max_samples, replace=False)
        self.dataframe = self.dataframe.select(indices.tolist())
        ...
        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

So "the training data" is a specific 2048 of 4174, reproducible only by replaying that draw.
Evaluating on the wrong 2048 would silently mix trained and never-trained prompts and destroy the
whole point of a train-set eval.

This script replays the draw and then PROVES it, by requiring the resulting
``(user_id, post_id, target_idx)`` key set to equal the unique keys the run actually judged for
that split (from its reward dump). That also implicitly confirms ``filter_overlong_prompts``
dropped nothing -- if it had, the dump would be short and the assertion would fail.

Usage (cluster; env turing-rl-train):
  python scripts/build_eval_subsets.py \
      --grpo_dir data/prism/full_s42_history_sft40_grpo60_test10/grpo \
      --reward_dump results/grpo/rl-generator/9b_half_kl1e4_lr1e4_temp1/reward_dump
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def keys_of(df: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [
        (str(e["user_id"]), str(e["post_id"]), str(e["target_idx"]))
        for e in df["extra_info"]
    ]


def dump_keys(dump_dir: Path, split: str) -> set[tuple[str, str, str]]:
    """Unique (user, post, target_idx) keys the run judged for `split`."""
    out: set[tuple[str, str, str]] = set()
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
                    continue
                if row.get("split") != split:
                    continue
                out.add((str(row["user_id"]), str(row["post_id"]), str(row["target_idx"])))
    if not out:
        raise SystemExit(f"FAIL: reward dump {dump_dir} has no rows with split={split!r}")
    return out


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_one(
    src: Path, out: Path, max_samples: int, seed: int, dump_dir: Path, split: str
) -> dict:
    df = pd.read_parquet(src)
    total = len(df)
    if max_samples >= total:
        raise SystemExit(
            f"FAIL: --{split}_max_samples={max_samples} >= {total} rows in {src}; "
            "the run would not have subsampled, so this script is the wrong tool."
        )
    # Replay veRL's draw verbatim: same rng, same call, and keep its (unsorted) order.
    indices = np.random.default_rng(seed).choice(total, size=max_samples, replace=False)
    subset = df.iloc[indices.tolist()].reset_index(drop=True)

    got = set(keys_of(subset))
    if len(got) != len(subset):
        raise SystemExit(f"FAIL: {split} subset has duplicate (user, post, target_idx) keys")

    want = dump_keys(dump_dir, split)
    if got != want:
        raise SystemExit(
            f"FAIL: replayed {split} subset does not match what the run judged.\n"
            f"  replayed={len(got)}  dumped={len(want)}\n"
            f"  in replay not dump: {len(got - want)}  in dump not replay: {len(want - got)}\n"
            "  The seed, max_samples, source parquet or veRL selection logic has changed. "
            "Do NOT evaluate on this subset."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)

    users = {k[0] for k in got}
    meta = {
        "source_parquet": str(src.resolve()),
        "source_rows": total,
        "output_parquet": str(out.resolve()),
        "output_sha256": sha256_file(out),
        "rows": len(subset),
        "users": len(users),
        "split": split,
        "selection": {
            "rule": "np.random.default_rng(seed).choice(total, size=max_samples, replace=False)",
            "seed": seed,
            "max_samples": max_samples,
            "source": "verl/utils/dataset/rl_dataset.py::_read_files_and_tokenize (data.shuffle=true)",
        },
        "verified_against_reward_dump": str(dump_dir.resolve()),
        "verified_key_count": len(want),
    }
    meta_path = out.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(
        f"[{split}] {len(subset)} rows / {len(users)} users -> {out}\n"
        f"        key set matches the reward dump exactly ({len(want)} keys)"
    )
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--grpo_dir", required=True, help="Dir holding train.parquet and val.parquet")
    ap.add_argument("--reward_dump", required=True, help="The GRPO run's reward_dump dir")
    ap.add_argument("--train_max_samples", type=int, default=2048)
    ap.add_argument("--val_max_samples", type=int, default=352)
    ap.add_argument("--seed", type=int, default=42, help="data.seed from the run")
    ap.add_argument("--train_out", default=None)
    ap.add_argument("--val_out", default=None)
    a = ap.parse_args()

    grpo = Path(a.grpo_dir)
    dump = Path(a.reward_dump)
    train_out = Path(a.train_out) if a.train_out else grpo / f"train_used{a.train_max_samples}.parquet"
    val_out = Path(a.val_out) if a.val_out else grpo / f"val_used{a.val_max_samples}.parquet"

    build_one(grpo / "train.parquet", train_out, a.train_max_samples, a.seed, dump, "train")
    build_one(grpo / "val.parquet", val_out, a.val_max_samples, a.seed, dump, "val")
    print("\nPASS: both subsets reproduce the prompt sets the run actually used")


if __name__ == "__main__":
    main()
