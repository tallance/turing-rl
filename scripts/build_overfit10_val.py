"""Write a val copy of the overfit-10 GRPO parquet tagged extra_info['split']='val'.

For the temp-1.0 / model-card-val overfit probe (plan `before-we-move-on-hazy-puppy`): validation
uses the SAME 10 training turns, but tagged so reward-dump rows are separable from train rollouts
(reward.py records extra_info['split'] in the dump). Train parquet is untouched (no 'split' key ->
defaults to 'train').

Usage (cluster):
  python scripts/build_overfit10_val.py \
    --in  data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet \
    --out data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10_val.parquet
"""
from __future__ import annotations
import argparse

import pandas as pd

_DEF = "data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet"


def _tag_split_val(ei):
    # extra_info is a dict per row; copy and set split=val (leave everything else intact).
    d = dict(ei) if ei is not None else {}
    d["split"] = "val"
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=_DEF)
    ap.add_argument("--out", dest="out", default=_DEF.replace(".parquet", "_val.parquet"))
    a = ap.parse_args()

    df = pd.read_parquet(a.inp)
    assert "extra_info" in df.columns, f"no extra_info column in {a.inp}"
    df = df.copy()
    df["extra_info"] = df["extra_info"].map(_tag_split_val)
    df.to_parquet(a.out, index=False)
    # verify
    back = pd.read_parquet(a.out)
    splits = {e.get("split") for e in back["extra_info"]}
    print(f"wrote {a.out}  rows={len(back)}  splits={splits}")
    assert splits == {"val"}, f"unexpected splits: {splits}"


if __name__ == "__main__":
    main()
