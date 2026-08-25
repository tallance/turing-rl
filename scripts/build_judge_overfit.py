"""Cut a tiny, side-balanced overfit subset from the judge training parquet.

Selection is by *pair*, keeping both A/B orders together, so the human side stays exactly
balanced. An unbalanced overfit set would let the model saturate by always answering one
slot, which would pass the gate while proving nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def build_judge_overfit(
    src: str, out: str, n_pairs: int = 8, select: str = "first"
) -> pd.DataFrame:
    """Cut a tiny side-balanced subset.

    ``select='first'`` takes the leading pairs -- deterministic, and right for the R0 overfit
    gate, where the question is only whether the loop learns.

    ``select='longest'`` takes the pairs with the longest prompts. Use it whenever the subset
    stands in for the corpus in a MEMORY test. The leading 8 pairs top out at 7,083 prompt
    tokens against the full train set's 10,049, so a smoke built with 'first' silently omits the
    ~3,000-token tail that decides whether `log_softmax` fits: three full runs were launched on
    a budget a 'first' smoke had passed, and all three OOMed at exactly that site.
    """
    df = pd.read_parquet(src)
    required = ["data_source", "prompt", "reward_model", "extra_info"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing veRL columns: {missing}")
    if select not in {"first", "longest"}:
        raise ValueError(f"select must be 'first' or 'longest', got {select!r}")

    pair_ids = list(dict.fromkeys(e["pair_id"] for e in df["extra_info"]))
    if n_pairs > len(pair_ids):
        raise ValueError(f"requested {n_pairs} pairs but only {len(pair_ids)} exist in {src}")

    if select == "longest":
        # Rank by the longest prompt in each pair. Character length is a monotone proxy for
        # tokens here and avoids loading a tokenizer into this dependency-light script.
        longest_char = {}
        for prompt, info in zip(df["prompt"], df["extra_info"]):
            pid = info["pair_id"]
            longest_char[pid] = max(longest_char.get(pid, 0), len(prompt[0]["content"]))
        pair_ids = sorted(pair_ids, key=lambda p: longest_char[p], reverse=True)

    keep = set(pair_ids[:n_pairs])

    subset = df.loc[df["extra_info"].map(lambda e: e["pair_id"] in keep)].reset_index(drop=True)
    sides = [e["human_is_b"] for e in subset["extra_info"]]
    if sum(sides) * 2 != len(sides):
        raise ValueError("overfit subset is not side-balanced; both orders must be present")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    subset.to_parquet(out, index=False)
    return subset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the judge overfit subset")
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_pairs", type=int, default=8)
    parser.add_argument("--select", choices=("first", "longest"), default="first")
    args = parser.parse_args()
    subset = build_judge_overfit(args.src, args.out, args.n_pairs, args.select)
    longest = max(len(p[0]["content"]) for p in subset["prompt"])
    print(
        f"wrote {len(subset)} rows ({args.n_pairs} pairs x 2 orders, select={args.select}, "
        f"longest prompt {longest} chars) -> {args.out}"
    )


if __name__ == "__main__":
    main()
