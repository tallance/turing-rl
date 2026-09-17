"""Real tokenizer lengths for a judge pair parquet, for setting data.max_prompt_length.

The pair builder's .meta.json reports `prompt_tokens_est_*`, which is chars / 3.9. That
estimate is not a substitute for this: on the rating_iter1 corpus it over-counted the maximum
by ~1.3x (est 7807 against a real 5866), which is safe in the conservative direction but wide
enough to pick a budget that wastes context -- or, if the bias ever runs the other way, one
that silently truncates.

`filter_overlong_prompts: true` DROPS over-budget rows from train AND val with nothing raising,
so this must be run against every split a config loads, and re-run per slice: the observed
maximum grows with sample size.

    python scripts/measure_prompt_tokens.py --parquet <train.parquet> <val.parquet> \
        --tokenizer Qwen/Qwen3.5-9B
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd


def measure(paths: list[str], tokenizer_id: str) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    combined_max = 0
    per_split = {}
    for path in paths:
        frame = pd.read_parquet(path)
        lengths = sorted(
            len(tokenizer(prompt[0]["content"])["input_ids"]) for prompt in frame["prompt"]
        )
        stats = {
            "n": len(lengths),
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(0.95 * len(lengths)) - 1],
            "p99": lengths[int(0.99 * len(lengths)) - 1],
            "max": lengths[-1],
        }
        per_split[path] = stats
        combined_max = max(combined_max, stats["max"])
    return {"per_split": per_split, "combined_max": combined_max}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", required=True, nargs="+")
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--margin", type=int, default=768,
        help="Headroom added to the combined maximum when suggesting max_prompt_length. The "
             "parent judge config runs ~730 tokens of slack over its measured maximum.",
    )
    args = parser.parse_args()

    result = measure(args.parquet, args.tokenizer)
    for path, stats in result["per_split"].items():
        print(f"{path}\n  " + "  ".join(f"{k}={v}" for k, v in stats.items()))

    combined = result["combined_max"]
    print(f"\ncombined max: {combined}")
    print(f"suggested data.max_prompt_length >= {combined + args.margin}")
    print("  (then set rollout.max_model_len = max_prompt_length + max_response_length)")


if __name__ == "__main__":
    main()
