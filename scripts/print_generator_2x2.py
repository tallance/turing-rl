"""Print the generator-sweep 2x2 table (judge accuracy at picking the true human).

LOWER acc = the generator is MORE human-like (fools the judge more). Reads the comparison
summary written by analyze_generator_sweep.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT = Path("results/2026-07-15-generator-sweep/derived/compare/comparison_summary.parquet")
GENS = ["qwen3-8b-base", "qwen3-8b-sft", "qwen35-9b-base", "qwen35-9b-sft"]
JUDGES = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen35-35b-a3b", "qwen35-122b", "qwen35-397b"]


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    df = pd.read_parquet(path)
    for mode in ("off", "on"):
        print(f"\n=== acc_parse_ok | thinking {mode} (LOWER = more human-like) ===")
        print("judge".ljust(16) + "".join(g.rjust(16) for g in GENS))
        sub = df[df["mode"] == mode]
        for j in JUDGES:
            cells = []
            for g in GENS:
                r = sub[(sub["generator"] == g) & (sub["judge"] == j)]
                cells.append(f"{r['acc_parse_ok'].iloc[0]:.3f}" if len(r) else "NA")
            print(j.ljust(16) + "".join(c.rjust(16) for c in cells))


if __name__ == "__main__":
    main()
