"""Grouped bar chart: accuracy (judge picks true human) per model, thinking off vs on.
All cells treated equally (anchor is just another model). Reads derived/summary.parquet.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd

# x-axis order: qwen3.5 dense by size, then MoE by size, then the cross-family bonus.
ORDER = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen35-35b-a3b",
         "qwen35-122b", "qwen35-397b", "qwen3-8b"]
LABELS = {"qwen35-4b": "3.5-4B", "qwen35-9b": "3.5-9B", "qwen35-27b": "3.5-27B",
          "qwen35-35b-a3b": "3.5-35B-A3B", "qwen35-122b": "3.5-122B(Int4)",
          "qwen35-397b": "3.5-397B(Int4)", "qwen3-8b": "qwen3-8B"}


def main() -> None:
    ap = argparse.ArgumentParser()
    base = REPO_ROOT / "results" / "2026-07-08-judge-sweep" / "derived"
    ap.add_argument("--summary", type=Path, default=base / "summary.parquet")
    ap.add_argument("--out", type=Path, default=base / "plots" / "accuracy_bar.png")
    args = ap.parse_args()
    df = pd.read_parquet(args.summary)
    acc = df.pivot_table(index="cell", columns="mode", values="accuracy")
    cells = [c for c in ORDER if c in acc.index]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(cells)); w = 0.38
    off = [acc.loc[c].get("off", np.nan) for c in cells]
    on = [acc.loc[c].get("on", np.nan) for c in cells]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w / 2, off, w, label="thinking off", color="#4C78A8")
    b2 = ax.bar(x + w / 2, on, w, label="thinking on", color="#F58518")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            if not np.isnan(h):
                ax.text(r.get_x() + r.get_width() / 2, h + 0.005, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, ls="--", c="gray", lw=1, alpha=0.7)  # chance
    ax.text(len(cells) - 0.5, 0.505, "chance", color="gray", fontsize=8, va="bottom", ha="right")
    ax.set_xticks(x); ax.set_xticklabels([LABELS.get(c, c) for c in cells], rotation=20, ha="right")
    ax.set_ylabel("accuracy (picks true human, ties excluded)")
    ax.set_ylim(0.45, 0.85); ax.set_title("Turing-judge accuracy by model — thinking off vs on")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130); plt.close(fig)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
