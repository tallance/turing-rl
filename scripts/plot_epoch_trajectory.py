"""Plot judge metrics as a function of SFT epoch (checkpoint trajectory).

Companion to the generator sweep: instead of one snapshot at the end of SFT, this reads the
per-epoch generator entries (qwen3-8b-base + qwen3-8b-sft-ep{1,2,3}, and the 9B analogues) and
draws, for each barplot metric, a LINE TRAJECTORY over epochs (x = 0/1/2/3), one line per judge,
with a separate subplot per model (8B, 9B). Epoch 0 = the base model (no SFT).

Input: the comparison_summary.parquet written by scripts/analyze_generator_sweep.py (columns
generator, judge, mode + acc_parse_ok/acc_penalized/parse_error/tie_rate). We filter to the
judges + thinking mode of interest. LOWER accuracy = generator is MORE human-like.

Usage:
  python scripts/plot_epoch_trajectory.py \
      [--summary results/2026-07-15-generator-sweep/derived/compare/comparison_summary.parquet] \
      [--out_dir results/2026-07-21-sft-checkpoint-trajectory] \
      [--mode on] [--judges qwen35-4b qwen35-9b qwen35-27b]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# Same metric set + y-labels/ylim/reference-line as analyze_generator_sweep.CMP_METRICS,
# so the trajectory plots stay visually consistent with the existing barplots/comparisons.
METRICS = [
    ("acc_parse_ok", "accuracy | parse ok (picks true human)", (0.0, 0.85), 0.5),
    ("acc_penalized", "accuracy (parse-fail counted wrong)", (0.0, 0.85), 0.5),
    ("parse_error", "parse-error rate", None, None),
    ("tie_rate", "tie rate (rating==4)", None, None),
]
# (model key in generator string, subplot title). Order = subplot order (left to right).
MODELS = [("qwen35-9b", "qwen3.5-9B"), ("qwen3-8b", "qwen3-8B")]
DEFAULT_JUDGES = ["qwen35-4b", "qwen35-9b", "qwen35-27b", "qwen35-397b"]
JUDGE_COLORS = {"qwen35-4b": "tab:blue", "qwen35-9b": "tab:orange",
                "qwen35-27b": "tab:green", "qwen35-397b": "tab:red"}


def parse_model_epoch(gen: str) -> tuple[str | None, int | None]:
    """Map a generator key -> (model_key, epoch). base -> epoch 0; -sft-ep{k} -> k.

    Plain '-sft' (the 2x2 final, pre-fix for 8B) is intentionally excluded — the trajectory
    uses the dedicated per-epoch runs so all points come from one training run per model.
    """
    model = "qwen35-9b" if gen.startswith("qwen35-9b") else (
        "qwen3-8b" if gen.startswith("qwen3-8b") else None)
    if model is None:
        return None, None
    if gen.endswith("-base"):
        return model, 0
    m = re.search(r"-sft-ep(\d+)$", gen)
    if m:
        return model, int(m.group(1))
    return None, None  # skip plain -sft and anything else


def main() -> None:
    base = Path(__file__).resolve().parents[1] / "results" / "2026-07-15-generator-sweep"
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=Path,
                    default=base / "derived" / "compare" / "comparison_summary.parquet")
    ap.add_argument("--out_dir", type=Path,
                    default=Path(__file__).resolve().parents[1] / "results"
                    / "2026-07-21-sft-checkpoint-trajectory")
    ap.add_argument("--mode", default="on", help="thinking mode to plot (default: on)")
    ap.add_argument("--judges", nargs="+", default=DEFAULT_JUDGES)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_parquet(args.summary)
    df = df[(df["mode"] == args.mode) & (df["judge"].isin(args.judges))].copy()
    me = df["generator"].map(parse_model_epoch)
    df["model_key"] = me.map(lambda t: t[0])
    df["epoch"] = me.map(lambda t: t[1])
    df = df[df["model_key"].notna()].copy()
    df["epoch"] = df["epoch"].astype(int)

    plots_dir = args.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Tidy long-form dump for the README / downstream inspection.
    keep = ["model_key", "epoch", "judge", "mode"] + [m[0] for m in METRICS]
    tidy = df[[c for c in keep if c in df.columns]].sort_values(["model_key", "judge", "epoch"])
    tidy.to_csv(args.out_dir / "trajectory.csv", index=False)

    n_written = 0
    for metric, ylab, ylim, ref in METRICS:
        if metric not in df.columns:
            print(f"[traj] metric {metric} absent — skipping", flush=True)
            continue
        fig, axes = plt.subplots(1, len(MODELS), figsize=(6.2 * len(MODELS), 4.8),
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for ax, (mkey, mtitle) in zip(axes, MODELS):
            sub = df[df["model_key"] == mkey]
            for judge in args.judges:
                js = sub[sub["judge"] == judge].sort_values("epoch")
                if js.empty:
                    continue
                ax.plot(js["epoch"].to_numpy(), js[metric].to_numpy(), marker="o",
                        color=JUDGE_COLORS.get(judge), label=judge)
            if ref is not None:
                ax.axhline(ref, ls="--", c="gray", lw=1)
            if ylim:
                ax.set_ylim(*ylim)
            ax.set_xticks([0, 1, 2, 3])
            ax.set_xlabel("SFT epoch (0 = base)")
            ax.set_title(mtitle)
            ax.grid(alpha=0.25)
            ax.legend(title="judge", fontsize=8)
        axes[0].set_ylabel(ylab)
        fig.suptitle(f"{metric} vs SFT epoch — thinking {args.mode} "
                     f"(lower accuracy = more human-like)")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        out = plots_dir / f"traj_{metric}_{args.mode}.png"
        fig.savefig(out, dpi=130)
        plt.close(fig)
        n_written += 1
        print(f"[traj] wrote {out}", flush=True)

    print(f"[traj] {n_written} figures + trajectory.csv in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
