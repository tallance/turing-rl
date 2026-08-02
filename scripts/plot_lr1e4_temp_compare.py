"""lr=1e-4 cell: win-rate-over-time comparison across model size, sampling temperature, and split.

Overlays, on one axis (3-epoch rolling mean per series):
  - 8B temp-1.0 TRAIN (temp 1.0) + INFER/val (temp 0.7)   [8b_proper_kl1e4_lr1e4_temp1_valcard]
  - 9B temp-1.0 TRAIN (temp 1.0) + INFER/val (temp 0.7)   [9b_proper_kl1e4_lr1e4_temp1_valcard]
  - 8B / 9B temp-0.6 TRAIN baseline (paper sampling; train-only, no val loop)

Win-rate per epoch = wins(Likert>=5)/non-tie, ties(4)/parse-fails(0/None) excluded (same defn as
plot_temp1_valcard / plot_winrate_over_time). Reuses that module's epoch reconstruction:
train = per-example ts-chunk by G; val = global ts-burst split (--val_gap). Color=model,
linestyle: temp1-train solid, temp1-val dashed, temp0.6-train dotted.

Usage (cluster; env turing-rl-train):
  python scripts/plot_lr1e4_temp_compare.py \
    --temp1 "8B:results/grpo/rl-generator/8b_proper_kl1e4_lr1e4_temp1_valcard/reward_dump" \
    --temp1 "9B:results/grpo/rl-generator/9b_proper_kl1e4_lr1e4_temp1_valcard/reward_dump" \
    --temp06 "8B:results/grpo/rl-generator/8b_proper_kl1e4_lr1e4/reward_dump" \
    --temp06 "9B:results/grpo/rl-generator/9b_proper_kl1e4_lr1e4/reward_dump" \
    --out results/2026-07-24-reward-hack-proper-checkpoint/temp1-valcard-9b/lr1e4_temp_compare_winrate.png
"""
from __future__ import annotations
import argparse, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))  # allow flat import of sibling
from plot_temp1_valcard import _load, train_epochs, val_passes, _winrate_series

_COLOR = {"8B": "tab:blue", "9B": "tab:orange"}


def _roll(wr, k=3):
    return [sum(wr[max(0, j - k + 1):j + 1]) / len(wr[max(0, j - k + 1):j + 1]) for j in range(len(wr))]


def _parse(specs):
    out = []
    for s in specs or []:
        label, _, d = s.partition(":")
        out.append((label, d))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--temp1", action="append", help="label:dump_dir (has train+val split)")
    ap.add_argument("--temp06", action="append", help="label:dump_dir (train-only baseline)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_size", type=int, default=4)
    ap.add_argument("--val_gap", type=float, default=400.0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    for label, d in _parse(a.temp1):
        rows = _load(d)
        tr = [r for r in rows if r.get("split") == "train"]
        va = [r for r in rows if r.get("split") == "val"]
        c = _COLOR.get(label, "tab:green")
        xt, wt = _winrate_series(train_epochs(tr, a.group_size), 1)
        xv, wv = _winrate_series(val_passes(va, a.val_gap), 0)
        ax.plot(xt, _roll(wt), color=c, ls="-", lw=1.9, alpha=0.9,
                label=f"{label} temp1.0 TRAIN (final {wt[-1]:.2f})")
        ax.plot(xv, _roll(wv), color=c, ls="--", lw=1.7, alpha=0.9,
                label=f"{label} temp0.7 INFER (final {wv[-1]:.2f})")

    for label, d in _parse(a.temp06):
        rows = _load(d)                       # temp-0.6 baselines are all train (split None or 'train')
        c = _COLOR.get(label, "tab:green")
        xb, wb = _winrate_series(train_epochs(rows, a.group_size), 1)
        ax.plot(xb, _roll(wb), color=c, ls=":", lw=2.0, alpha=0.8,
                label=f"{label} temp0.6 TRAIN baseline (final {wb[-1]:.2f})")

    ax.axhline(0.5, ls="-", color="gray", lw=0.8, alpha=0.6)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("overfit epoch (3-epoch rolling mean; train G rollouts/example, ts-chunked; val=pass)")
    ax.set_ylabel("overall win-rate (Likert>=5; ties/parse-fails excluded)")
    ax.set_title("lr=1e-4 reward-hack: win-rate vs epoch — 8B vs 9B, temp-1.0 train/val vs temp-0.6 train\n"
                 "(color=model; solid=temp1.0 train, dashed=temp0.7 infer, dotted=temp0.6 train)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
