"""3xN grid of judge-Likert histograms: columns = TRAIN | VAL, rows = chronological blocks (val passes).

Reads a reward_dump, splits rows by extra_info['split'] (train/val). Val is scored in discrete passes
(val_before_train + every test_freq steps); we recover the passes by ts-gap-splitting the val rows
(gap > --val_gap => new pass). Those pass start-times define block boundaries; each row of the figure is
one time block containing (left) the TRAIN rollouts in that block and (right) that block's VAL pass.
Each subplot is a histogram over integer Likert 1..7 (parse-fails / 0 excluded, counted in the title).

Usage (cluster; env turing-rl-train):
  python scripts/plot_likert_hist_train_val.py \
    --dump results/grpo/rl-generator/9b_half_kl1e4_lr1e4_temp1/reward_dump \
    --out  results/grpo/rl-generator/9b_half_kl1e4_lr1e4_temp1/likert_hist_train_val.png \
    [--val_gap 400]
"""
from __future__ import annotations
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(dump_dir):
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for l in open(f):
            try:
                rows.append(json.loads(l))
            except Exception:
                continue
    return rows


def _likerts(rows):
    """Valid integer Likerts 1..7 (drop parse-fails / 0)."""
    vals = []
    for r in rows:
        v = r.get("turing_judge_score_raw")
        if v is None or int(round(float(v))) == 0:
            continue
        vals.append(int(round(float(v))))
    return vals


def _hist(ax, vals, title, color):
    import statistics
    ax.hist(vals, bins=[b - 0.5 for b in range(1, 9)], color=color, alpha=0.8, edgecolor="white")
    ax.axvline(4.5, ls="--", color="green", lw=0.9)   # win threshold Likert>=5
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.5, 7.5)
    if vals:
        mean = statistics.mean(vals)
        win = sum(1 for v in vals if v >= 5) / len(vals)
        f7 = sum(1 for v in vals if v == 7) / len(vals)
        ax.set_title(f"{title}  n={len(vals)}  mean={mean:.2f}  win={win:.2f}  %7={f7*100:.1f}", fontsize=9)
    else:
        ax.set_title(f"{title}  (no data)", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val_gap", type=float, default=400.0)
    a = ap.parse_args()

    rows = _load(a.dump)
    train = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r.get("ts") or 0.0)
    val = sorted((r for r in rows if r.get("split") == "val"), key=lambda r: r.get("ts") or 0.0)

    # recover val passes by ts-gap split
    passes, cur, prev = [], [], None
    for r in val:
        t = r.get("ts") or 0.0
        if prev is not None and (t - prev) > a.val_gap and cur:
            passes.append(cur); cur = []
        cur.append(r); prev = t
    if cur:
        passes.append(cur)

    # block edges from val-pass start times: block i = [starts[i], starts[i+1])
    starts = [p[0].get("ts") or 0.0 for p in passes]
    edges = starts + [float("inf")]
    n = len(passes)

    fig, axes = plt.subplots(n, 2, figsize=(11, 2.7 * n), sharex=True, sharey=False, squeeze=False)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        tr_block = [r for r in train if lo <= (r.get("ts") or 0.0) < hi]
        tv = _likerts(tr_block)
        vv = _likerts(passes[i])
        step = "baseline (step 0)" if i == 0 else f"~step {i * 8}"
        _hist(axes[i][0], tv, f"TRAIN block {i} ({step})", "tab:blue")
        _hist(axes[i][1], vv, f"VAL pass {i} ({step})", "tab:orange")
        axes[i][0].set_ylabel("count")
    for j in range(2):
        axes[-1][j].set_xlabel("judge Likert (1-7; green=win>=5; parse-fails excluded)")
    fig.suptitle("9b_half_kl1e4_lr1e4_temp1: judge-Likert distribution by block — TRAIN (temp 1.0) vs VAL (temp 0.7)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}  (train_rows={len(train)} val_rows={len(val)} passes={n})")


if __name__ == "__main__":
    main()
