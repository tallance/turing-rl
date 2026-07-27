"""Win-rate-over-time plot for RL-generator overfit runs (one subplot per run).

For each run, reads reward-*.jsonl dumps, groups rows by (user_id, post_id,
target_idx), orders each example's rows by ts and chunks them into epochs of
`group_size` rollouts (GRPO G). For each epoch, pools ALL examples' rollouts in
that epoch and computes the OVERALL win-rate = wins / non-tie, where a win is
Likert>=5 (judge picks the generated turn) and ties (Likert==4) and parse-fails
(0/None) are excluded. Plots win-rate vs epoch, one subplot per run, with a 0.5
reference line. This is the aggregate-metric companion to the per-example
Likert scatter (plot_overfit_ratings.py).

Usage:
  python scripts/plot_winrate_over_time.py \
    --run "kl=1e-3:/path/9b_overfit/reward_dump" \
    --run "kl=1e-4:/path/9b_overfit_kl1e4/reward_dump" \
    --run "kl=0:/path/9b_overfit_kl0/reward_dump" \
    --run "lr=1e-4:/path/9b_overfit_lr1e4_kl1e3/reward_dump" \
    --out results/.../winrate_over_time.png [--group_size 4]
"""
from __future__ import annotations
import argparse, glob, json, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(dump_dir: str) -> list[dict]:
    rows = []
    for f in glob.glob(os.path.join(dump_dir, "reward-*.jsonl")):
        for line in open(f):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def _key(r: dict) -> tuple:
    return (r.get("user_id"), r.get("post_id"), r.get("target_idx"))


def winrate_series(dump_dir: str, G: int):
    """Return (epochs, winrates, n_nontie_per_epoch, final_winrate). Overall win-rate
    per epoch pooled across all examples' G-rollout chunks (ts-ordered)."""
    rows = _load(dump_dir)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(_key(r), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.get("ts") or 0)

    # max number of epochs across examples
    max_epochs = max((len(v) + G - 1) // G for v in groups.values()) if groups else 0
    epochs, winrates, counts = [], [], []
    for e in range(max_epochs):
        wins = 0
        nontie = 0
        for v in groups.values():
            chunk = v[e * G:(e + 1) * G]
            for r in chunk:
                x = r.get("turing_judge_score_raw")
                if x is None:
                    continue
                xi = int(round(float(x)))
                if xi == 0 or xi == 4:   # parse-fail / tie excluded
                    continue
                nontie += 1
                if x >= 5:
                    wins += 1
        if nontie > 0:
            epochs.append(e + 1)
            winrates.append(wins / nontie)
            counts.append(nontie)
    final = winrates[-1] if winrates else 0.0
    return epochs, winrates, counts, final, len(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                    help="label:dump_dir  (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_size", type=int, default=4)
    a = ap.parse_args()

    runs = []
    for spec in a.run:
        label, _, d = spec.partition(":")
        runs.append((label, d))

    n = len(runs)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow), sharey=True, sharex=True)
    axes = axes.flatten() if n > 1 else [axes]

    for i, (label, d) in enumerate(runs):
        ax = axes[i]
        ep, wr, cnt, final, nrows = winrate_series(d, a.group_size)
        ax.plot(ep, wr, color="tab:blue", lw=1.3, marker="o", ms=3, alpha=0.85, label="per-epoch win-rate")
        # 3-epoch rolling mean for trend
        if len(wr) >= 3:
            roll = [sum(wr[max(0, j - 2):j + 1]) / len(wr[max(0, j - 2):j + 1]) for j in range(len(wr))]
            ax.plot(ep, roll, color="tab:red", lw=1.8, alpha=0.9, label="3-epoch rolling mean")
        ax.axhline(0.5, ls="--", color="gray", lw=0.9)
        # step-0 baseline = epoch-1 win-rate (rollouts from the initial SFT policy, pre-update)
        step0 = wr[0] if wr else 0.0
        ax.axhline(step0, ls=":", color="tab:green", lw=1.4, label="step-0 baseline (epoch 1)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{label}   step0={step0:.2f} → final={final:.2f}  ({len(ep)} epochs, {nrows} rows)", fontsize=10)
        ax.grid(True, alpha=0.25)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("RL-generator overfit gate: overall win-rate vs epoch (10 examples pooled per epoch; "
                 "win=Likert>=5, ties/parse-fails excluded; gray=0.5)", fontsize=11)
    fig.supxlabel("overfit epoch (G rollouts/example/epoch, ts-chunked)")
    fig.supylabel("overall win-rate")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.savefig(a.out, dpi=120)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
