"""Per-example judge-rating plot for an RL-generator overfit run.

Reads reward-*.jsonl dumps, groups rows by (user_id, post_id, target_idx),
orders each example's rows by ts and chunks them into epochs of `group_size`
rollouts (GRPO G). Draws one subplot per example with BOTH:
  - the per-rollout individual Likert scores as a scatter (jittered), and
  - the per-epoch mean Likert as a line on top.
Reference lines: 5 = win threshold (Likert>=5 => judge picks the generated turn),
4 = tie. Parse-fail sentinels (turing_judge_score_raw==0 / None) are excluded
from the mean; shown as red x's in the scatter so they are visible.

Usage:
  python scripts/plot_overfit_ratings.py \
    --dump_dir results/grpo/rl-generator/9b_overfit/reward_dump \
    --out results/grpo/rl-generator/9b_overfit/rating_per_example_scatter.png \
    [--group_size 4]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_size", type=int, default=4, help="GRPO rollouts per prompt per epoch (G)")
    ap.add_argument("--jitter", type=float, default=0.08, help="x-jitter for scatter so overlapping rollouts are visible")
    a = ap.parse_args()

    rows = _load(a.dump_dir)
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(_key(r), []).append(r)
    for k in groups:
        groups[k].sort(key=lambda r: r.get("ts") or 0)

    # Fixed, canonical subplot order so the same example sits in the same panel
    # across every run's plot (dump insertion order otherwise varies per run).
    def _order(k: tuple):
        uid, pid, tgt = k
        # numeric-aware user id ("user1" < "user100"), then post id, then target idx
        import re
        m = re.search(r"(\d+)$", str(uid))
        uid_num = int(m.group(1)) if m else 0
        return (uid_num, str(uid), str(pid), tgt if tgt is not None else -1)

    keys = sorted(groups.keys(), key=_order)
    n = len(keys)
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 2.4 * nrow), sharey=True)
    axes = axes.flatten()
    G = a.group_size

    # deterministic pseudo-jitter (no RNG): spread the G rollouts evenly in [-jitter, +jitter]
    def _jit(j: int) -> float:
        if G <= 1:
            return 0.0
        return a.jitter * (2.0 * (j / (G - 1)) - 1.0)

    for i, k in enumerate(keys):
        ax = axes[i]
        rs = groups[k]
        epochs = [rs[j:j + G] for j in range(0, len(rs), G)]
        xs, means = [], []
        sx, sy = [], []          # valid (1-7) rollout scatter
        fx, fy = [], []          # parse-fail rollouts (0/None) shown distinctly
        for e_idx, chunk in enumerate(epochs, start=1):
            valid = []
            for j, r in enumerate(chunk):
                v = r.get("turing_judge_score_raw")
                if v is None or int(round(v)) == 0:
                    fx.append(e_idx + _jit(j)); fy.append(0.7)   # park parse-fails low
                    continue
                sx.append(e_idx + _jit(j)); sy.append(v)
                valid.append(v)
            if valid:
                xs.append(e_idx); means.append(sum(valid) / len(valid))
        ax.scatter(sx, sy, s=16, alpha=0.35, color="tab:blue", zorder=2, label="per-rollout")
        if fx:
            ax.scatter(fx, fy, s=18, marker="x", color="red", alpha=0.6, zorder=2, label="parse-fail")
        ax.plot(xs, means, color="tab:red", lw=1.4, zorder=3, label="epoch mean")
        ax.axhline(5, ls="--", color="green", lw=0.8)
        ax.axhline(4, ls=":", color="gray", lw=0.8)

        # final-epoch summary: how many of the last epoch's non-tie rollouts are wins (Likert>=5)
        last = [r.get("turing_judge_score_raw") for r in epochs[-1]] if epochs else []
        last = [v for v in last if v is not None and int(round(v)) != 0]
        nontie = [v for v in last if int(round(v)) != 4]
        wins = sum(1 for v in nontie if v >= 5)
        frac = wins / len(nontie) if nontie else 0.0
        ax.set_title(f"{k[0]}/{k[1]}/t{k[2]}   final wins={wins}/{len(nontie)} (frac={frac:.2f})", fontsize=9)
        ax.set_ylim(0.4, 7.4)
        ax.grid(True, alpha=0.2)

    for j in range(n, len(axes)):
        axes[j].axis("off")
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("9B overfit gate: per-example judge Likert vs epoch  (blue=per-rollout, red=epoch mean; green=win Likert>=5, gray=tie 4)")
    fig.supxlabel("overfit epoch (G rollouts/epoch, ts-chunked)")
    fig.supylabel("judge Likert (1-7)")
    fig.tight_layout(rect=[0, 0.02, 1, 0.98])
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}  ({n} examples, {len(rows)} rows)")


if __name__ == "__main__":
    main()
