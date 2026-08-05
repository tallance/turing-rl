"""Plot train / val / test against the 9B judge: one figure, two panels.

  left   mean judge Likert rating (1-7); 4 = "cannot tell", the tie point
  right  win rate, i.e. fraction of pairs rated >= 5 (judge prefers the
         generated turn); 0.5 = parity

All three splits are scored by the SAME judge (Qwen3.5-9B), the one the GRPO run
was trained against, so the curves differ by which prompts/users were seen, not
by who did the grading.

Two things about the train curve are NOT apples-to-apples with val/test, and the
figure says so in its subtitle:

  * Train rollouts are sampled at the training temperature (temp=1, per the run
    tag 9b_half_kl1e4_lr1e4_temp1); val and test use the model-card decode
    (temp 0.7 / top_p 0.8 / top_k 20).
  * A train point aggregates the 8 GRPO steps ENDING at that x -- the reward dump
    has no step field, and judge latency smears rows across step boundaries, so
    the 8-step blocks between val passes are the finest exact split available.
    There is therefore no train point at step 0: no rollouts exist before training.

Inputs (both under --eval_root):
  train_val_splits.csv   scripts/summarize_train_val_splits.py, over the run's reward_dump
  summary_qwen35-9b.csv  scripts/summarize_test_eval.py --cell qwen35-9b

Usage:
  python scripts/plot_train_val_test.py \
      --eval_root results/2026-08-03-test-eval-9b-half [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import INK, SLOT, apply_rc, declutter, style_axes  # noqa: E402

# Fixed slot order by entity. Test keeps slot 2 (orange) so that the same
# quantity -- test set under the 9B judge -- is the same colour here as it is in
# test_eval_judges.png, where 9B is the emphasised judge.
SPLITS = [
    ("train", "Train", SLOT[1]),
    ("val", "Val", SLOT[3]),
    ("test", "Test", SLOT[2]),   # drawn last: the held-out headline sits on top
]

PANELS = [
    ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
    ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
]

STEP_RE = re.compile(r"step(\d+)$")


def read_train_val(path: Path) -> dict[str, list[dict]]:
    """Split the aggregator's long-format CSV into per-split rows sorted by step."""
    out: dict[str, list[dict]] = {"train": [], "val": []}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            s = r["split"]
            if s not in out:
                raise SystemExit(f"FAIL: unexpected split {s!r} in {path}")
            out[s].append({
                "step": int(r["step"]),
                "likert_mean": float(r["likert_mean"]),
                "win_rate_ge5": float(r["win_rate_ge5"]),
                "n": int(r["n_likert"]),
                "n_total": int(r["n_rows"]),
            })
    for s, rows in out.items():
        if not rows:
            raise SystemExit(f"FAIL: {path} has no {s} rows")
        rows.sort(key=lambda d: d["step"])
    return out


def read_test(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            m = STEP_RE.search(r["checkpoint"])
            if not m:
                raise SystemExit(f"FAIL: cannot parse a step from {r['checkpoint']!r} in {path}")
            rows.append({
                "step": int(m.group(1)),
                "likert_mean": float(r["likert_mean"]),
                "win_rate_ge5": float(r["win_rate_ge5"]),
                "n": int(r["n_likert"]),
                "n_total": int(r["n_scored"]),
            })
    return sorted(rows, key=lambda d: d["step"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--out_dir", default=None, help="defaults to --eval_root")
    ap.add_argument("--stem", default="train_val_test_9b")
    a = ap.parse_args()

    root = Path(a.eval_root)
    out_dir = Path(a.out_dir) if a.out_dir else root
    out_dir.mkdir(parents=True, exist_ok=True)

    tv_path = root / "train_val_splits.csv"
    test_path = root / "summary_qwen35-9b.csv"
    for p, how in ((tv_path, "scripts/summarize_train_val_splits.py"),
                   (test_path, "scripts/summarize_test_eval.py --cell qwen35-9b")):
        if not p.exists():
            raise SystemExit(f"FAIL: missing {p} -- produce it with {how}")

    data = read_train_val(tv_path)
    data["test"] = read_test(test_path)

    # Val and test are the two curves that must line up step-for-step; train is
    # allowed to be short one point (no rollouts exist before training starts).
    val_steps = [r["step"] for r in data["val"]]
    test_steps = [r["step"] for r in data["test"]]
    if val_steps != test_steps:
        raise SystemExit(f"FAIL: val covers {val_steps} but test covers {test_steps}; "
                         f"the curves would not be comparable")
    train_steps = [r["step"] for r in data["train"]]
    if not set(train_steps) <= set(val_steps):
        raise SystemExit(f"FAIL: train steps {train_steps} are not a subset of {val_steps}")

    steps = val_steps
    # Pairs per point, and the worst-case share of them the judge failed to return
    # a parseable 1-7 for. Quoting a single n_likert would overstate precision --
    # it varies by a few rows from point to point.
    nominal = {}
    for k, rows in data.items():
        sizes = {r["n_total"] for r in rows}
        if len(sizes) != 1:
            raise SystemExit(f"FAIL: {k} points have unequal pair counts {sorted(sizes)}")
        nominal[k] = sizes.pop()
    worst_drop = max((r["n_total"] - r["n"]) / r["n_total"]
                     for rows in data.values() for r in rows)

    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    for ax, (field, title, ylab, ref, ref_note) in zip(axes, PANELS):
        # Reference line first, so data marks draw over it.
        ax.axhline(ref, color=INK["muted"], lw=1, ls=(0, (4, 3)), zorder=1)

        ends = []
        for i, (key, label, color) in enumerate(SPLITS):
            xs = [r["step"] for r in data[key]]
            ys = [r[field] for r in data[key]]
            ax.plot(xs, ys,
                    color=color, lw=2.4,
                    marker="o", markersize=8,
                    markerfacecolor=color,
                    # 2px surface ring keeps overlapping markers separable.
                    markeredgecolor=INK["surface"], markeredgewidth=2,
                    zorder=3 + i, label=label, solid_capstyle="round")
            ends.append((ys[-1], label))

        # Anchor the note to the RIGHT end of the reference line: at the last step
        # every curve sits well above it in both panels, so nothing overprints the
        # text. (At the left end the rising curves cross straight through it.)
        ax.annotate(ref_note, xy=(steps[-1], ref), xytext=(0, 5), textcoords="offset points",
                    color=INK["muted"], fontsize=8.5, va="bottom", ha="right", zorder=1)

        ax.set_xlim(steps[0] - 1.5, steps[-1] + 5.5)   # headroom for the direct labels

        # Direct labels at the line ends, in ink rather than the series colour --
        # the adjacent mark carries identity. Spread them so they never overlap.
        lo, hi = ax.get_ylim()
        placed = declutter([(y - lo) / (hi - lo) for y, _ in ends])
        for (y, label), yf in zip(ends, placed):
            ax.annotate(label,
                        xy=(steps[-1], lo + yf * (hi - lo)),
                        xytext=(9, 0), textcoords="offset points",
                        color=INK["secondary"], fontsize=10.5,
                        va="center", ha="left", zorder=8, annotation_clip=False)

        style_axes(ax, title, "GRPO step", ylab, steps)

    handles, labels = axes[0].get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
                     bbox_to_anchor=(0.5, -0.015), fontsize=10.5)
    for t in leg.get_texts():
        t.set_color(INK["secondary"])

    fig.suptitle("9B GRPO generator: train vs val vs test, all judged by Qwen3.5-9B",
                 x=0.008, ha="left", color=INK["primary"], fontsize=14,
                 fontweight="bold", y=0.995)
    fig.text(0.008, 0.935,
             f"Per point: {nominal['train']} train rollouts, {nominal['val']} val, "
             f"{nominal['test']} test; unparseable ratings dropped "
             f"({worst_drop:.1%} at worst).\n"
             f"Train is sampled at the training temperature (1.0) while val/test use "
             f"the model-card decode (0.7), and each train point covers the 8 steps "
             f"ending at that x.",
             ha="left", va="top", color=INK["muted"], fontsize=9.5, linespacing=1.5)

    # Header (title + 2-line subtitle) lives inside the figure box, so the rect
    # top has to clear it -- tight_layout does not see fig.text/suptitle.
    fig.tight_layout(rect=(0, 0.07, 1, 0.86))

    for ext in ("png", "pdf"):
        p = out_dir / f"{a.stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
