"""Bar chart of judge accuracy at step 320 -- the figure that states the finding.

The two-panel trajectory figure plots Likert mean and generator win rate, which
describe how generous a judge is. This one plots the thing that decides whether
a judge is usable at all: how often it picks the REAL human. 0.5 is chance;
below it the judge is not merely noisy but reliably fooled.

Reads comparison_step320.csv from scripts/compare_judges_step320.py, so every
judge shown is scored on the identical pair set.

  python scripts/plot_judge_accuracy_step320.py --eval_root results/2026-09-15-...
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import INK, apply_rc, style_axes  # noqa: E402

STEP0_ANCHOR = 0.518  # same 9B judge on the pre-RL init, 880 pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--csv", default="comparison_step320.csv")
    ap.add_argument("--stem", default="judge_accuracy_step320")
    ap.add_argument("--title", default="Which judge can still tell a human from the generator?")
    a = ap.parse_args()

    root = Path(a.eval_root)
    rows = list(csv.DictReader(open(root / a.csv)))
    if not rows:
        raise SystemExit(f"FAIL: no rows in {root / a.csv}")
    # Worst at the top of the list -> drawn at the bottom of a horizontal chart.
    rows.sort(key=lambda r: float(r["judge_accuracy"]))

    labels = [r["judge"] for r in rows]
    acc = [float(r["judge_accuracy"]) for r in rows]
    lo = [float(r["acc_lo"]) for r in rows]
    hi = [float(r["acc_hi"]) for r in rows]
    n_pairs = {r["n_pairs"] for r in rows}
    # Frontier judges get the accent; the open-weight incumbents stay recessive.
    colors = ["#b5451f" if r["cell"].startswith("claude") else INK["secondary"]
              for r in rows]

    apply_rc()
    fig, ax = plt.subplots(figsize=(8.4, 0.52 * len(rows) + 2.4))

    ax.barh(labels, acc, height=0.62, color=colors, zorder=3)
    ax.errorbar(acc, labels,
                xerr=[[a_ - l for a_, l in zip(acc, lo)],
                      [h - a_ for a_, h in zip(acc, hi)]],
                fmt="none", ecolor=INK["primary"], elinewidth=1.4, capsize=4, zorder=4)

    ax.axvline(0.5, color=INK["primary"], lw=1.2, ls=(0, (4, 3)), zorder=2)
    ax.annotate("0.5 = chance", xy=(0.5, len(rows) - 0.4), xytext=(4, 0),
                textcoords="offset points", color=INK["primary"], fontsize=9,
                va="center", ha="left")
    ax.axvline(STEP0_ANCHOR, color=INK["muted"], lw=1, ls=(0, (1, 2)), zorder=2)
    ax.annotate(f"{STEP0_ANCHOR} = same 9B judge before GRPO (step 0)",
                xy=(STEP0_ANCHOR, -0.62), xytext=(4, 0), textcoords="offset points",
                color=INK["muted"], fontsize=8.5, va="center", ha="left")

    for y, (v, h) in enumerate(zip(acc, hi)):
        ax.annotate(f"{v:.3f}", xy=(h, y), xytext=(6, 0), textcoords="offset points",
                    color=INK["primary"], fontsize=10, va="center", ha="left")

    style_axes(ax, "", "judge accuracy: picked the real human", "", [])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(True, axis="x", color=INK["grid"], lw=0.8, zorder=0)
    ax.grid(False, axis="y")
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="y", labelsize=10.5, colors=INK["primary"])

    fig.suptitle(a.title, x=0.008, ha="left", color=INK["primary"],
                 fontsize=14, fontweight="bold", y=0.995)
    n_note = (f"{n_pairs.pop()} pairs" if len(n_pairs) == 1
              else "DIFFERING pair counts: " + ", ".join(sorted(n_pairs)))
    fig.text(0.008, 0.93,
             f"GRPO step 320, held-out test set, {n_note} -- every judge scored on the "
             f"identical pairs.\nBars below 0.5 are judges the generator reliably fools. "
             f"Whiskers are 95% Wilson intervals.",
             ha="left", va="top", color=INK["muted"], fontsize=9.5)

    fig.tight_layout(rect=(0, 0, 1, 0.87))
    p = root / f"{a.stem}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
