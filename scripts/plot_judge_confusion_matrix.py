"""Confusion matrix of the 9B judge's rating against the 27B judge's, per GRPO step.

Both judges rate the SAME generated turn on the same 1-7 Likert scale, so the
question is agreement, not correlation: how often do they land on the same number,
and when they differ, by how much and in which direction.

One 7x7 panel per checkpoint step (small multiples), x = the 9B rating (the judge
GRPO optimized against), y = the 27B rating. A cell is the number of test-set pairs
with that (9B, 27B) pair of ratings. The colour ramp is shared across panels -- the
per-step sample sizes are within 2 pairs of each other, so counts are directly
comparable panel to panel.

Input is the tidy per-pair CSV from scripts/export_judge_rating_pairs.py. Pairs
either judge failed to parse (empty rating field) are dropped, and the count of
drops is reported so the drop is never silent.

Usage:
  python scripts/plot_judge_confusion_matrix.py \
      --csv results/2026-08-03-test-eval-9b-half/judge_rating_pairs.csv \
      --out results/2026-08-03-test-eval-9b-half/judge_confusion_9b_vs_27b.png
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, apply_rc  # noqa: E402

X_JUDGE, Y_JUDGE = "r_qwen35_9b", "r_qwen35_27b"
X_LABEL, Y_LABEL = "Qwen3.5-9B rating", "Qwen3.5-27B rating"

RATINGS = [1, 2, 3, 4, 5, 6, 7]

# Sequential = one hue, light to dark. Magnitude here is a count, so hue never varies.
# The lightest step still separates from the surface, so "1 pair" cannot read as empty.
RAMP = ["#e4eefa", "#b4d0ec", "#7fa9dd", "#4b7cc4", "#25569f", "#0d3269"]
# From this bin down the fill is dark enough that secondary ink loses contrast, so the
# printed count flips to the surface colour.
DARK_TEXT_BIN = 3


def load(csv_path: Path):
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"FAIL: {csv_path} has no rows")
    for col in ("step", X_JUDGE, Y_JUDGE):
        if col not in rows[0]:
            raise SystemExit(f"FAIL: {csv_path} has no column {col!r}")
    rated = [r for r in rows if r[X_JUDGE] and r[Y_JUDGE]]

    per_step: dict[int, Counter] = {}
    for r in rated:
        x, y = int(r[X_JUDGE]), int(r[Y_JUDGE])
        for v, who in ((x, X_JUDGE), (y, Y_JUDGE)):
            if v not in RATINGS:
                raise SystemExit(f"FAIL: {who} rating {v} is outside the 1-7 Likert scale")
        per_step.setdefault(int(r["step"]), Counter())[(x, y)] += 1
    return rows, rated, per_step


def cell_bin(n: int, vmax: int) -> int:
    """Snap a count onto the sequential ramp. -1 means 0 pairs, which keeps the surface."""
    if n == 0:
        return -1
    # Equal-width bins over 1..vmax; the top bin ends at the largest cell in the figure.
    return min(len(RAMP) - 1, int((n - 1) / max(vmax, 1) * len(RAMP)))


def draw_panel(ax, counts: Counter, step: int, vmax: int, show_y: bool, show_x: bool) -> None:
    import matplotlib.patches as mpatches

    n = sum(counts.values())
    exact = sum(v for (a, b), v in counts.items() if a == b)
    within1 = sum(v for (a, b), v in counts.items() if abs(a - b) <= 1)

    for x in RATINGS:
        for y in RATINGS:
            c = counts.get((x, y), 0)
            b = cell_bin(c, vmax)
            face = INK["surface"] if b < 0 else RAMP[b]
            # A 2px surface gap between fills, so cells are separated without borders.
            ax.add_patch(mpatches.Rectangle((x - 0.46, y - 0.46), 0.92, 0.92,
                                            facecolor=face, edgecolor="none", zorder=2))
            if x == y:
                # The tie cells. Ringed rather than crossed by a diagonal rule, so the
                # marking never runs through a printed count.
                ax.add_patch(mpatches.Rectangle((x - 0.46, y - 0.46), 0.92, 0.92,
                                                facecolor="none", edgecolor=INK["muted"],
                                                lw=0.9, zorder=4))
            if c:
                dark = b >= DARK_TEXT_BIN
                ax.text(x, y, str(c), ha="center", va="center", zorder=5,
                        fontsize=7.4, color=INK["surface"] if dark else INK["secondary"],
                        fontweight="bold" if x == y else "normal")

    ax.set_title(f"step {step}", color=INK["primary"], fontsize=11.5,
                 fontweight="bold", loc="left", pad=17)
    ax.text(0, 1.008, f"n = {n}   ·   exact {exact / n:.0%}   ·   within 1 {within1 / n:.0%}",
            transform=ax.transAxes, ha="left", va="bottom",
            color=INK["secondary"], fontsize=8.6)

    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(0.5, 7.5)
    ax.set_aspect("equal")
    ax.set_xticks(RATINGS)
    ax.set_yticks(RATINGS)
    ax.tick_params(colors=INK["muted"], labelsize=8.5, length=0)
    if not show_y:
        ax.set_yticklabels([])
    if not show_x:
        ax.set_xticklabels([])
    if show_x:
        ax.set_xlabel(X_LABEL, color=INK["secondary"], fontsize=9.5, labelpad=6)
    if show_y:
        ax.set_ylabel(Y_LABEL, color=INK["secondary"], fontsize=9.5, labelpad=6)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def draw_legend(ax, vmax: int, totals: list[tuple[int, int, int, int]]) -> None:
    """The scale legend plus the table of per-step agreement -- the table-view twin."""
    import matplotlib.patches as mpatches

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0, 0.97, "pairs per cell", color=INK["primary"], fontsize=10,
            fontweight="bold", ha="left", va="top")
    sw_w, sw_h, x0, y0 = 0.62 / len(RAMP), 0.05, 0.0, 0.855
    edges = [0] + [int(round((i + 1) / len(RAMP) * vmax)) for i in range(len(RAMP))]
    for i, c in enumerate(RAMP):
        ax.add_patch(mpatches.Rectangle((x0 + i * sw_w, y0 - sw_h), sw_w * 0.94, sw_h,
                                        facecolor=c, edgecolor="none"))
    for i in (0, len(RAMP) // 2, len(RAMP) - 1):
        ax.text(x0 + i * sw_w + sw_w * 0.47, y0 - sw_h - 0.018,
                f"{edges[i] + 1}–{edges[i + 1]}", ha="center", va="top",
                color=INK["muted"], fontsize=7.6)
    ax.text(x0, y0 - sw_h - 0.075,
            "empty cell = no pair got that rating combination\n"
            "ringed cell = both judges gave the same rating",
            ha="left", va="top", color=INK["muted"], fontsize=8, linespacing=1.6)

    ax.text(0, 0.60, "agreement by step", color=INK["primary"], fontsize=10,
            fontweight="bold", ha="left", va="top")
    cols = [(0.0, "step"), (0.24, "n"), (0.52, "exact"), (0.84, "within 1")]
    for cx, head in cols:
        ax.text(cx, 0.54, head, ha="left", va="top", color=INK["muted"], fontsize=8.4)
    for row, (step, n, exact, within1) in enumerate(totals):
        yy = 0.485 - row * 0.052
        vals = [str(step), str(n), f"{exact / n:.1%}", f"{within1 / n:.1%}"]
        for (cx, _), v in zip(cols, vals):
            ax.text(cx, yy, v, ha="left", va="top", color=INK["secondary"], fontsize=8.8,
                    family="monospace")

    pooled_n = sum(t[1] for t in totals)
    pooled_e = sum(t[2] for t in totals)
    pooled_w = sum(t[3] for t in totals)
    yy = 0.485 - len(totals) * 0.052 - 0.022
    ax.plot([0.0, 1.0], [yy + 0.030, yy + 0.030], color=INK["grid"], lw=0.9)
    vals = ["all", str(pooled_n), f"{pooled_e / pooled_n:.1%}", f"{pooled_w / pooled_n:.1%}"]
    for (cx, _), v in zip(cols, vals):
        ax.text(cx, yy, v, ha="left", va="top", color=INK["primary"], fontsize=8.8,
                family="monospace", fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default=None,
                    help="table-view CSV (default: <out stem>_table.csv)")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, rated, per_step = load(Path(a.csv))
    steps = sorted(per_step)
    dropped = len(rows) - len(rated)
    print(f"{len(rated)} rated pairs over steps {steps} "
          f"({dropped} dropped: a judge failed to parse)", file=sys.stderr)

    vmax = max(max(c.values()) for c in per_step.values())
    totals = [(s, sum(per_step[s].values()),
               sum(v for (x, y), v in per_step[s].items() if x == y),
               sum(v for (x, y), v in per_step[s].items() if abs(x - y) <= 1))
              for s in steps]

    apply_rc()
    ncol = 3
    nrow = -(-(len(steps) + 1) // ncol)          # +1 for the legend/table panel
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.35 * ncol, 4.9 * nrow))
    flat = list(axes.ravel())

    for i, s in enumerate(steps):
        # Label the outer edge only: repeating the axis on every panel is chrome.
        draw_panel(flat[i], per_step[s], s, vmax,
                   show_y=(i % ncol == 0), show_x=(i >= len(steps) - ncol))
    draw_legend(flat[len(steps)], vmax, totals)
    for ax in flat[len(steps) + 1:]:
        ax.axis("off")

    fig.text(0.055, 0.980,
             "Do the 9B and 27B judges give the same rating? Test set, by GRPO step",
             ha="left", va="top", color=INK["primary"], fontsize=14.5, fontweight="bold")
    fig.text(0.055, 0.951,
             f"{len(rated)} held-out pairs, each rated 1–7 by both judges (thinking on). "
             f"The ringed diagonal is where they tied; mass\nabove it means the 27B judge rated "
             f"the generated turn more human than the 9B judge did.",
             ha="left", va="top", color=INK["secondary"], fontsize=9.8, linespacing=1.5)

    fig.subplots_adjust(left=0.055, right=0.985, top=0.870, bottom=0.050,
                        wspace=0.16, hspace=0.22)
    out = Path(a.out)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}", file=sys.stderr)

    table = Path(a.table) if a.table else out.with_name(out.stem + "_table.csv")
    with open(table, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "rating_qwen35_9b", "rating_qwen35_27b", "n_pairs"])
        for s in steps:
            for x in RATINGS:
                for y in RATINGS:
                    w.writerow([s, x, y, per_step[s].get((x, y), 0)])
    print(f"wrote {table}", file=sys.stderr)


if __name__ == "__main__":
    main()
