"""Scatter the 9B judge's rating against the 27B judge's, all GRPO steps on one plot.

Reads the tidy per-pair table from scripts/export_judge_rating_pairs.py and draws
one panel: x = Qwen3.5-9B Likert, y = Qwen3.5-27B Likert, coloured by GRPO step
(step 0 blue -> step 32 red).

ONE DOT PER PAIR, JITTERED. Both axes are integers 1-7, so without jitter the 4396
rated pairs would stack onto at most 49 lattice points -- 665 of them land on a
single point, 172 on a single (step, point) -- and the scatter would be 49 dots.
So every pair is drawn, displaced from its lattice point by a deterministic random
offset. Two things shape that offset:

  1. A fixed per-step ring offset, so the five steps form five separate clouds
     around each lattice point instead of five blue->red clouds mixing to brown.
  2. Random jitter, uniform inside a disc whose RADIUS SCALES AS sqrt(count). Area
     then grows in proportion to the count, so every cloud has the same dot
     density and a cloud's size reads directly as how many pairs are in it. Fixed-
     radius jitter would instead make a 172-pair cloud and a 5-pair cloud the same
     size and leave only opacity to tell them apart.

The jitter is cosmetic: it moves where a dot is drawn, never what it counts as.
Ratings are read from the CSV as integers, and the companion table CSV carries the
exact per-cell counts.

The step centroids (mean 9B, mean 27B) are overlaid and joined 0 -> 32, because the
drift of the cloud is the thing the individual dots are too dense to show.

Colour is a diverging ramp: two opposite hues with a NEUTRAL GRAY midpoint (never a
hue at the midpoint). Steps are ordered, so the ramp is read as position along
training, and the legend spells out every step -- identity is never colour-alone.

Usage:
  python scripts/plot_judge_rating_scatter.py \
      --eval_root results/2026-08-03-test-eval-9b-half [--out_dir <dir>]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotstyle import INK, apply_rc, style_axes  # noqa: E402

X_JUDGE, Y_JUDGE = "r_qwen35_9b", "r_qwen35_27b"
X_LABEL, Y_LABEL = "Qwen3.5-9B rating", "Qwen3.5-27B rating"

# Diverging ramp, assigned by STEP (an ordered quantity), blue -> neutral gray -> red.
# Fixed by entity: adding or dropping a step must not repaint the others.
#
# Stepped in OKLCh so LIGHTNESS, not just hue, separates the five: poles dark (L~0.40),
# mid steps L~0.58, neutral midpoint lightest (L~0.74). A first cut kept all three middle
# steps at L~0.63-0.69 and failed the checks below -- step 8 vs 16 came out at dE 11.8 and
# 16 vs 24 at 13.8 normal-vision (floor is 15), and 16 vs 24 at only 7.0 under protanopia.
#
# Validated with OKLab dE x100 and the Machado-2009 CVD matrices at severity 1.0. The
# dataviz skill's scripts/validate_palette.js needs a JS runtime, which this Mac does not
# have, so the same checks were computed directly. Results: lightness band PASS for all
# five; adjacent-pair separation >= 16.4 under deuter/protan/tritan (target 8); ALL pairs
# x all CVD >= 16.4; normal-vision floor 17.1 (hard floor 15). The midpoint sits below the
# chroma floor on purpose -- a diverging midpoint must read as neutral. Its contrast on the
# surface is 2.2, which obligates non-colour labelling: the legend names every step and the
# companion table CSV carries the same numbers.
STEP_COLOR = {
    0:  "#0b438d",   # blue pole
    8:  "#3d7bc0",
    16: "#aeaca5",   # neutral gray midpoint -- never a hue here
    24: "#bb582e",
    32: "#8d0d09",   # red pole
}
RING = 0.175         # radius of the per-step offset ring, in rating units
CLOUD_R = 0.150      # jitter disc radius for the BIGGEST cell; others scale as sqrt(count)
DOT_AREA = 3.5       # points^2 per dot -- one dot is one rated pair
DOT_ALPHA = 0.55
JITTER_SEED = 20260805   # fixed: the same CSV must always produce the same figure

# RING + CLOUD_R = 0.325 < 0.5, so a cloud can never spill into the neighbouring
# lattice point's territory and be misread as a different rating.
assert RING + CLOUD_R < 0.5


def load(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--csv", default=None, help="defaults to <eval_root>/judge_rating_pairs.csv")
    ap.add_argument("--out_dir", default=None, help="defaults to <eval_root>")
    a = ap.parse_args()

    root = Path(a.eval_root)
    src = Path(a.csv) if a.csv else root / "judge_rating_pairs.csv"
    out_dir = Path(a.out_dir) if a.out_dir else root
    if not src.exists():
        raise SystemExit(f"FAIL: {src} not found. Run scripts/export_judge_rating_pairs.py "
                         f"on the cluster and pull the CSV first.")

    rows = load(src)
    steps = sorted({int(r["step"]) for r in rows})
    if set(steps) != set(STEP_COLOR):
        raise SystemExit(f"FAIL: CSV covers steps {steps}, but the colour ramp is fixed for "
                         f"{sorted(STEP_COLOR)}. Extend STEP_COLOR deliberately rather than "
                         f"cycling hues.")

    # A pair is plotted only if BOTH judges returned a rating; a parse failure is not a 0.
    total = len(rows)
    rated = [r for r in rows if r[X_JUDGE] and r[Y_JUDGE]]
    dropped = total - len(rated)

    per_step: dict[int, Counter] = {s: Counter() for s in steps}
    for r in rated:
        per_step[int(r["step"])][(int(r[X_JUDGE]), int(r[Y_JUDGE]))] += 1
    biggest = max(c for s in steps for c in per_step[s].values())

    apply_rc()
    # Portrait-ish canvas: the axes is set to equal aspect (both axes are the same 1-7
    # rating scale), so a wide figure would just pad the sides with dead space.
    fig, ax = plt.subplots(figsize=(7.8, 8.7))

    lo, hi = 0.4, 7.6
    # y = x: where the two judges agree exactly. Recessive, solid hairline, behind data.
    ax.plot([lo, hi], [lo, hi], color=INK["axis"], lw=1.0, zorder=1)
    ax.annotate("9B = 27B", xy=(7.15, 7.15), xytext=(0, 6), textcoords="offset points",
                color=INK["muted"], fontsize=9, ha="right", rotation=45, rotation_mode="anchor")

    # One dot per rated pair. Seeded and iterated over sorted cells, so the same CSV
    # always yields pixel-identical jitter.
    rng = random.Random(JITTER_SEED)
    drawn = 0
    for i, s in enumerate(steps):
        ang = -math.pi / 2 + 2 * math.pi * i / len(steps)
        dx, dy = RING * math.cos(ang), RING * math.sin(ang)
        xs, ys = [], []
        for (gx, gy), n in sorted(per_step[s].items()):
            # radius ~ sqrt(count) => area ~ count => constant dot density everywhere.
            r_cloud = CLOUD_R * math.sqrt(n / biggest)
            for _ in range(n):
                th = rng.random() * 2 * math.pi
                rr = r_cloud * math.sqrt(rng.random())   # sqrt => uniform over the disc
                xs.append(gx + dx + rr * math.cos(th))
                ys.append(gy + dy + rr * math.sin(th))
        # No marker edge: a 2px surface ring is for separating a few overlapping markers,
        # and on hundreds of ~2px dots it would erase the fill it is meant to outline.
        ax.scatter(xs, ys, s=DOT_AREA, color=STEP_COLOR[s], alpha=DOT_ALPHA,
                   linewidths=0, zorder=3, rasterized=True)
        drawn += len(xs)

    # The whole point of a scatter is that every observation is on it.
    if drawn != len(rated):
        raise SystemExit(f"FAIL: drew {drawn} dots for {len(rated)} rated pairs")

    # Centroids: the drift the per-cell marks are too dense to show. 2px surface ring.
    cx, cy = [], []
    for s in steps:
        pts = [(int(r[X_JUDGE]), int(r[Y_JUDGE])) for r in rated if int(r["step"]) == s]
        cx.append(statistics.mean(p[0] for p in pts))
        cy.append(statistics.mean(p[1] for p in pts))
    ax.plot(cx, cy, color=INK["secondary"], lw=1.4, zorder=4, alpha=0.55)
    for s, x, y in zip(steps, cx, cy):
        ax.scatter([x], [y], s=150, marker="D", color=STEP_COLOR[s],
                   linewidths=2.0, edgecolors=INK["surface"], zorder=5)
    # Direct-label only the two endpoints of the drift, not every point. Both labels sit in
    # the half-integer band to the right of their diamond, which no lattice mark reaches.
    for s, x, y, off in ((steps[0], cx[0], cy[0], (13, -9)),
                         (steps[-1], cx[-1], cy[-1], (13, -7))):
        ax.annotate(f"step {s} mean", xy=(x, y), xytext=off, textcoords="offset points",
                    color=INK["primary"], fontsize=9.5, fontweight="bold")

    # Title is drawn on the figure, not the axes, so the header block (title + the three
    # subtitle lines) shares one left edge instead of the subtitle hanging off to its left.
    style_axes(ax, "", X_LABEL, Y_LABEL, list(range(1, 8)))
    ax.grid(True, axis="x", color=INK["grid"], lw=0.8, zorder=0)
    ax.set_yticks(range(1, 8))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")

    n_by_step = {s: sum(per_step[s].values()) for s in steps}
    fig.text(0.095, 0.985, "Judge agreement on the held-out test set, by GRPO step",
             color=INK["primary"], fontsize=13, fontweight="bold", va="top")
    fig.text(0.095, 0.951,
             f"One dot per rated pair. Ratings are integers, so each dot is jittered off its "
             f"lattice point: a fixed per-step\n"
             f"offset separates the five clouds, then random spread whose radius scales as "
             f"sqrt(count), making a cloud's\n"
             f"area proportional to the pairs in it (largest = {biggest}). "
             f"Diamonds are step means.\n"
             f"{len(rated)} of {total} pairs rated by both judges ({dropped} dropped: a judge "
             f"failed to parse). Per step: "
             f"{', '.join(f'{s}:{n_by_step[s]}' for s in steps)}.",
             color=INK["secondary"], fontsize=8.8, va="top", linespacing=1.5)

    step_handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=8,
                               markerfacecolor=STEP_COLOR[s], markeredgecolor=INK["surface"],
                               label=f"step {s}" + (" (SFT init)" if s == 0 else ""))
                    for s in steps]
    leg = ax.legend(handles=step_handles, title="GRPO step", loc="upper left",
                    frameon=False, fontsize=9.5, title_fontsize=9.5,
                    labelcolor=INK["secondary"])
    leg.get_title().set_color(INK["secondary"])

    # No size legend: dot density is constant, so a cloud's extent -- not any single
    # mark -- carries the count, and the exact numbers live in the table CSV.

    fig.subplots_adjust(left=0.095, right=0.985, top=0.855, bottom=0.065)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "judge_rating_scatter_9b_vs_27b"
    fig.savefig(f"{stem}.png", dpi=200)
    plt.close(fig)

    # Table-view twin: the WCAG-clean equivalent of a colour-encoded scatter.
    tbl = out_dir / "judge_rating_scatter_9b_vs_27b_table.csv"
    with open(tbl, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "rating_qwen35_9b", "rating_qwen35_27b", "n_pairs"])
        for s in steps:
            for (x, y), n in sorted(per_step[s].items()):
                w.writerow([s, x, y, n])
    print(f"wrote {stem}.png and {tbl.name}", file=sys.stderr)
    for s in steps:
        print(f"  step {s:>2}  n={n_by_step[s]}  mean 9B={cx[steps.index(s)]:.3f}  "
              f"mean 27B={cy[steps.index(s)]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
