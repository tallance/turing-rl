"""Bar chart of judge accuracy on the frozen 880-pair held-out set.

Form: magnitude across named models -> horizontal bars, sorted, zero baseline. The bars
start at 0 and the chance line sits at 0.5; a truncated axis would exaggerate the spread,
which for a bar chart is the classic distortion (bar length must stay proportional).

Colour encodes the only distinction that matters -- RL-trained vs zero-shot -- not one hue
per model. Slots 1 and 2 of the documented categorical palette, in fixed order.

Usage:
  python scripts/plot_judge_eval.py \
    --csv results/2026-08-12-judge-only-rlvr/judge_eval_880.csv \
    --out results/2026-08-12-judge-only-rlvr/judge_eval_880_accuracy.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

# Documented palette (references/palette.md), light mode.
SERIES_RL = "#2a78d6"       # categorical slot 1
SERIES_ZS = "#eb6834"       # categorical slot 2
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

DISPLAY = {
    "judge-4b-graded-step52": "Qwen3.5-4B judge — Brier RL",
    "judge-4b-directional-step52": "Qwen3.5-4B judge — 0/1 RL",
    "qwen35-27b": "Qwen3.5-27B",
    "gemma4-31b": "Gemma-4-31B",
    "gemma4-12b": "Gemma-4-12B",
    "qwen35-9b": "Qwen3.5-9B",
    "qwen35-4b": "Qwen3.5-4B",
}


def load(path: Path) -> list[dict]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["accuracy"] = float(row["accuracy"])
        row["se"] = float(row["se"])
        row["tie_rate"] = float(row["tie_rate"])
    # Ascending: barh fills bottom-up, so the strongest model lands at the top.
    return sorted(rows, key=lambda r: r["accuracy"])


def render(rows: list[dict], out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    height = 0.62  # < spacing, which leaves the surface gap between adjacent bars
    radius = height / 2.6
    for i, row in enumerate(rows):
        colour = SERIES_RL if row["kind"] == "rl-trained" else SERIES_ZS
        # Rounded data-end. Drawn as a rounded box then clipped at x=0 by the axis limit,
        # so the left end stays square against the baseline.
        ax.add_patch(
            FancyBboxPatch(
                (0, i - height / 2), row["accuracy"], height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=0.02,
                linewidth=0, facecolor=colour, zorder=3, clip_on=False,
            )
        )
        ax.errorbar(
            row["accuracy"], i, xerr=1.96 * row["se"],
            fmt="none", ecolor=INK_SECONDARY, elinewidth=1.2, capsize=3, zorder=4,
        )
        ax.text(
            row["accuracy"] + 1.96 * row["se"] + 0.012, i,
            f"{row['accuracy']:.3f}",
            va="center", ha="left", fontsize=9.5, color=INK_PRIMARY, zorder=5,
        )

    ax.axvline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    # Below the shortest bar, not above the tallest: at the top it crowds the title.
    ax.text(0.5, -0.62, "  chance (0.50)",
            color=INK_MUTED, fontsize=9, va="center", ha="left")

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY.get(r["model"], r["model"]) for r in rows], fontsize=10)
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlabel("accuracy on 880 held-out pairs (95% CI)", fontsize=10, color=INK_SECONDARY)
    ax.set_title(title, fontsize=12.5, color=INK_PRIMARY, pad=14, loc="left")

    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0, colors=INK_PRIMARY)

    # Outside the axes. Inside (lower right) it sat on top of the shortest bar's value
    # label -- the validator checks colour, not collisions, so this one comes from looking.
    ax.legend(
        handles=[Patch(facecolor=SERIES_RL, label="RL-trained (this work)"),
                 Patch(facecolor=SERIES_ZS, label="zero-shot baseline")],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Judge accuracy on the frozen 880-pair held-out set "
                                       "(fakes from Qwen3.5-9B SFT ep3, thinking on)")
    a = ap.parse_args()
    render(load(Path(a.csv)), Path(a.out), a.title)


if __name__ == "__main__":
    main()
