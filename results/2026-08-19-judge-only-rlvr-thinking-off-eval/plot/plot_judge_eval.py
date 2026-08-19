"""Render the judge-accuracy bar plot from the retained summary CSV."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

SERIES_RL = "#2a78d6"
SERIES_ZS = "#eb6834"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

DISPLAY = {
    "judge-9b-graded-step52": "Qwen3.5-9B judge — Brier RL",
    "judge-9b-directional-step52": "Qwen3.5-9B judge — 0/1 RL",
    "judge-4b-graded-step52": "Qwen3.5-4B judge — Brier RL",
    "judge-4b-directional-step52": "Qwen3.5-4B judge — 0/1 RL",
    "qwen35-27b": "Qwen3.5-27B",
    "gemma4-31b": "Gemma-4-31B",
    "gemma4-12b": "Gemma-4-12B",
    "qwen35-9b": "Qwen3.5-9B",
    "qwen35-4b": "Qwen3.5-4B",
}


def load(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["accuracy"] = float(row["accuracy"])
        row["se"] = float(row["se"])
    return sorted(rows, key=lambda row: float(row["accuracy"]))


def render(rows: list[dict[str, object]], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.62
    radius = height / 2.6

    for i, row in enumerate(rows):
        accuracy = float(row["accuracy"])
        se = float(row["se"])
        colour = SERIES_RL if row["kind"] == "rl-trained" else SERIES_ZS
        ax.add_patch(
            FancyBboxPatch(
                (0, i - height / 2), accuracy, height,
                boxstyle=f"round,pad=0,rounding_size={radius}",
                mutation_aspect=0.02, linewidth=0, facecolor=colour,
                zorder=3, clip_on=False,
            )
        )
        ax.errorbar(
            accuracy, i, xerr=1.96 * se, fmt="none", ecolor=INK_SECONDARY,
            elinewidth=1.2, capsize=3, zorder=4,
        )
        ax.text(
            accuracy + 1.96 * se + 0.012, i, f"{accuracy:.3f}",
            va="center", ha="left", fontsize=9.5, color=INK_PRIMARY, zorder=5,
        )

    ax.axvline(0.5, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(0.5, -0.62, "  chance (0.50)", color=INK_MUTED, fontsize=9,
            va="center", ha="left")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([DISPLAY.get(str(row["model"]), str(row["model"])) for row in rows],
                       fontsize=10)
    ax.set_xlim(0, 0.82)
    ax.set_ylim(-1.0, len(rows) - 0.35)
    ax.set_xlabel("accuracy on 880 held-out pairs (95% CI)", fontsize=10,
                  color=INK_SECONDARY)
    ax.set_title(
        "Judge accuracy on the frozen 880-pair held-out set (thinking OFF)",
        fontsize=12.5, color=INK_PRIMARY, pad=14, loc="left",
    )
    ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(axis="x", colors=INK_MUTED, labelsize=9)
    ax.tick_params(axis="y", length=0, colors=INK_PRIMARY)
    ax.legend(
        handles=[Patch(facecolor=SERIES_RL, label="RL-trained"),
                 Patch(facecolor=SERIES_ZS, label="zero-shot baseline")],
        loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
        frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    render(load(Path(args.csv)), Path(args.out))


if __name__ == "__main__":
    main()
