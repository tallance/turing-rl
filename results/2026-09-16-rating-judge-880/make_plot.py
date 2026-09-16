"""Judge plots on the frozen 880-pair held-out set, from comparison.csv.

Copied from results/2026-09-01-gemma-single-token-judge/make_plot.py and extended with the
rating_only thinking-ON judge (J1', GRPO step 52) and its zero-shot control.

One structural change against the original: GROUPS maps a series key to a cell name, so a
group can carry any number of bars. The original hard-coded three (thinking OFF / ON / single
token) and centred them with a fixed ``offset - 1``; the rating groups have one, so the offset
is computed from how many bars a group actually has.

All 28 cells share one pair set and one decode policy (each model's generation_config.json
defaults, no wire override), so the prompt is the only judge-side variable on the figure.
Earlier runs of these two cells at other decode settings are kept in the sibling CSVs; see
README.txt.

Error bars are a normal approximation, 1.96*sqrt(p(1-p)/n); with the half-tie rule the
outcome is not strictly Bernoulli, so read them as indicative.

  python make_plot.py
"""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = list(csv.DictReader((HERE / "comparison.csv").open()))


def cell(name, style, think, column):
    for r in rows:
        if r["cell"] == name and r["prompt_style"] == style and r["thinking_mode"] == think:
            value = r[column]
            return (float(value) if value else None), int(r["n_pairs"])
    return None, None


# Series key -> (prompt_style, thinking_mode) used to find the row.
SERIES_SPEC = {
    "off": ("full", "off"),
    "on": ("full", "on"),
    "st": ("single_token", "off"),
    "r": ("rating_only", "on"),
}
# Draw order within a group, bottom to top.
FULL_SET = ["off", "on", "st"]
RATING_SET = ["r"]

# Bottom-to-top: zero-shot by model size, then the trained judges.
# (row label, {series key: cell name}, arm, draw order)
GROUPS = [
    ("Qwen3.5-4B",  {"off": "qwen35-4b",  "on": "qwen35-4b",  "st": "qwen35-4b-st"},  "zero", FULL_SET),
    ("Qwen3.5-9B",  {"off": "qwen35-9b",  "on": "qwen35-9b",  "st": "qwen35-9b-st"},  "zero", FULL_SET),
    ("Gemma-4-12B", {"off": "gemma4-12b", "on": "gemma4-12b", "st": "gemma4-12b-st"}, "zero", FULL_SET),
    ("Qwen3.5-27B", {"off": "qwen35-27b", "on": "qwen35-27b", "st": "qwen35-27b-st"}, "zero", FULL_SET),
    ("Gemma-4-31B", {"off": "gemma4-31b", "on": "gemma4-31b", "st": "gemma4-31b-st"}, "zero", FULL_SET),
    ("Qwen3.5-9B\nrating-only", {"r": "judge-9b-rating-zeroshot"}, "zero", RATING_SET),
    ("Qwen3.5-4B judge", {"off": "judge-4b-graded-step52", "on": "judge-4b-graded-step52",
                          "st": "judge-4b-ce-st"}, "trained", FULL_SET),
    ("Qwen3.5-9B judge", {"off": "judge-9b-graded-step52", "on": "judge-9b-graded-step52",
                          "st": "judge-9b-ce-st"}, "trained", FULL_SET),
    # Single-token only: there is no trained full-schema Gemma judge.
    ("Gemma-4-12B judge", {"st": "judge-gemma12b-ce-st"}, "trained", ["st"]),
    ("Qwen3.5-9B judge\nrating-only", {"r": "judge-9b-rating-trained"}, "trained", RATING_SET),
]

# Warm = trained, cold = zero-shot. Rating-only keeps its own hue so the newest protocol is
# distinguishable from the three it is being compared against.
COLOURS = {
    ("zero", "off"): "#1f77b4", ("zero", "on"): "#7d5bd0", ("zero", "st"): "#12a1a1",
    ("zero", "r"): "#2e7d32",
    ("trained", "off"): "#d1402f", ("trained", "on"): "#f2952e", ("trained", "st"): "#8e1b4d",
    ("trained", "r"): "#5d4037",
}
LEGEND = [
    ("trained", "off", "trained — full schema, thinking OFF"),
    ("trained", "on",  "trained — full schema, thinking ON"),
    ("trained", "st",  "trained — single token"),
    ("trained", "r",   "trained — rating only, thinking ON"),
    ("zero", "off", "zero-shot — full schema, thinking OFF"),
    ("zero", "on",  "zero-shot — full schema, thinking ON"),
    ("zero", "st",  "zero-shot — single token"),
    ("zero", "r",   "zero-shot — rating only, thinking ON"),
]

# Between the last zero-shot group and the first trained one.
DIVIDER_AFTER = 5

FOOTNOTE = (
    "All 28 cells share one pair set (gen_9b-full5ep-step0_880.parquet; its AI turns were "
    "sampled at T=0.7) and one decode policy (each model's generation_config.json defaults, no "
    "wire override). Training data differs: the trained rating-only judge saw a mixed-"
    "temperature corpus (2 of 4 fakes per context at T=1.0), every other trained judge saw "
    "T=0.7 only."
)

FIGURES = [
    {
        "column": "accuracy_half_tie",
        "path": "judge_accuracy_880.png",
        "title": "Judge accuracy on the frozen 880-pair held-out set",
        "xlabel": "accuracy on the held-out pairs (95% CI)   —   ties count half, failures wrong",
        "xmax": 0.88,
        "refline": (0.5, "chance (0.50)"),
    },
    {
        "column": "a_rate_half_tie",
        "path": "judge_a_rate_880.png",
        "title": "How often the judge answers A — frozen 880-pair held-out set",
        "xlabel": "share of verdicts that say A (95% CI)   —   a tie counts as half an A",
        "xmax": 1.0,
        "refline": (0.5, "no position bias (0.50)"),
    },
]


def draw(spec):
    fig, ax = plt.subplots(figsize=(13, 11.5))
    height, gap = 0.24, 1.15
    yticks, ylabels = [], []

    for index, (label, cells, arm, order) in enumerate(GROUPS):
        centre = index * gap
        yticks.append(centre)
        ylabels.append(label)
        # Centre however many bars this group actually has, rather than assuming three.
        count = len(order)
        for position, key in enumerate(order):
            name = cells.get(key)
            if name is None:
                continue
            style, think = SERIES_SPEC[key]
            value, n = cell(name, style, think, spec["column"])
            if value is None:
                continue
            y = centre + (position - (count - 1) / 2) * height
            err = 1.96 * math.sqrt(value * (1 - value) / n)
            ax.barh(y, value, height=height, color=COLOURS[(arm, key)],
                    edgecolor="none", zorder=3)
            ax.errorbar(value, y, xerr=err, fmt="none", ecolor="#3a3a3a",
                        elinewidth=1.1, capsize=3, zorder=4)
            ax.text(value + err + 0.008, y, f"{value:.3f}", va="center", fontsize=9.5, zorder=5)

    refx, reftext = spec["refline"]
    ax.axvline(refx, color="#8a8a8a", linestyle="--", linewidth=1.3, zorder=2)
    ax.text(refx, -0.85, reftext, color="#8a8a8a", fontsize=10, ha="center")
    ax.axhline((DIVIDER_AFTER * gap + (DIVIDER_AFTER + 1) * gap) / 2,
               color="#cccccc", linewidth=1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=11)
    ax.set_xlim(0.0, spec["xmax"])
    ax.set_ylim(-1.05, (len(GROUPS) - 1) * gap + 0.5)
    ax.set_xlabel(spec["xlabel"], fontsize=11)
    ax.set_title(spec["title"], fontsize=15, pad=14)
    ax.grid(axis="x", color="#e3e3e3", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOURS[(arm, key)]) for arm, key, _ in LEGEND]
    ax.legend(handles, [text for _, _, text in LEGEND], loc="upper center",
              bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False, fontsize=10)
    fig.text(0.5, -0.045, FOOTNOTE, ha="center", fontsize=8.5, color="#6a6a6a", wrap=True)

    fig.tight_layout()
    path = HERE / spec["path"]
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


for figure in FIGURES:
    draw(figure)
