"""Overlay the sparse trained-Brier-judge trajectory on the finalized five-judge curves."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STEP_RE = re.compile(r"step(\d+)$")
BRIER_STEPS = list(range(0, 321, 32))
SERIES = [
    ("qwen35-4b", "4B judge", "#2f7ed8", "-", "o", 2.0),
    ("qwen35-9b", "9B judge (trained against)", "#f26b32", "-", "o", 3.2),
    ("qwen35-27b", "27B judge", "#19ad7b", "-", "o", 2.0),
    ("gemma4-12b", "Gemma 4 12B judge", "#5b5b57", "--", "D", 2.0),
    ("gemma4-31b", "Gemma 4 31B judge", "#8b5fbf", "-", "s", 2.0),
]


def read_summary(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = STEP_RE.search(row["checkpoint"])
            if not match:
                raise SystemExit(f"FAIL: cannot parse step from {row['checkpoint']!r} in {path}")
            rows.append(
                {
                    "step": int(match.group(1)),
                    "n_scored": int(row["n_scored"]),
                    "likert_mean": float(row["likert_mean"]),
                    "win_rate_ge5": float(row["win_rate_ge5"]),
                }
            )
    rows.sort(key=lambda row: int(row["step"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval-root", required=True)
    parser.add_argument("--brier-summary", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--title",
        default="9B GRPO generator, full-dataset 5-epoch run, on the held-out test set",
    )
    args = parser.parse_args()

    base = Path(args.base_eval_root)
    dense = {
        cell: read_summary(base / f"summary_{cell}.csv")
        for cell, _label, _color, _line, _marker, _width in SERIES
    }
    brier = read_summary(Path(args.brier_summary))
    brier_steps = [int(row["step"]) for row in brier]
    if brier_steps != BRIER_STEPS:
        raise SystemExit(f"FAIL: expected Brier steps {BRIER_STEPS}, got {brier_steps}")

    dense_counts = {int(row["n_scored"]) for rows in dense.values() for row in rows}
    if len(dense_counts) != 1:
        raise SystemExit(f"FAIL: finalized judge summaries use different pair counts: {dense_counts}")
    brier_counts = {int(row["n_scored"]) for row in brier}
    if len(brier_counts) != 1:
        raise SystemExit(f"FAIL: Brier checkpoints use different pair counts: {brier_counts}")
    dense_n = dense_counts.pop()
    brier_n = brier_counts.pop()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#b9b7af",
            "axes.labelcolor": "#56544e",
            "xtick.color": "#76736b",
            "ytick.color": "#76736b",
            "grid.color": "#dedbd3",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    panels = [
        ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
        ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
    ]

    for axis, (field, heading, ylabel, reference, note) in zip(axes, panels):
        axis.axhline(reference, color="#8b887f", lw=1.1, ls=(0, (4, 3)), zorder=1)
        for cell, label, color, line, marker, width in SERIES:
            rows = dense[cell]
            axis.plot(
                [row["step"] for row in rows],
                [row[field] for row in rows],
                label=label,
                color=color,
                linestyle=line,
                marker=marker,
                lw=width,
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                zorder=3,
            )
        axis.plot(
            [row["step"] for row in brier],
            [row[field] for row in brier],
            label="9B Brier RL judge",
            color="#c51b7d",
            linestyle=(0, (5, 2)),
            marker="^",
            lw=3.3,
            markersize=9,
            markeredgecolor="white",
            markeredgewidth=1.5,
            zorder=5,
        )
        axis.set_title(heading, loc="left", fontsize=12, fontweight="bold")
        axis.set_xlabel("GRPO step")
        axis.set_ylabel(ylabel)
        axis.set_xticks(list(range(0, 321, 32)))
        axis.grid(axis="y", linewidth=0.8)
        axis.annotate(
            note,
            xy=(320, reference),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color="#8b887f",
            fontsize=8.5,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle(args.title, x=0.02, ha="left", fontsize=14, fontweight="bold", y=0.99)
    fig.text(
        0.02,
        0.925,
        f"{dense_n} pairs per zero-shot checkpoint; trained Brier judge scores every checkpoint "
        f"on its {brier_n}-pair common subset.",
        ha="left",
        va="top",
        color="#76736b",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.86))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
