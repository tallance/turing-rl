#!/usr/bin/env python3
"""Validate frozen summaries and rebuild the Qwen-0.8B-reward trajectory plot."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import INK, apply_rc, style_axes  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
STEPS = list(range(0, 121, 12))
STEP_RE = re.compile(r"step(\d+)$")
MAX_PARSE_ERROR_FRAC = 0.02
OUT_NAME = "test_eval_judges_train10pct_20ep_qwen08b_reward_test50pct_full_schema.png"
JUDGES = [
    ("qwen35-0.8b", "Qwen 0.8B judge (trained against; thinking OFF)", "#ef6c32", "-", "o", 3.2),
    ("gemma4-12b", "Gemma 4 12B judge (thinking ON)", INK["secondary"], "--", "D", 2.6),
    ("gemma4-31b", "Gemma 4 31B judge (thinking ON)", "#8b5fbf", "-", "s", 2.6),
    ("qwen35-9b", "Qwen 9B judge (thinking ON)", "#317bd2", "-", "^", 2.6),
]


def read_summary(cell: str) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with (ROOT / f"summary_{cell}.csv").open(newline="") as handle:
        for raw in csv.DictReader(handle):
            match = STEP_RE.search(raw["checkpoint"])
            if not match:
                raise SystemExit(f"cannot parse checkpoint {raw['checkpoint']!r}")
            rows.append(
                {
                    "step": int(match.group(1)),
                    "n_scored": int(raw["n_scored"]),
                    "n_unique_pairs": int(raw["n_unique_pairs"]),
                    "n_parse_error": int(raw["n_parse_error"]),
                    "likert_mean": float(raw["likert_mean"]),
                    "win_rate_ge5": float(raw["win_rate_ge5"]),
                }
            )
    if [row["step"] for row in rows] != STEPS:
        raise SystemExit(f"unexpected checkpoint sequence for {cell}")
    if any(row["n_scored"] != 440 or row["n_unique_pairs"] != 440 for row in rows):
        raise SystemExit(f"incomplete 440-pair cell for {cell}")
    if any(row["n_parse_error"] / row["n_scored"] >= MAX_PARSE_ERROR_FRAC for row in rows):
        raise SystemExit(f"parse-error threshold exceeded for {cell}")
    return rows


def write_plot(data: dict[str, list[dict[str, float | int]]]) -> Path:
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.4))
    panels = [
        ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
        ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
    ]
    for ax, (field, title, ylabel, reference, note) in zip(axes, panels):
        ax.axhline(reference, color=INK["muted"], lw=1.3, ls=(0, (4, 3)), zorder=1)
        for cell, label, color, linestyle, marker, width in JUDGES:
            rows = data[cell]
            ax.plot(
                [row["step"] for row in rows],
                [row[field] for row in rows],
                color=color,
                linestyle=linestyle,
                marker=marker,
                lw=width,
                markersize=8,
                markeredgecolor=INK["surface"],
                markeredgewidth=1.4,
                label=label,
                zorder=4,
            )
        style_axes(ax, title, "GRPO step", ylabel, STEPS)
        ax.annotate(
            note,
            xy=(STEPS[-1], reference),
            xytext=(-4, 7),
            textcoords="offset points",
            ha="right",
            color=INK["muted"],
            fontsize=9,
        )
    axes[0].set_ylim(1.0, 4.75)
    axes[1].set_ylim(0.0, 0.56)
    fig.suptitle(
        "9B generator trained with Qwen 3.5 0.8B reward, 10%-dataset 20-epoch run",
        x=0.025,
        y=0.985,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.025,
        0.915,
        (
            "440 pairs per checkpoint on the same frozen 50% held-out subset. "
            "The Qwen 0.8B judge uses thinking off; both Gemma judges and Qwen 9B use thinking on."
        ),
        ha="left",
        color=INK["muted"],
        fontsize=10.2,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
        fontsize=10.2,
    )
    fig.tight_layout(rect=(0.015, 0.12, 0.995, 0.865))
    out = ROOT / OUT_NAME
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    return out


def write_validation(data: dict[str, list[dict[str, float | int]]]) -> None:
    lines = [
        "Source completeness: PASS, 44/44 cells with exactly 440 unique pair keys.\n",
        "Combined summaries: PASS, 4 judges x 11 checkpoints.\n",
        "Parse-error threshold: PASS, every cell is below 2%; metrics exclude invalid Likert rows.\n",
    ]
    for cell, *_rest in JUDGES:
        rows = data[cell]
        total = sum(int(row["n_scored"]) for row in rows)
        errors = sum(int(row["n_parse_error"]) for row in rows)
        worst = max(int(row["n_parse_error"]) for row in rows)
        lines.append(
            f"  {cell}: {errors}/{total} parse failures "
            f"({100 * errors / total:.2f}% overall; worst cell {worst}/440 = {100 * worst / 440:.2f}%).\n"
        )
    (ROOT / "validation.txt").write_text("".join(lines))


def main() -> None:
    data = {cell: read_summary(cell) for cell, *_rest in JUDGES}
    out = write_plot(data)
    write_validation(data)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
