"""Plot zero-shot and trained Qwen 9B judges on one generator trajectory."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


STEP_RE = re.compile(r"step(\d+)$")
EXPECTED_STEPS = list(range(0, 321, 32))


def read_summary(path: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = STEP_RE.search(row["checkpoint"])
            if not match:
                raise SystemExit(f"FAIL: cannot parse checkpoint in {path}: {row['checkpoint']!r}")
            rows.append(
                {
                    "step": int(match.group(1)),
                    "n_scored": int(row["n_scored"]),
                    "likert_mean": float(row["likert_mean"]),
                    "win_rate_ge5": float(row["win_rate_ge5"]),
                }
            )
    rows.sort(key=lambda row: int(row["step"]))
    if [int(row["step"]) for row in rows] != EXPECTED_STEPS:
        raise SystemExit(f"FAIL: expected steps {EXPECTED_STEPS} in {path}")
    if {int(row["n_scored"]) for row in rows} != {880}:
        raise SystemExit(f"FAIL: expected 880 scored pairs at every checkpoint in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regular-summary", required=True)
    parser.add_argument("--brier-off-summary", required=True)
    parser.add_argument("--brier-on-summary")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    series = [
        (
            read_summary(Path(args.regular_summary)),
            "Regular Qwen 9B judge (thinking ON)",
            "#f26b32",
            "o",
            "-",
        ),
        (
            read_summary(Path(args.brier_off_summary)),
            "Brier-trained Qwen 9B judge (train OFF, eval OFF)",
            "#8b5fbf",
            "^",
            "--",
        ),
    ]
    if args.brier_on_summary:
        series.append(
            (
                read_summary(Path(args.brier_on_summary)),
                "Brier-trained Qwen 9B judge (train ON, eval ON)",
                "#19ad7b",
                "s",
                "-.",
            )
        )

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
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.1))
    panels = [
        ("likert_mean", "Mean judge rating", "Likert 1–7", 4.0, "4 = cannot tell"),
        ("win_rate_ge5", "Generator win rate", "fraction rated ≥ 5", 0.5, "0.5 = parity"),
    ]
    for axis, (field, title, ylabel, reference, note) in zip(axes, panels):
        axis.axhline(reference, color="#8b887f", lw=1.2, ls=(0, (4, 3)), zorder=1)
        for rows, label, color, marker, linestyle in series:
            axis.plot(
                [row["step"] for row in rows],
                [row[field] for row in rows],
                label=label,
                color=color,
                marker=marker,
                linestyle=linestyle,
                lw=3.0,
                markersize=8,
                markeredgecolor="white",
                markeredgewidth=1.4,
                zorder=3,
            )
        axis.set_title(title, loc="left", fontsize=14, fontweight="bold")
        axis.set_xlabel("GRPO step", fontsize=11)
        axis.set_ylabel(ylabel, fontsize=11)
        axis.set_xticks(EXPECTED_STEPS)
        axis.grid(axis="y", linewidth=0.8)
        axis.annotate(
            note,
            xy=(320, reference),
            xytext=(0, 5),
            textcoords="offset points",
            ha="right",
            color="#8b887f",
            fontsize=9,
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(series), frameon=False, fontsize=10)
    fig.suptitle(
        "9B judges on the full-dataset 5-epoch GRPO trajectory",
        x=0.02,
        ha="left",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    modes = (
        "Regular judge: zero-shot, thinking ON. Brier judge: trained thinking OFF and evaluated "
        "thinking OFF."
    )
    if args.brier_on_summary:
        modes += " Additional Brier judge: trained thinking ON and evaluated thinking ON."
    fig.text(
        0.02,
        0.93,
        f"Same frozen 880 held-out pairs per checkpoint. {modes}",
        ha="left",
        va="top",
        color="#76736b",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.86))
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
