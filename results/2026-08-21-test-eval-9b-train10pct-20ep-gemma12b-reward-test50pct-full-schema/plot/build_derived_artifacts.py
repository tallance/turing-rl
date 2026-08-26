#!/usr/bin/env python3
"""Validate the frozen summaries and rebuild tables and the final plot."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plotstyle import INK, apply_rc, style_axes  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
STEPS = list(range(0, 121, 12))
MAX_PARSE_ERROR_FRAC = 0.02
JUDGES = [
    ("gemma4-12b", "Gemma 4 12B judge (trained against)", INK["secondary"], "--", "D"),
    ("qwen35-9b", "Qwen 9B judge", "#ef6c32", "-", "o"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_summaries() -> dict[str, list[dict[str, float | int]]]:
    parsed: dict[str, list[dict[str, float | int]]] = {}
    for cell, _label, _color, _linestyle, _marker in JUDGES:
        rows = read_csv(ROOT / f"summary_{cell}.csv")
        values = []
        for row in rows:
            values.append(
                {
                    "step": int(row["checkpoint"].rsplit("step", 1)[1]),
                    "n_scored": int(row["n_scored"]),
                    "n_unique_pairs": int(row["n_unique_pairs"]),
                    "n_likert": int(row["n_likert"]),
                    "n_parse_error": int(row["n_parse_error"]),
                    "likert_mean": float(row["likert_mean"]),
                    "win_rate_ge5": float(row["win_rate_ge5"]),
                }
            )
        if [row["step"] for row in values] != STEPS:
            raise SystemExit(f"unexpected checkpoint sequence for {cell}")
        if any(row["n_scored"] != 440 or row["n_unique_pairs"] != 440 for row in values):
            raise SystemExit(f"incomplete summary row for {cell}")
        if any(row["n_likert"] + row["n_parse_error"] != row["n_scored"] for row in values):
            raise SystemExit(f"inconsistent parse accounting for {cell}")
        if any(row["n_parse_error"] / row["n_scored"] >= MAX_PARSE_ERROR_FRAC for row in values):
            raise SystemExit(f"parse-error threshold exceeded for {cell}")
        parsed[cell] = values
    return parsed


def write_job_table() -> None:
    rows = read_csv(PROVENANCE / "pipeline_jobs.csv")
    retained = []
    for row in rows:
        if row["stage"] == "judge":
            if row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
                raise SystemExit(f"non-completed judge row: {row}")
            retained.append(row)
        elif row["stage"] == "reuse":
            if row["state"] != "REUSED" or row["exit_code"] != "0:0":
                raise SystemExit(f"invalid reuse row: {row}")
            retained.append(row)
    expected = {(judge[0], step) for judge in JUDGES for step in STEPS}
    actual = {(row["judge"], int(row["checkpoint"])) for row in retained}
    if actual != expected or len(retained) != 22:
        raise SystemExit(f"judge table mismatch: missing={expected-actual}, extra={actual-expected}")
    order = {cell: index for index, (cell, *_rest) in enumerate(JUDGES)}
    retained.sort(key=lambda row: (order[row["judge"]], int(row["checkpoint"])))
    columns = [
        "judge",
        "checkpoint",
        "job_id",
        "source_job_id",
        "state",
        "exit_code",
        "gpus",
        "active_seconds",
        "model_startup_seconds",
        "scoring_seconds",
        "submit",
        "start",
        "end",
    ]
    with (ROOT / "judge_jobs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in retained)


def validate_sources() -> None:
    validation = (PROVENANCE / "validation.txt").read_text()
    if "PASS: 22 cell(s)" not in validation:
        raise SystemExit("source completeness validation does not record 22 passing cells")
    timing = json.loads((PROVENANCE / "timing_summary.json").read_text())
    if timing.get("completed_job_count") != 80 or timing.get("reused_cell_count") != 2:
        raise SystemExit("unexpected pipeline timing job counts")
    pair_rows = read_csv(PROVENANCE / "pair_sha256.csv")
    if [int(row["checkpoint"]) for row in pair_rows] != STEPS:
        raise SystemExit("pair hash manifest has an unexpected checkpoint sequence")
    merge_rows = read_csv(PROVENANCE / "merge_provenance.csv")
    if [int(row["step"]) for row in merge_rows] != STEPS[1:]:
        raise SystemExit("merge provenance has an unexpected checkpoint sequence")
    if any(int(row["n_shards"]) != 8 for row in merge_rows):
        raise SystemExit("merge provenance records an incomplete actor checkpoint")


def write_plot(data: dict[str, list[dict[str, float | int]]]) -> None:
    apply_rc()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))
    panels = [
        ("likert_mean", "Mean judge rating", "Likert 1-7", 4.0, "4 = cannot tell"),
        ("win_rate_ge5", "Generator win rate", "fraction rated >= 5", 0.5, "0.5 = parity"),
    ]
    for ax, (field, title, ylabel, reference, note) in zip(axes, panels):
        ax.axhline(reference, color=INK["muted"], lw=1.3, ls=(0, (4, 3)), zorder=1)
        for cell, label, color, linestyle, marker in JUDGES:
            rows = data[cell]
            ax.plot(
                [row["step"] for row in rows],
                [row[field] for row in rows],
                color=color,
                linestyle=linestyle,
                marker=marker,
                lw=3.0,
                markersize=8,
                markeredgecolor=INK["surface"],
                markeredgewidth=1.4,
                label=label,
                zorder=3,
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
    axes[0].set_ylim(3.7, 5.05)
    axes[1].set_ylim(0.32, 0.9)
    fig.suptitle(
        "9B generator trained with Gemma 4 12B reward, 10%-dataset 20-epoch run",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.03,
        0.91,
        (
            "440 pairs per checkpoint on the same frozen 50% held-out subset. "
            "Both judges use thinking on and the corrected full ordered schema."
        ),
        ha="left",
        color=INK["muted"],
        fontsize=10.5,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.09, 0.99, 0.87))
    out = ROOT / "test_eval_judges_train10pct_20ep_gemma12b_reward_test50pct_full_schema.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


def main() -> None:
    validate_sources()
    data = validate_summaries()
    shutil.copy2(PROVENANCE / "pipeline_jobs.csv", ROOT / "pipeline_jobs.csv")
    shutil.copy2(PROVENANCE / "timing_summary.json", ROOT / "timing_summary.json")
    write_job_table()
    write_plot(data)
    parse_lines = []
    for cell, *_rest in JUDGES:
        total_rows = sum(row["n_scored"] for row in data[cell])
        total_errors = sum(row["n_parse_error"] for row in data[cell])
        max_errors = max(row["n_parse_error"] for row in data[cell])
        parse_lines.append(
            f"  {cell}: {total_errors}/{total_rows} parse failures "
            f"({100 * total_errors / total_rows:.2f}% overall; "
            f"worst cell {max_errors}/440 = {100 * max_errors / 440:.2f}%).\n"
        )
    (ROOT / "validation.txt").write_text(
        "Source completeness: PASS, 22/22 cells with exactly 440 unique pair keys.\n"
        "Combined summaries: PASS, 2 judges x 11 checkpoints, 440 judge rows per cell.\n"
        "Parse-error threshold: PASS, every cell is below 2%; metrics exclude rows "
        "without a valid Likert score.\n"
        + "".join(parse_lines)
        + "Retained judge cells: PASS, 20 completed jobs plus 2 checksum-verified reused cells.\n"
        + "Pipeline accounting: PASS, 80 completed jobs and 2 reused cells.\n"
    )


if __name__ == "__main__":
    main()
