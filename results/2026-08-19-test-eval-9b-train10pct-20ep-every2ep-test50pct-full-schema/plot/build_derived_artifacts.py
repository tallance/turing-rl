#!/usr/bin/env python3
"""Build the combined 20-epoch summaries, job table, validation, and plot."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = ROOT / "provenance"
CELLS = ["qwen35-9b", "gemma4-12b", "gemma4-31b", "qwen35-4b", "qwen35-27b"]
BASE_STEPS = list(range(0, 61, 6))
EXTENSION_STEPS = [72, 84, 96, 108, 120]
EXPECTED_STEPS = BASE_STEPS + EXTENSION_STEPS
JOB_RE = re.compile(r"^te_t20t50_(.+)_(\d+)$")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def step_of(row: dict[str, str]) -> int:
    return int(row["checkpoint"].rsplit("step", 1)[1])


def write_summary(cell: str) -> None:
    rows = read_csv(PROVENANCE / f"base_summary_{cell}.csv")
    rows += read_csv(PROVENANCE / f"extension_summary_{cell}.csv")
    by_step = {step_of(row): row for row in rows}
    if len(by_step) != len(rows):
        raise SystemExit(f"duplicate checkpoint in {cell} source summaries")
    if sorted(by_step) != EXPECTED_STEPS:
        raise SystemExit(f"unexpected steps for {cell}: {sorted(by_step)}")
    ordered = [by_step[step] for step in EXPECTED_STEPS]
    for row in ordered:
        if int(row["n_scored"]) != 440 or int(row["n_unique_pairs"]) != 440:
            raise SystemExit(f"incomplete summary row for {cell}: {row}")

    columns = list(ordered[0])
    out_csv = ROOT / f"summary_{cell}.csv"
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)

    guard = json.loads((PROVENANCE / "base_split_guard.json").read_text())
    split_note = (
        f"# split: {guard['verdict']} expect={guard['expect']} rows={guard['eval_rows']} "
        f"users={guard['eval_users']} parquet={guard['eval_parquet']}"
    )
    header = f"| {' | '.join(columns)} |\n|{'|'.join(['---'] * len(columns))}|"
    body = "\n".join(
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for row in ordered
    )
    (ROOT / f"summary_{cell}.md").write_text(f"{split_note}\n\n{header}\n{body}\n")


def write_job_table() -> None:
    base_rows = read_csv(PROVENANCE / "base_judge_jobs.csv")
    extension_rows: dict[tuple[str, int], dict[str, str]] = {}
    with (PROVENANCE / "qwen20ep_extension_sacct.psv").open() as handle:
        for line in handle:
            job_id, job_name, state, elapsed, start, end, exit_code = line.rstrip("\n").split("|")
            match = JOB_RE.match(job_name)
            if not match or state != "COMPLETED" or exit_code != "0:0":
                continue
            cell, raw_step = match.groups()
            if cell not in CELLS:
                continue
            key = (cell, int(raw_step))
            if key in extension_rows:
                raise SystemExit(f"duplicate completed extension job for {key}")
            extension_rows[key] = {
                "judge": cell,
                "step": raw_step,
                "job_id": job_id,
                "job_name": job_name,
                "state": state,
                "elapsed": elapsed,
                "start": start,
                "end": end,
                "exit_code": exit_code,
            }
    expected = {(cell, step) for cell in CELLS for step in EXTENSION_STEPS}
    if set(extension_rows) != expected:
        raise SystemExit(
            f"extension judge jobs differ from expected: missing={sorted(expected-set(extension_rows))} "
            f"extra={sorted(set(extension_rows)-expected)}"
        )
    rows = base_rows + list(extension_rows.values())
    order = {cell: index for index, cell in enumerate(CELLS)}
    rows.sort(key=lambda row: (order[row["judge"]], int(row["step"])))
    if len(rows) != 80:
        raise SystemExit(f"expected 80 retained judge jobs, found {len(rows)}")
    with (ROOT / "judge_jobs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_validation() -> None:
    if "PASS: 55 cell(s)" not in (PROVENANCE / "base_validation.txt").read_text():
        raise SystemExit("base validation does not record 55 passing cells")
    if "PASS: 25 cell(s)" not in (
        PROVENANCE / "qwen20ep_extension_validation.txt"
    ).read_text():
        raise SystemExit("extension validation does not record 25 passing cells")
    for cell in CELLS:
        rows = read_csv(ROOT / f"summary_{cell}.csv")
        if len(rows) != 16 or [step_of(row) for row in rows] != EXPECTED_STEPS:
            raise SystemExit(f"combined summary validation failed for {cell}")
        if any(int(row["n_scored"]) != 440 for row in rows):
            raise SystemExit(f"combined summary has incomplete rows for {cell}")
    jobs = read_csv(ROOT / "judge_jobs.csv")
    if len(jobs) != 80 or any(
        row["state"] != "COMPLETED" or row["exit_code"] != "0:0" for row in jobs
    ):
        raise SystemExit("combined retained job table is incomplete")
    (ROOT / "validation.txt").write_text(
        "Base source: PASS, 55/55 cells with exactly 440 unique pair keys.\n"
        "Extension source: PASS, 25/25 cells with exactly 440 unique pair keys.\n"
        "Combined summaries: PASS, 5 judges x 16 checkpoints, 440 scored pairs per row.\n"
        "Retained judge jobs: PASS, 80/80 COMPLETED with exit code 0:0.\n"
    )


def write_plot() -> None:
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "plot" / "plot_test_eval_judges.py"),
            "--eval_root",
            str(ROOT),
            "--out_dir",
            str(ROOT),
            "--stem",
            "test_eval_judges_train10pct_20ep_test50pct_full_schema",
            "--title",
            "9B GRPO generator, 10%-dataset run extended to 20 epochs",
            "--subtitle",
            (
                "{n} pairs per checkpoint on the same frozen 50% held-out subset. "
                "All five judges score every checkpoint with the corrected full ordered "
                "schema.  * = judge the run was trained against."
            ),
        ],
        check=True,
    )


def main() -> None:
    if (PROVENANCE / "base_split_guard.json").read_bytes() != (
        PROVENANCE / "qwen20ep_split_guard.json"
    ).read_bytes():
        raise SystemExit("base and extension split guards differ")
    shutil.copy2(PROVENANCE / "base_split_guard.json", ROOT / "split_guard.json")
    for cell in CELLS:
        write_summary(cell)
    write_job_table()
    write_validation()
    write_plot()


if __name__ == "__main__":
    main()
