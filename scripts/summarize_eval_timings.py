#!/usr/bin/env python3
"""Summarize Slurm and judge-phase timings for an evaluation pipeline."""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable


JUDGE_RE = re.compile(
    r"_(gemma4-12b|gemma4-31b|qwen35-9b|qwen35-4b|qwen35-27b)_(\d+)$"
)
STAGE_RE = re.compile(r"_(merge|gen|build)_(\d+)$")


def parse_time(value: str) -> datetime | None:
    if not value or value in {"Unknown", "None", "N/A"}:
        return None
    return datetime.fromisoformat(value.rstrip("Z"))


def seconds_between(start: str, end: str) -> float | None:
    start_time = parse_time(start)
    end_time = parse_time(end)
    if start_time is None or end_time is None:
        return None
    return round((end_time - start_time).total_seconds(), 3)


def parse_gpus(tres: str) -> int:
    for item in tres.split(","):
        if item.startswith("gres/gpu="):
            return int(item.split("=", 1)[1])
        if item.startswith("gres/gpu:") and "=" in item:
            return int(item.rsplit("=", 1)[1])
    return 0


def classify_job(name: str) -> tuple[str, str, str]:
    match = JUDGE_RE.search(name)
    if match:
        return "judge", match.group(2), match.group(1)
    match = STAGE_RE.search(name)
    if match:
        stage = {"gen": "generation", "build": "pair_build"}.get(match.group(1), match.group(1))
        return stage, match.group(2), ""
    if name.endswith("_continue"):
        return "continuation", "", ""
    if name.endswith("_prepare"):
        return "prepare", "", ""
    return "other", "", ""


def load_judge_timings(eval_root: Path) -> dict[str, dict[str, object]]:
    timings: dict[str, dict[str, object]] = {}
    for path in sorted(eval_root.glob("raw/*/sweep/*/*/timing.json")):
        data = json.loads(path.read_text())
        job_id = str(data.get("slurm_job_id") or "")
        if not job_id:
            raise ValueError(f"missing slurm_job_id in {path}")
        if job_id in timings:
            raise ValueError(f"duplicate timing record for Slurm job {job_id}")
        timings[job_id] = data
    return timings


def numeric_stats(values: Iterable[float]) -> dict[str, float]:
    items = list(values)
    return {
        "count": len(items),
        "total_seconds": round(sum(items), 3),
        "minimum_seconds": round(min(items), 3),
        "median_seconds": round(statistics.median(items), 3),
        "maximum_seconds": round(max(items), 3),
    }


def active_stats(values: Iterable[float]) -> dict[str, float]:
    stats = numeric_stats(values)
    return {
        "count": stats["count"],
        "total_active_seconds": stats["total_seconds"],
        "minimum_active_seconds": stats["minimum_seconds"],
        "median_active_seconds": stats["median_seconds"],
        "maximum_active_seconds": stats["maximum_seconds"],
    }


def interval_union_seconds(intervals: Iterable[tuple[datetime, datetime]]) -> float:
    ordered = sorted(intervals)
    if not ordered:
        return 0.0
    total = 0.0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += (current_end - current_start).total_seconds()
            current_start, current_end = start, end
    total += (current_end - current_start).total_seconds()
    return round(total, 3)


def summarize_timings(
    *,
    sacct_path: Path,
    eval_root: Path,
    jobs_out: Path,
    summary_out: Path,
) -> dict[str, object]:
    with sacct_path.open(newline="") as handle:
        source_rows = [
            row
            for row in csv.DictReader(handle, delimiter="|")
            if row.get("JobIDRaw") and "." not in row["JobIDRaw"]
        ]
    timings = load_judge_timings(eval_root)
    rows: list[dict[str, object]] = []
    intervals: list[tuple[datetime, datetime]] = []
    for source in source_rows:
        stage, checkpoint, judge = classify_job(source["JobName"])
        active_seconds = float(source["ElapsedRaw"]) if source.get("ElapsedRaw") else None
        queue_wait = seconds_between(source.get("Eligible", ""), source.get("Start", ""))
        submit_to_start = seconds_between(source.get("Submit", ""), source.get("Start", ""))
        allocated = source.get("AllocTRES") or source.get("ReqTRES") or ""
        timing = timings.get(source["JobIDRaw"], {})
        row: dict[str, object] = {
            "job_id": source["JobIDRaw"],
            "job_name": source["JobName"],
            "stage": stage,
            "checkpoint": checkpoint,
            "judge": judge,
            "state": source.get("State", ""),
            "exit_code": source.get("ExitCode", ""),
            "gpus": parse_gpus(allocated),
            "submit": source.get("Submit", ""),
            "eligible": source.get("Eligible", ""),
            "start": source.get("Start", ""),
            "end": source.get("End", ""),
            "queue_wait_seconds": queue_wait,
            "submit_to_start_seconds": submit_to_start,
            "active_seconds": active_seconds,
            "model_startup_seconds": timing.get("model_startup_seconds", ""),
            "scoring_seconds": timing.get("scoring_seconds", ""),
            "instrumented_total_seconds": timing.get("instrumented_total_seconds", ""),
        }
        rows.append(row)
        start = parse_time(source.get("Start", ""))
        end = parse_time(source.get("End", ""))
        if source.get("State") == "COMPLETED" and start is not None and end is not None:
            intervals.append((start, end))

    jobs_out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with jobs_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    stage_values: dict[str, list[float]] = defaultdict(list)
    judge_startup: dict[str, list[float]] = defaultdict(list)
    judge_scoring: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["state"] == "COMPLETED" and row["active_seconds"] not in (None, ""):
            stage_values[str(row["stage"])].append(float(row["active_seconds"]))
        if row["judge"] and row["model_startup_seconds"] != "":
            judge_startup[str(row["judge"])].append(float(row["model_startup_seconds"]))
        if row["judge"] and row["scoring_seconds"] != "":
            judge_scoring[str(row["judge"])].append(float(row["scoring_seconds"]))

    stages = {stage: active_stats(values) for stage, values in sorted(stage_values.items())}
    judges: dict[str, dict[str, float | int]] = {}
    for judge in sorted(set(judge_startup) | set(judge_scoring)):
        startup = numeric_stats(judge_startup[judge])
        scoring = numeric_stats(judge_scoring[judge])
        judges[judge] = {
            "timed_cell_count": min(startup["count"], scoring["count"]),
            "total_model_startup_seconds": startup["total_seconds"],
            "median_model_startup_seconds": startup["median_seconds"],
            "total_scoring_seconds": scoring["total_seconds"],
            "median_scoring_seconds": scoring["median_seconds"],
        }

    summary: dict[str, object] = {
        "format_version": 1,
        "job_count": len(rows),
        "completed_job_count": sum(row["state"] == "COMPLETED" for row in rows),
        "critical_path_active_seconds": interval_union_seconds(intervals),
        "stages": stages,
        "judges": judges,
    }
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sacct", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--jobs-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_timings(
        sacct_path=args.sacct,
        eval_root=args.eval_root,
        jobs_out=args.jobs_out,
        summary_out=args.summary_out,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
