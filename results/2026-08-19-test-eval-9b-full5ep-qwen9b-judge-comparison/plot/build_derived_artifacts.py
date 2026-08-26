"""Build the comparison job table and mechanical validation report."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGULAR_SOURCE_ROOT = ROOT.parent / "2026-08-10-test-eval-9b-full5ep-full-schema"
EXPECTED_STEPS = list(range(0, 321, 32))
CHECKPOINT_RE = re.compile(r"step(\d+)$")
BRIER_JOB_RE = re.compile(r"te_brier_off_(\d+)$")
COMPLETE_FRAGMENT = (
    "rows=880 unique=880 expected=880 missing=0 extra=0 duplicated=0"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        return json.load(handle)


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps: list[int] = []
    for row in rows:
        match = CHECKPOINT_RE.search(row["checkpoint"])
        if not match:
            raise SystemExit(f"FAIL: cannot parse checkpoint in {path}: {row['checkpoint']!r}")
        steps.append(int(match.group(1)))
        if int(row["n_scored"]) != 880 or int(row["n_unique_pairs"]) != 880:
            raise SystemExit(f"FAIL: incomplete summary row in {path}: {row}")
    if steps != EXPECTED_STEPS:
        raise SystemExit(f"FAIL: expected steps {EXPECTED_STEPS} in {path}; got {steps}")
    return rows


def regular_job_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows = []
    for source in source_rows:
        if source["judge"] != "qwen35-9b":
            continue
        rows.append(
            {
                "judge": "qwen35-9b-zero-shot",
                "thinking_mode": "on",
                "step": source["step"],
                "job_id": source["job_id"],
                "job_name": source["job_name"],
                "state": source["state"],
                "elapsed": source["elapsed"],
                "submit": "",
                "start": source["start"],
                "end": source["end"],
                "exit_code": source["exit_code"],
            }
        )
    return sorted(rows, key=lambda row: int(row["step"]))


def brier_job_rows(path: Path, baseline_job_id: str) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="|"))
    rows = []
    for source in source_rows:
        job_id = source["JobIDRaw"]
        if job_id == baseline_job_id:
            step = 0
        else:
            match = BRIER_JOB_RE.fullmatch(source["JobName"])
            if not match:
                raise SystemExit(f"FAIL: cannot derive Brier step from Slurm row: {source}")
            step = int(match.group(1))
        rows.append(
            {
                "judge": "qwen35-9b-brier-train-off",
                "thinking_mode": "off",
                "step": str(step),
                "job_id": job_id,
                "job_name": source["JobName"],
                "state": source["State"],
                "elapsed": source["Elapsed"],
                "submit": source["Submit"],
                "start": source["Start"],
                "end": source["End"],
                "exit_code": source["ExitCode"],
            }
        )
    return sorted(rows, key=lambda row: int(row["step"]))


def validate_job_rows(rows: list[dict[str, str]], label: str) -> None:
    steps = [int(row["step"]) for row in rows]
    if steps != EXPECTED_STEPS:
        raise SystemExit(f"FAIL: expected steps {EXPECTED_STEPS} for {label}; got {steps}")
    for row in rows:
        if row["state"] != "COMPLETED" or row["exit_code"] != "0:0":
            raise SystemExit(f"FAIL: unsuccessful job in {label}: {row}")


def count_verifier_rows(path: Path, prefix: str) -> int:
    lines = path.read_text().splitlines()
    matching = [line for line in lines if line.startswith(prefix)]
    if len(matching) != len(EXPECTED_STEPS):
        raise SystemExit(
            f"FAIL: expected {len(EXPECTED_STEPS)} verifier rows starting {prefix!r} "
            f"in {path}; got {len(matching)}"
        )
    incomplete = [line for line in matching if COMPLETE_FRAGMENT not in line]
    if incomplete:
        raise SystemExit(f"FAIL: incomplete verifier rows in {path}: {incomplete}")
    return len(matching)


def main() -> None:
    regular_summary = ROOT / "summary_qwen35-9b-zero-shot-thinking-on.csv"
    brier_summary = ROOT / "summary_qwen35-9b-brier-train-off-eval-off.csv"
    regular_jobs_source = REGULAR_SOURCE_ROOT / "judge_jobs.csv"
    brier_sacct = ROOT / "provenance/brier_sacct.psv"
    baseline_reuse_path = ROOT / "provenance/baseline_reuse.json"
    regular_validation = ROOT / "provenance/regular_validation.txt"
    brier_validation = ROOT / "provenance/brier_validation.txt"
    launch_path = ROOT / "provenance/brier_launch.json"
    expected_runtime_path = ROOT / "provenance/expected_runtime.json"
    split_guard_path = ROOT / "split_guard.json"

    read_summary(regular_summary)
    read_summary(brier_summary)

    baseline_reuse = read_json(baseline_reuse_path)
    baseline_job_id = str(baseline_reuse["source_slurm_job_id"])
    regular_jobs = regular_job_rows(regular_jobs_source)
    brier_jobs = brier_job_rows(brier_sacct, baseline_job_id)
    validate_job_rows(regular_jobs, "regular Qwen 9B")
    validate_job_rows(brier_jobs, "Brier Qwen 9B")

    jobs_path = ROOT / "judge_jobs.csv"
    fieldnames = [
        "judge",
        "thinking_mode",
        "step",
        "job_id",
        "job_name",
        "state",
        "elapsed",
        "submit",
        "start",
        "end",
        "exit_code",
    ]
    with jobs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(regular_jobs + brier_jobs)

    regular_verified = count_verifier_rows(regular_validation, "[qwen35-9b/on]")
    brier_verified = count_verifier_rows(
        brier_validation, "[judge-9b-brier-train-off/off]"
    )

    split_guard = read_json(split_guard_path)
    if split_guard["verdict"] != "PASS" or int(split_guard["eval_rows"]) != 880:
        raise SystemExit(f"FAIL: held-out split guard did not pass: {split_guard}")

    launch = read_json(launch_path)
    expected_runtime = read_json(expected_runtime_path)
    if launch["source_sha"] != expected_runtime["source_sha"]:
        raise SystemExit("FAIL: Brier launch and runtime manifests disagree on source SHA")

    validation_lines = [
        "PASS: package mechanical validation",
        f"steps={','.join(str(step) for step in EXPECTED_STEPS)}",
        "expected_pairs_per_step=880",
        f"regular_summary_sha256={sha256(regular_summary)}",
        f"brier_summary_sha256={sha256(brier_summary)}",
        f"regular_source_verifier_rows={regular_verified}",
        f"brier_source_verifier_rows={brier_verified}",
        f"regular_job_rows={len(regular_jobs)}",
        f"brier_job_rows={len(brier_jobs)}",
        f"split_guard_verdict={split_guard['verdict']}",
        f"split_guard_eval_rows={split_guard['eval_rows']}",
        f"split_guard_eval_users={split_guard['eval_users']}",
        f"brier_source_sha={launch['source_sha']}",
    ]
    (ROOT / "validation.txt").write_text("\n".join(validation_lines) + "\n")
    print(f"wrote {jobs_path}")
    print(f"wrote {ROOT / 'validation.txt'}")


if __name__ == "__main__":
    main()
