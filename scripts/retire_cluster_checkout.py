#!/usr/bin/env python3
"""Recoverably convert the legacy mixed cluster checkout into a state-only root."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
import subprocess
from pathlib import Path


DEFAULT_STATE_ROOT = Path("/home/lancewicki/projects/turing-rl")
PRESERVE = {
    ".env",
    "checkpoints",
    "data",
    "logs",
    "outputs",
    "results",
    "tmp",
    "wandb",
}


def record_uses_legacy_source(record: str, state_root: Path) -> bool:
    prefix = str(state_root)
    command_match = re.search(r"(?:^|\s)Command=(\S+)", record)
    workdir_match = re.search(r"(?:^|\s)WorkDir=(\S+)", record)
    command = command_match.group(1) if command_match else ""
    workdir = workdir_match.group(1) if workdir_match else ""
    return command == prefix or command.startswith(f"{prefix}/") or workdir == prefix


def legacy_jobs(state_root: Path) -> list[str]:
    queue = subprocess.run(
        ["squeue", "--me", "-h", "-o", "%A"],
        text=True,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout.split()
    blocked: list[str] = []
    for job_id in queue:
        record = subprocess.run(
            ["scontrol", "show", "job", "-o", job_id],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        if record_uses_legacy_source(record, state_root):
            blocked.append(job_id)
    return blocked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    state_root = args.state_root.resolve()
    expected_state_root = DEFAULT_STATE_ROOT.resolve()
    if state_root != expected_state_root:
        raise SystemExit(f"refusing unexpected state root: {state_root}")
    blocked = legacy_jobs(state_root)
    if blocked:
        raise SystemExit(f"legacy jobs still execute from the mixed checkout: {blocked}")
    movable = sorted(path for path in state_root.iterdir() if path.name not in PRESERVE)
    print("entries to quarantine:")
    for path in movable:
        print(f"  {path}")
    if not args.execute:
        print("dry run only; pass --execute after reviewing the inventory")
        return 0
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    quarantine = state_root.parent / "turing-rl-quarantine" / stamp
    quarantine.mkdir(parents=True)
    for path in movable:
        shutil.move(str(path), quarantine / path.name)
    (state_root / "STATE_ROOT_ONLY.txt").write_text(
        "Mutable experiment state only. Do not deploy or execute repository source here.\n"
        "Use scripts/cluster_launch.sh from a local committed worktree.\n"
        f"Quarantined legacy source: {quarantine}\n"
    )
    print(f"legacy source moved recoverably to {quarantine}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
