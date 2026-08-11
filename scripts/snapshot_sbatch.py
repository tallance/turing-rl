#!/usr/bin/env python3
"""The sole maintained gateway from repository launchers to Slurm sbatch."""

from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_workflow import canonical_json, redact_argument


REQUIRED_ENV = (
    "TURING_RL_CODE_ROOT",
    "TURING_RL_STATE_ROOT",
    "TURING_RL_SOURCE_SHA",
    "TURING_RL_RUN_CLASS",
    "TURING_RL_RUN_ROOT",
    "TURING_RL_DEPENDENCY_PROFILE",
)


def atomic_write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value))
    os.replace(temporary, path)


def main() -> int:
    for name in REQUIRED_ENV:
        if not os.environ.get(name):
            print(f"FATAL: {name} is required", file=sys.stderr)
            return 2
    args = sys.argv[1:]
    if not args:
        print("usage: snapshot_sbatch.py [sbatch options] -- SCRIPT [args...]", file=sys.stderr)
        return 2
    if "--" not in args:
        print("FATAL: use an explicit '--' boundary before the Slurm script", file=sys.stderr)
        return 2
    boundary = args.index("--")
    options = args[:boundary]
    payload = args[boundary + 1 :]
    if not payload:
        print("FATAL: no Slurm script follows the explicit '--' boundary", file=sys.stderr)
        return 2
    if "--wrap" in options or any(arg.startswith("--wrap=") for arg in options):
        print("FATAL: --wrap bypasses immutable script validation", file=sys.stderr)
        return 2
    output_options = ("--output", "--error", "-o", "-e")
    for arg in options:
        if arg in output_options or any(
            arg.startswith(f"{option}=") for option in output_options
        ) or arg.startswith(("-o", "-e")) and not arg.startswith("--"):
            print(
                "FATAL: output paths are managed below TURING_RL_RUN_ROOT",
                file=sys.stderr,
            )
            return 2
    for arg in options:
        if arg.startswith("--export="):
            exported = arg.split("=", 1)[1]
            if exported != "ALL" and not exported.startswith("ALL,"):
                print("FATAL: --export must include ALL so TURING_RL_* provenance survives", file=sys.stderr)
                return 2
    options = [arg for arg in options if arg != "--parsable"]
    script_arg = payload[0]
    script_arguments = payload[1:]
    script = Path(script_arg).resolve()
    code_root = Path(os.environ["TURING_RL_CODE_ROOT"]).resolve()
    try:
        script.relative_to(code_root)
    except ValueError:
        print(f"FATAL: Slurm script is outside immutable source: {script}", file=sys.stderr)
        return 2
    if not script.is_file():
        print(f"FATAL: Slurm script missing: {script}", file=sys.stderr)
        return 2

    run_root = Path(os.environ["TURING_RL_RUN_ROOT"])
    log_root = run_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    sbatch_arguments = [
        *options,
        f"--output={log_root}/slurm-%x-%j.out",
        f"--error={log_root}/slurm-%x-%j.err",
        str(script),
        *script_arguments,
    ]

    result = subprocess.run(
        ["sbatch", "--parsable", *sbatch_arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        return result.returncode
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        print(f"FATAL: unexpected sbatch output: {result.stdout!r}", file=sys.stderr)
        return 2
    provenance = {
        "format_version": 1,
        "job_id": job_id,
        "submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_sha": os.environ["TURING_RL_SOURCE_SHA"],
        "code_root": str(code_root),
        "run_class": os.environ["TURING_RL_RUN_CLASS"],
        "script": str(script),
        "dependency_profile": os.environ["TURING_RL_DEPENDENCY_PROFILE"],
        "sbatch_arguments": [redact_argument(value) for value in sbatch_arguments],
    }
    out = run_root / "provenance" / "jobs" / job_id / "submission.json"
    atomic_write(out, provenance)
    print(job_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
