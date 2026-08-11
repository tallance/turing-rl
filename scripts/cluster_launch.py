#!/usr/bin/env python3
"""Publish immutable source and run one cluster launcher from that snapshot."""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_workflow import (
    DEFAULT_MAIN_REF,
    DEFAULT_REMOTE_SOURCE_ROOT,
    DEFAULT_STATE_ROOT,
    SSH_HOST,
    SSH_OPTIONS,
    clean_worktree,
    git,
    is_secret_name,
    is_ancestor,
    redact_argument,
    validate_relative_path,
)


OUTPUT_PATH_KEYS = {
    "CHECKPOINT_DIR",
    "EVAL_ROOT",
    "JUDGE_ENDPOINT_FILE",
    "OUT",
    "OUT_DIR",
    "OUT_PARQUET",
    "OUT_ROOT",
    "RL_CKPT_DIR",
    "RUN_DIR",
    "SWEEP_BASE",
    "SWEEP_ROOT",
    "WANDB_DIR",
}


def parse_environment(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--env must be NAME=VALUE, got {value!r}")
        key, assigned = value.split("=", 1)
        if not key or not key.replace("_", "a").isalnum() or key[0].isdigit():
            raise ValueError(f"invalid environment variable name: {key!r}")
        if key.startswith("TURING_RL_"):
            raise ValueError(f"reserved environment variable: {key}")
        if is_secret_name(key):
            raise ValueError(f"pass {key} through the state-root .env, not --env")
        result[key] = assigned
    return result


def validate_run_root(state_root: str, run_root: str, *, debug: bool, label: str | None) -> None:
    state = PurePosixPath(state_root)
    run = PurePosixPath(run_root)
    if not state.is_absolute() or not run.is_absolute():
        raise ValueError("state root and run root must be absolute cluster paths")
    if ".." in state.parts or ".." in run.parts:
        raise ValueError("state root and run root may not contain '..'")
    if run == state:
        raise ValueError("run root must be a directory below state root, not the state root itself")
    if debug and (not label or PurePosixPath(label).name != label or label in (".", "..")):
        raise ValueError("debug label must be one safe path component")
    try:
        run.relative_to(state)
    except ValueError as exc:
        raise ValueError(f"run root must be below state root {state}") from exc
    debug_root = state / "results" / "debug"
    if debug:
        assert label
        required = debug_root / label
        try:
            run.relative_to(required)
        except ValueError as exc:
            raise ValueError(f"debug run root must be below {required}") from exc
    else:
        try:
            run.relative_to(debug_root)
        except ValueError:
            pass
        else:
            raise ValueError("retained runs may not write below results/debug")


def _ssh(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["ssh", *SSH_OPTIONS, SSH_HOST, shlex.join(command)],
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--main-ref", default=DEFAULT_MAIN_REF)
    parser.add_argument("--remote-source-root", default=DEFAULT_REMOTE_SOURCE_ROOT)
    parser.add_argument("--state-root", default=DEFAULT_STATE_ROOT)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("launcher")
    parser.add_argument("launcher_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = Path(args.repo).resolve()
    try:
        launcher = validate_relative_path(args.launcher, name="launcher")
        environment = parse_environment(args.env)
        for argument in args.launcher_args:
            key = argument.split("=", 1)[0]
            if is_secret_name(key):
                raise ValueError(f"pass secret argument {key} through the state-root .env")
        if args.debug and not args.label:
            raise ValueError("--debug requires --label")
        if args.label and not args.debug:
            raise ValueError("--label requires --debug")
        validate_run_root(
            args.state_root,
            args.run_root,
            debug=args.debug,
            label=args.label,
        )
        if args.debug:
            run_path = PurePosixPath(args.run_root)
            for key in OUTPUT_PATH_KEYS & environment.keys():
                value = PurePosixPath(environment[key])
                if value.is_absolute():
                    try:
                        value.relative_to(run_path)
                    except ValueError as exc:
                        raise ValueError(f"debug output {key} must be below {run_path}") from exc
        clean, status = clean_worktree(repo)
        if not clean:
            raise ValueError(f"refusing dirty source tree:\n{status}")
        source_sha = git(repo, "rev-parse", f"{args.commit}^{{commit}}")
        main_sha = git(repo, "rev-parse", f"{args.main_ref}^{{commit}}")
        if not args.debug and not is_ancestor(repo, main_sha, source_sha):
            raise ValueError(f"retained source {source_sha} does not contain main {main_sha}")
        subprocess.run(
            ["git", "cat-file", "-e", f"{source_sha}:{launcher}"],
            cwd=repo,
            check=True,
        )
    except (ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    plan = {
        "source_sha": source_sha,
        "main_sha": main_sha,
        "run_class": "debug" if args.debug else "retained",
        "debug_label": args.label if args.debug else None,
        "state_root": args.state_root,
        "run_root": args.run_root,
        "launcher": launcher,
        "launcher_args": [redact_argument(value) for value in args.launcher_args],
        "environment_keys": sorted(environment),
    }
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    publish_command = [
        sys.executable,
        str(repo / "scripts/publish_cluster_source.py"),
        "--repo",
        str(repo),
        "--commit",
        source_sha,
        "--main-ref",
        args.main_ref,
        "--remote-root",
        args.remote_source_root,
        "--json",
    ]
    if args.debug:
        publish_command.extend(("--debug", "--label", args.label))
    published = subprocess.run(
        publish_command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    publication = json.loads(published.stdout)
    if publication.get("source_sha") != source_sha:
        print("ERROR: publisher returned a different source SHA", file=sys.stderr)
        return 2
    main_sha = str(publication["main_sha"])
    plan["main_sha"] = main_sha
    code_root = str(publication["remote_path"])
    run_class = str(publication["run_class"])
    base_environment = {
        "TURING_RL_CODE_ROOT": code_root,
        "TURING_RL_STATE_ROOT": args.state_root,
        "TURING_RL_SOURCE_SHA": source_sha,
        "TURING_RL_MAIN_SHA": main_sha,
        "TURING_RL_RUN_CLASS": run_class,
        "TURING_RL_RUN_ROOT": args.run_root,
    }
    if args.label:
        base_environment["TURING_RL_DEBUG_LABEL"] = args.label
    base_environment.update(environment)

    launch_record = {
        **plan,
        "format_version": 1,
        "launched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "code_root": code_root,
    }
    encoded_record = base64.b64encode(
        (json.dumps(launch_record, sort_keys=True, indent=2) + "\n").encode()
    ).decode()
    setup = [
        "mkdir",
        "-p",
        f"{args.run_root}/provenance",
    ]
    result = _ssh(setup)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    write_record = [
        "bash",
        "-c",
        f"printf %s {shlex.quote(encoded_record)} | base64 -d > {shlex.quote(args.run_root + '/provenance/launch.json')}",
    ]
    _ssh(write_record)
    env_args = [f"{key}={value}" for key, value in base_environment.items()]
    expected = [
        "env",
        f"PYTHONPATH={code_root}",
        *env_args,
        "/usr/bin/python3",
        f"{code_root}/scripts/record_runtime_manifest.py",
        "--initialize",
        "--out",
        f"{args.run_root}/provenance/expected_runtime.json",
    ]
    result = _ssh(expected, check=False)
    if result.returncode:
        sys.stderr.write(result.stderr)
        return result.returncode
    command = [
        "env",
        *env_args,
        "bash",
        f"{code_root}/scripts/cluster_launcher_entrypoint.sh",
        "bash",
        f"{code_root}/{launcher}",
        *args.launcher_args,
    ]
    process = subprocess.run(["ssh", *SSH_OPTIONS, SSH_HOST, shlex.join(command)])
    return process.returncode


if __name__ == "__main__":
    raise SystemExit(main())
