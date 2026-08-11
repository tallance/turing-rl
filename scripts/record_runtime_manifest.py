#!/usr/bin/env python3
"""Fingerprint external runtime dependencies and record per-job context."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import tempfile
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_workflow import (
    DEPENDENCY_PROFILES,
    ENVIRONMENT_PATHS,
    canonical_json,
    sha256_bytes,
    sha256_file,
)


KEY_PACKAGES = ("torch", "transformers", "trl", "vllm", "hydra-core", "ray", "peft", "verl")


def _command(args: list[str], cwd: Path | None = None) -> tuple[int, bytes, bytes]:
    try:
        result = subprocess.run(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        return 127, b"", str(exc).encode()
    return result.returncode, result.stdout, result.stderr


def _git_value(repo: Path, *args: str) -> str | None:
    code, stdout, _ = _command(["git", *args], cwd=repo)
    return stdout.decode(errors="replace").strip() if code == 0 else None


def fingerprint_verl(path: Path) -> dict[str, object]:
    record: dict[str, object] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return record
    record["sha"] = _git_value(path, "rev-parse", "HEAD")
    status = _git_value(path, "status", "--porcelain=v1", "--untracked-files=all") or ""
    record["dirty_paths"] = status.splitlines()
    _, diff, _ = _command(["git", "diff", "--binary", "HEAD"], cwd=path)
    record["tracked_diff_sha256"] = sha256_bytes(diff)
    untracked: list[dict[str, object]] = []
    raw_untracked = _git_value(
        path, "ls-files", "--others", "--exclude-standard", "-z"
    ) or ""
    for relative in sorted(item for item in raw_untracked.split("\0") if item):
        candidate = path / relative
        if candidate.is_file():
            untracked.append(
                {
                    "path": relative,
                    "size": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    record["untracked_files"] = untracked
    return record


def fingerprint_environment(name: str, path: Path) -> dict[str, object]:
    python = path / "bin/python"
    record: dict[str, object] = {
        "name": name,
        "path": str(path),
        "python": str(python),
        "exists": python.exists(),
    }
    if not python.exists():
        return record
    probe = r"""
import importlib.metadata as m, json, platform, sys
packages = sorted((d.metadata.get('Name') or d.name, d.version) for d in m.distributions())
keys = {}
for name in %r:
    try: keys[name] = m.version(name)
    except m.PackageNotFoundError: keys[name] = None
print(json.dumps({'python_version': platform.python_version(), 'executable': sys.executable,
                  'packages': packages, 'key_packages': keys}, sort_keys=True))
""" % (KEY_PACKAGES,)
    code, stdout, stderr = _command([str(python), "-c", probe])
    if code != 0:
        record["probe_error"] = stderr.decode(errors="replace")[-2000:]
        return record
    details = json.loads(stdout)
    packages = details.pop("packages")
    record.update(details)
    record["package_count"] = len(packages)
    record["package_list_sha256"] = sha256_bytes(canonical_json(packages))
    return record


def external_dependencies() -> dict[str, object]:
    verl_dir = Path(os.environ.get("TURING_RL_VERL_DIR", "/storage/home/lancewicki/src/verl"))
    return {
        "verl": fingerprint_verl(verl_dir),
        "environments": [
            fingerprint_environment(name, Path(path))
            for name, path in ENVIRONMENT_PATHS.items()
        ],
    }


def enforced_dependencies(inventory: dict[str, object], profile: str) -> dict[str, object]:
    try:
        policy = DEPENDENCY_PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"unknown dependency profile {profile!r}; expected one of {sorted(DEPENDENCY_PROFILES)}"
        ) from exc
    environments = inventory.get("environments")
    if not isinstance(environments, list):
        raise ValueError("external dependency inventory has no environment list")
    by_name = {str(item.get("name")): item for item in environments if isinstance(item, dict)}
    required_names = tuple(policy["environments"])
    missing = [name for name in required_names if name not in by_name]
    if missing:
        raise ValueError(f"external dependency inventory is missing profile environments: {missing}")
    return {
        "profile": profile,
        "verl": inventory.get("verl") if policy["include_verl"] else None,
        "environments": [by_name[name] for name in required_names],
    }


def runtime_context() -> dict[str, object]:
    model_keys = (
        "MODEL",
        "MODEL_ID",
        "JUDGE_MODEL",
        "GEMMA_SNAPSHOT",
        "RUN_TAG",
    )
    context: dict[str, object] = {
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "slurm": {key: value for key, value in os.environ.items() if key.startswith("SLURM_")},
        "models": {key: os.environ[key] for key in model_keys if key in os.environ},
    }
    code, stdout, _ = _command(
        ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"]
    )
    context["gpus"] = stdout.decode(errors="replace").splitlines() if code == 0 else []
    code, stdout, _ = _command(["nvidia-smi"])
    context["nvidia_smi_header"] = (
        stdout.decode(errors="replace").splitlines()[:3] if code == 0 else []
    )
    code, stdout, _ = _command(["nvcc", "--version"])
    context["nvcc_version"] = stdout.decode(errors="replace").splitlines() if code == 0 else []
    return context


def build_manifest() -> dict[str, object]:
    profile = os.environ.get("TURING_RL_DEPENDENCY_PROFILE", "")
    if profile not in DEPENDENCY_PROFILES:
        raise SystemExit(
            "TURING_RL_DEPENDENCY_PROFILE must be one of "
            + ", ".join(sorted(DEPENDENCY_PROFILES))
        )
    inventory = external_dependencies()
    enforced = enforced_dependencies(inventory, profile)
    return {
        "format_version": 2,
        "source_sha": os.environ.get("TURING_RL_SOURCE_SHA"),
        "code_root": os.environ.get("TURING_RL_CODE_ROOT"),
        "run_class": os.environ.get("TURING_RL_RUN_CLASS"),
        "dependency_profile": profile,
        "external_dependencies": inventory,
        "external_inventory_sha256": sha256_bytes(canonical_json(inventory)),
        "enforced_dependencies": enforced,
        "enforced_fingerprint_sha256": sha256_bytes(canonical_json(enforced)),
        "runtime_context": runtime_context(),
    }


def atomic_write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(canonical_json(value))
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    manifest = build_manifest()
    if args.compare:
        expected = json.loads(args.compare.read_text())
        if expected.get("source_sha") != manifest.get("source_sha"):
            raise SystemExit(
                f"source mismatch: expected {expected.get('source_sha')} got {manifest.get('source_sha')}"
            )
        if expected.get("dependency_profile") != manifest["dependency_profile"]:
            raise SystemExit(
                f"dependency profile mismatch: expected {expected.get('dependency_profile')} "
                f"got {manifest['dependency_profile']}"
            )
        if expected.get("enforced_fingerprint_sha256") != manifest["enforced_fingerprint_sha256"]:
            raise SystemExit(
                "an enforced runtime dependency changed since submission; "
                "inspect expected and per-job manifests"
            )
    if args.initialize and args.out.exists():
        expected = json.loads(args.out.read_text())
        if (
            expected.get("source_sha") != manifest.get("source_sha")
            or expected.get("dependency_profile") != manifest.get("dependency_profile")
            or expected.get("enforced_fingerprint_sha256")
            != manifest.get("enforced_fingerprint_sha256")
        ):
            raise SystemExit(f"refusing to overwrite incompatible expected runtime: {args.out}")
        return 0
    atomic_write(args.out, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
