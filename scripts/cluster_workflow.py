"""Shared primitives for immutable cluster source publication and execution."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterator, Sequence


SOURCE_MANIFEST = "SOURCE_MANIFEST.json"
DEFAULT_MAIN_REF = "lancewicki/main"
DEFAULT_REMOTE_SOURCE_ROOT = "/home/lancewicki/projects/turing-rl-sources"
DEFAULT_STATE_ROOT = "/home/lancewicki/projects/turing-rl"
ENVIRONMENT_PATHS = {
    "train": "/home/lancewicki/miniconda3/envs/turing-rl-train",
    "rl_qwen35": "/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35",
    "sft_qwen35": "/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35",
    "judge_vllm": "/home/lancewicki/miniconda3/envs/judge-vllm",
    "gemma4": "/home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly",
    "verl_upstream": "/home/lancewicki/miniconda3/envs/verl-upstream",
}
DEPENDENCY_PROFILES = {
    "eval": {
        "environments": ("train", "rl_qwen35", "sft_qwen35", "judge_vllm", "gemma4"),
        "include_verl": False,
    },
    "training": {
        "environments": ("train", "rl_qwen35", "judge_vllm"),
        "include_verl": True,
    },
    "sft": {
        "environments": ("train", "sft_qwen35"),
        "include_verl": False,
    },
    "data": {
        "environments": ("train", "verl_upstream"),
        "include_verl": False,
    },
    "all": {
        "environments": tuple(ENVIRONMENT_PATHS),
        "include_verl": True,
    },
}
SSH_OPTIONS = (
    "-p",
    "2223",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=10",
)
SSH_HOST = "lancewicki@localhost"
SCP_OPTIONS = (
    "-P",
    "2223",
    "-o",
    "StrictHostKeyChecking=no",
    "-o",
    "UserKnownHostsFile=/dev/null",
    "-o",
    "ConnectTimeout=10",
)


def run(
    args: Sequence[str],
    *,
    cwd: Path | None = None,
    input_bytes: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = run(("git", *args), cwd=repo, check=check)
    return result.stdout.decode("utf-8", errors="strict").strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git_common_dir(repo: Path) -> Path:
    value = Path(git(repo, "rev-parse", "--git-common-dir"))
    if not value.is_absolute():
        value = repo / value
    return value.resolve()


@contextlib.contextmanager
def repository_lock(repo: Path, *, exclusive: bool, owner: str) -> Iterator[Path]:
    """Hold the shared-repository integration lock for the current process."""

    lock_path = git_common_dir(repo) / "turing-rl-integration.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        if exclusive:
            handle.seek(0)
            handle.truncate()
            handle.write(
                json.dumps(
                    {"owner": owner, "pid": os.getpid(), "mode": "exclusive"},
                    sort_keys=True,
                )
                + "\n"
            )
            handle.flush()
        try:
            yield lock_path
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def clean_worktree(repo: Path) -> tuple[bool, str]:
    status = git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    return not bool(status), status


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    result = run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=repo,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return result.returncode == 0


def source_manifest(repo: Path, source_sha: str) -> dict[str, object]:
    """Describe every Git leaf by object ID and extracted byte digest."""

    raw = run(
        ("git", "ls-tree", "-rz", "--full-tree", source_sha),
        cwd=repo,
    ).stdout
    entries: list[dict[str, object]] = []
    records = [record for record in raw.split(b"\0") if record]
    for record in records:
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode().split(" ")
        if object_type != "blob":
            raise RuntimeError(
                f"unsupported Git object {object_type!r} at {os.fsdecode(raw_path)!r}"
            )
        content = run(("git", "cat-file", "blob", object_id), cwd=repo).stdout
        entries.append(
            {
                "path": os.fsdecode(raw_path),
                "mode": mode,
                "type": object_type,
                "object_id": object_id,
                "size": len(content),
                "sha256": sha256_bytes(content),
            }
        )
    entries.sort(key=lambda item: str(item["path"]).encode("utf-8", errors="surrogateescape"))
    return {
        "source_sha": source_sha,
        "source_tree_sha": git(repo, "rev-parse", f"{source_sha}^{{tree}}"),
        "paths": entries,
        "path_manifest_sha256": sha256_bytes(canonical_json(entries)),
    }


def validate_relative_path(value: str, *, name: str) -> str:
    path = Path(value)
    if path.is_absolute() or value in ("", ".") or ".." in path.parts:
        raise ValueError(f"{name} must be a non-empty relative path without '..': {value!r}")
    return value


def is_secret_name(value: str) -> bool:
    normalized = value.lstrip("-").replace("-", "_").upper()
    return (
        any(marker in normalized for marker in ("TOKEN", "PASSWORD", "SECRET", "CREDENTIAL"))
        or normalized in {"KEY", "API_KEY"}
        or normalized.endswith(("_API_KEY", "_ACCESS_KEY", "_PRIVATE_KEY"))
    )


def redact_argument(value: str) -> str:
    """Redact secret-like assignments while retaining argument structure."""

    if "=" not in value:
        return value
    prefix, payload = value.split("=", 1)
    if is_secret_name(prefix):
        return f"{prefix}=<redacted>"
    assignments = payload.split(",")
    cleaned: list[str] = []
    for assignment in assignments:
        if "=" not in assignment:
            cleaned.append(assignment)
            continue
        key, assigned = assignment.split("=", 1)
        cleaned.append(
            f"{key}=<redacted>"
            if is_secret_name(key)
            else f"{key}={assigned}"
        )
    return f"{prefix}={','.join(cleaned)}"
