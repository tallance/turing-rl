#!/usr/bin/env python3
"""Atomically publish one clean Git commit as a read-only cluster source tree."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import uuid
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_workflow import (
    DEFAULT_MAIN_REF,
    DEFAULT_REMOTE_SOURCE_ROOT,
    SOURCE_MANIFEST,
    SCP_OPTIONS,
    SSH_HOST,
    SSH_OPTIONS,
    canonical_json,
    clean_worktree,
    git,
    is_ancestor,
    repository_lock,
    sha256_bytes,
    sha256_file,
    source_manifest,
)


def _safe_archive_member(member: tarfile.TarInfo) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or member.isdev() or member.isfifo():
        raise RuntimeError(f"unsafe archive entry: {member.name!r}")
    if member.issym() or member.islnk():
        target = PurePosixPath(member.linkname)
        if target.is_absolute() or ".." in target.parts:
            raise RuntimeError(f"unsafe archive link: {member.name!r} -> {member.linkname!r}")


def _leaf_digest(path: Path) -> tuple[str, int]:
    if path.is_symlink():
        value = os.readlink(path).encode("utf-8", errors="surrogateescape")
        return sha256_bytes(value), len(value)
    return sha256_file(path), path.stat().st_size


def verify_extracted_tree(root: Path, manifest: dict[str, object]) -> None:
    expected_entries = manifest["paths"]
    if not isinstance(expected_entries, list):
        raise RuntimeError("manifest paths must be a list")
    expected = {str(entry["path"]): entry for entry in expected_entries}
    expected_directories: set[str] = set()
    for relative in expected:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual: set[str] = set()
    actual_directories: set[str] = set()
    for path in root.rglob("*"):
        if path.is_dir() and not path.is_symlink():
            actual_directories.add(path.relative_to(root).as_posix())
            continue
        relative = path.relative_to(root).as_posix()
        if relative == SOURCE_MANIFEST:
            continue
        actual.add(relative)
    missing = sorted(set(expected) - actual)
    extra = sorted(actual - set(expected))
    missing_directories = sorted(expected_directories - actual_directories)
    extra_directories = sorted(actual_directories - expected_directories)
    if missing or extra or missing_directories or extra_directories:
        raise RuntimeError(
            "snapshot path mismatch: "
            f"missing={missing[:10]} extra={extra[:10]} "
            f"missing_dirs={missing_directories[:10]} extra_dirs={extra_directories[:10]}"
        )
    for relative, entry in expected.items():
        path = root / relative
        mode = str(entry["mode"])
        if mode == "120000":
            if not path.is_symlink():
                raise RuntimeError(f"expected symlink: {relative}")
        elif mode in ("100644", "100755"):
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(f"expected regular file: {relative}")
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            if executable != (mode == "100755"):
                raise RuntimeError(f"executable mode mismatch: {relative}")
        else:
            raise RuntimeError(f"unsupported Git mode {mode} at {relative}")
        digest, size = _leaf_digest(path)
        if digest != entry["sha256"] or size != entry["size"]:
            raise RuntimeError(f"content mismatch: {relative}")


def make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        if path.is_dir():
            path.chmod(0o555)
        else:
            executable = bool(path.stat().st_mode & stat.S_IXUSR)
            path.chmod(0o555 if executable else 0o444)
    root.chmod(0o555)


def remote_publish(archive: Path, manifest_path: Path, destination: Path, lock: Path) -> None:
    import fcntl

    manifest = json.loads(manifest_path.read_text())
    expected_archive = str(manifest["archive_sha256"])
    if sha256_file(archive) != expected_archive:
        raise RuntimeError("uploaded archive digest does not match SOURCE_MANIFEST.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            verify_extracted_tree(destination, manifest)
            existing = json.loads((destination / SOURCE_MANIFEST).read_text())
            publication_fields = {"published_at_utc"}
            existing_identity = {k: v for k, v in existing.items() if k not in publication_fields}
            requested_identity = {k: v for k, v in manifest.items() if k not in publication_fields}
            if existing_identity != requested_identity:
                raise RuntimeError(f"existing snapshot manifest differs: {destination}")
            return
        temporary = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
        try:
            temporary.mkdir(mode=0o700)
            with tarfile.open(archive, "r:") as tar:
                for member in tar.getmembers():
                    _safe_archive_member(member)
                try:
                    tar.extractall(temporary, filter="data")
                except TypeError:  # Python < 3.12
                    tar.extractall(temporary)
            verify_extracted_tree(temporary, manifest)
            (temporary / SOURCE_MANIFEST).write_bytes(canonical_json(manifest))
            verify_extracted_tree(temporary, manifest)
            make_read_only(temporary)
            os.rename(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)


def _scp(local_paths: list[Path], remote_dir: str) -> None:
    command = ["scp", *SCP_OPTIONS]
    command.extend(str(path) for path in local_paths)
    command.append(f"{SSH_HOST}:{remote_dir}/")
    subprocess.run(command, check=True)


def publish(args: argparse.Namespace) -> dict[str, object]:
    repo = Path(args.repo).resolve()
    with repository_lock(repo, exclusive=False, owner=args.owner):
        clean, status = clean_worktree(repo)
        if not clean:
            raise RuntimeError(f"refusing dirty source tree:\n{status}")
        source_sha = git(repo, "rev-parse", f"{args.commit}^{{commit}}")
        main_sha = git(repo, "rev-parse", f"{args.main_ref}^{{commit}}")
        if args.debug:
            if not args.label:
                raise RuntimeError("--debug requires --label")
        elif not is_ancestor(repo, main_sha, source_sha):
            raise RuntimeError(
                f"retained source {source_sha} does not contain current {args.main_ref} {main_sha}; "
                "integrate main or publish with --debug --label"
            )
        elif args.label:
            raise RuntimeError("--label is valid only with --debug")

        with tempfile.TemporaryDirectory(prefix="turing-rl-publish-") as temp_value:
            temp = Path(temp_value)
            archive = temp / "source.tar"
            subprocess.run(
                ["git", "archive", "--format=tar", "--output", str(archive), source_sha],
                cwd=repo,
                check=True,
            )
            manifest = source_manifest(repo, source_sha)
            manifest.update(
                {
                    "format_version": 1,
                    "archive_sha256": sha256_file(archive),
                    "published_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
            )
            manifest_path = temp / SOURCE_MANIFEST
            manifest_path.write_bytes(canonical_json(manifest))

            incoming = f"{args.remote_root}/.incoming/{uuid.uuid4().hex}"
            subprocess.run(
                ["ssh", *SSH_OPTIONS, SSH_HOST, "mkdir", "-p", f"{incoming}/scripts"],
                check=True,
            )
            _scp([archive, manifest_path], incoming)
            helper = Path(__file__).with_name("cluster_workflow.py").resolve()
            _scp([Path(__file__).resolve(), helper], f"{incoming}/scripts")
            destination = f"{args.remote_root}/{source_sha}"
            remote_script = f"{incoming}/scripts/{Path(__file__).name}"
            try:
                subprocess.run(
                    [
                        "ssh",
                        *SSH_OPTIONS,
                        SSH_HOST,
                        "env",
                        f"PYTHONPATH={incoming}",
                        "python3",
                        remote_script,
                        "--remote-publish",
                        "--archive",
                        f"{incoming}/{archive.name}",
                        "--manifest",
                        f"{incoming}/{manifest_path.name}",
                        "--destination",
                        destination,
                        "--lock",
                        f"{args.remote_root}/.publish.lock",
                    ],
                    check=True,
                )
            finally:
                subprocess.run(
                    ["ssh", *SSH_OPTIONS, SSH_HOST, "rm", "-rf", incoming],
                    check=False,
                )
    return {
        "source_sha": source_sha,
        "main_sha": main_sha,
        "run_class": "debug" if args.debug else "retained",
        "debug_label": args.label if args.debug else None,
        "remote_path": destination,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--commit", default="HEAD")
    parser.add_argument("--main-ref", default=DEFAULT_MAIN_REF)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_SOURCE_ROOT)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--remote-publish", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--archive", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--destination", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--lock", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.remote_publish:
        required = (args.archive, args.manifest, args.destination, args.lock)
        if not all(required):
            raise SystemExit("remote publish requires archive, manifest, destination, and lock")
        remote_publish(args.archive, args.manifest, args.destination, args.lock)
        return 0
    try:
        result = publish(args)
    except (RuntimeError, subprocess.CalledProcessError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(result["remote_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
