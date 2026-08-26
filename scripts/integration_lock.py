#!/usr/bin/env python3
"""Run a command while holding the repository integration lock."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cluster_workflow import repository_lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shared", action="store_true", help="Take a shared/read lock")
    parser.add_argument("--owner", default=os.environ.get("USER", "unknown"))
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("provide a command after --")
    repo = Path(__file__).resolve().parents[1]
    with repository_lock(repo, exclusive=not args.shared, owner=args.owner):
        return subprocess.call(command, cwd=repo)


if __name__ == "__main__":
    raise SystemExit(main())
