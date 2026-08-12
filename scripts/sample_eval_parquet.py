#!/usr/bin/env python3
"""Create a deterministic row subset of a GRPO-format evaluation parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


def select_indices(keys: Sequence[str], *, rows: int, seed: int) -> list[int]:
    """Select the lowest seeded SHA-256 priorities, retaining source row order."""
    if not 1 <= rows <= len(keys):
        raise ValueError(f"rows must be between 1 and {len(keys)}, got {rows}")
    if len(set(keys)) != len(keys):
        raise ValueError("evaluation keys must be unique")
    ranked = sorted(
        (
            hashlib.sha256(f"{seed}\0{key}".encode()).digest(),
            key,
            index,
        )
        for index, key in enumerate(keys)
    )
    return sorted(index for _, _, index in ranked[:rows])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(extra: object) -> str:
    return "\0".join(
        str(extra[name]) for name in ("user_id", "post_id", "target_idx")  # type: ignore[index]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import pandas as pd

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if not source.is_file():
        raise SystemExit(f"FAIL: input parquet does not exist: {source}")
    if source == output:
        raise SystemExit("FAIL: input and output parquet paths must differ")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise SystemExit(f"FAIL: refusing to overwrite existing subset artifacts: {output}")

    frame = pd.read_parquet(source)
    keys = [row_key(extra) for extra in frame["extra_info"]]
    try:
        indices = select_indices(keys, rows=args.rows, seed=args.seed)
    except ValueError as exc:
        raise SystemExit(f"FAIL: {exc}") from exc
    subset = frame.iloc[indices].reset_index(drop=True)
    selected_keys = [keys[index] for index in indices]

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    try:
        subset.to_parquet(temporary, index=False)
        written = pd.read_parquet(temporary, columns=["extra_info"])
        written_keys = [row_key(extra) for extra in written["extra_info"]]
        if written_keys != selected_keys:
            raise SystemExit("FAIL: written subset keys differ from the selected source keys")
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    key_digest = hashlib.sha256("\n".join(sorted(selected_keys)).encode()).hexdigest()
    users = {key.split("\0", 1)[0] for key in selected_keys}
    metadata = {
        "format_version": 1,
        "source_parquet": str(source),
        "source_sha256": sha256_file(source),
        "source_rows": len(frame),
        "output_parquet": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(subset),
        "users": len(users),
        "selected_key_set_sha256": key_digest,
        "selection": {
            "rule": "lowest SHA-256 priority of '<seed>\\0<user_id>\\0<post_id>\\0<target_idx>'",
            "seed": args.seed,
            "preserve_source_order": True,
        },
    }
    meta_path = output.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {len(subset)}/{len(frame)} rows ({len(users)} users) to {output}\n"
        f"sha256={metadata['output_sha256']}"
    )


if __name__ == "__main__":
    main()
