#!/usr/bin/env python3
"""Copy a previously validated evaluation step into a fresh result root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PairKey = tuple[str, str, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"source tree contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()


def copy_cell_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(source / "run_metadata.json", destination / "run_metadata.json")
    shutil.copytree(source / "reward", destination / "reward")


def reward_keys(reward_dir: Path, *, expect_pairs: int, cell: str) -> set[PairKey]:
    files = sorted(reward_dir.glob("reward-*.jsonl"))
    if not files:
        raise ValueError(f"no reward JSONL files for {cell}: {reward_dir}")
    keys: list[PairKey] = []
    for path in files:
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                try:
                    key = (str(row["user_id"]), str(row["post_id"]), int(row["target_idx"]))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid pair key in {path}:{line_number}") from exc
                keys.append(key)
    unique = set(keys)
    if len(keys) != len(unique):
        raise ValueError(f"duplicate reward keys for {cell}: rows={len(keys)} unique={len(unique)}")
    if len(unique) != expect_pairs:
        raise ValueError(f"expected {expect_pairs} unique keys for {cell}; got {len(unique)}")
    return unique


def validate_cell(
    cell_dir: Path,
    *,
    cell: str,
    mode: str,
    pair_path: Path,
    expect_pairs: int,
) -> tuple[set[PairKey], str]:
    metadata_path = cell_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise ValueError(f"missing source metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    for key in ("cell_name", "thinking_mode", "slurm_job_id", "pair_source", "n_pairs_total"):
        if key not in metadata or metadata[key] in (None, ""):
            raise ValueError(f"missing {key} in {metadata_path}")
    if metadata["cell_name"] != cell or metadata["thinking_mode"] != mode:
        raise ValueError(f"source metadata does not describe {cell}/{mode}: {metadata_path}")
    if int(metadata["n_pairs_total"]) != expect_pairs:
        raise ValueError(
            f"source metadata expected {metadata['n_pairs_total']} pairs for {cell}; "
            f"required {expect_pairs}"
        )
    if Path(str(metadata["pair_source"])).resolve() != pair_path.resolve():
        raise ValueError(f"source metadata pair_source mismatch for {cell}: {metadata['pair_source']}")
    keys = reward_keys(cell_dir / "reward", expect_pairs=expect_pairs, cell=cell)
    return keys, str(metadata["slurm_job_id"])


def reuse_step(
    *,
    source_root: Path,
    destination_root: Path,
    source_gen_key: str,
    destination_gen_key: str,
    pairs_tag: int,
    cells: Iterable[str],
    expect_pairs: int,
    mode: str = "on",
) -> dict[str, object]:
    cells = tuple(cells)
    if not cells:
        raise ValueError("at least one cell is required")
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    pair_source = source_root / "raw/pairs" / f"gen_{source_gen_key}_{pairs_tag}.parquet"
    if not pair_source.is_file():
        raise ValueError(f"missing source pair parquet: {pair_source}")

    pair_destination = (
        destination_root / "raw/pairs" / f"gen_{destination_gen_key}_{pairs_tag}.parquet"
    )
    step_destination = destination_root / "raw" / destination_gen_key
    manifest_path = destination_root / "provenance/step0_reuse.json"
    for path in (pair_destination, step_destination, manifest_path):
        if path.exists():
            raise ValueError(f"destination already exists: {path}")

    cell_sources: dict[str, Path] = {}
    cell_hashes: dict[str, str] = {}
    source_job_ids: dict[str, str] = {}
    reference_keys: set[PairKey] | None = None
    for cell in cells:
        cell_source = source_root / "raw" / source_gen_key / "sweep" / cell / mode
        keys, job_id = validate_cell(
            cell_source,
            cell=cell,
            mode=mode,
            pair_path=pair_source,
            expect_pairs=expect_pairs,
        )
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"key set differs for {cell}")
        cell_sources[cell] = cell_source
        source_job_ids[cell] = job_id

    destination_root.mkdir(parents=True, exist_ok=True)
    stage_parent = destination_root / ".reuse-staging"
    stage_parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="step0-", dir=stage_parent))
    try:
        staged_pair = stage / "pair.parquet"
        shutil.copy2(pair_source, staged_pair)
        staged_step = stage / "step"
        for cell, cell_source in cell_sources.items():
            staged_cell = staged_step / "sweep" / cell / mode
            copy_cell_artifacts(cell_source, staged_cell)
            cell_hashes[cell] = sha256_tree(staged_cell)

        destination_cell_hashes = {
            cell: sha256_tree(staged_step / "sweep" / cell / mode) for cell in cells
        }
        pair_hash = sha256_file(pair_source)
        destination_pair_hash = sha256_file(staged_pair)
        if pair_hash != destination_pair_hash or cell_hashes != destination_cell_hashes:
            raise ValueError("copied step-0 artifacts failed SHA-256 verification")

        pair_destination.parent.mkdir(parents=True, exist_ok=True)
        step_destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staged_pair, pair_destination)
        os.replace(staged_step, step_destination)

        manifest: dict[str, object] = {
            "format_version": 1,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "destination_root": str(destination_root),
            "source_gen_key": source_gen_key,
            "destination_gen_key": destination_gen_key,
            "mode": mode,
            "cells": list(cells),
            "expected_pairs": expect_pairs,
            "pair_source": str(pair_source),
            "pair_destination": str(pair_destination),
            "pair_sha256": pair_hash,
            "destination_pair_sha256": destination_pair_hash,
            "cell_tree_sha256": cell_hashes,
            "destination_cell_tree_sha256": destination_cell_hashes,
            "source_job_ids": source_job_ids,
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_manifest = manifest_path.with_suffix(".json.tmp")
        temporary_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_manifest, manifest_path)
        return manifest
    finally:
        shutil.rmtree(stage, ignore_errors=True)
        try:
            stage_parent.rmdir()
        except OSError:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--destination-root", type=Path, required=True)
    parser.add_argument("--source-gen-key", required=True)
    parser.add_argument("--destination-gen-key", required=True)
    parser.add_argument("--pairs-tag", type=int, required=True)
    parser.add_argument("--expect-pairs", type=int, required=True)
    parser.add_argument("--mode", choices=("on", "off"), default="on")
    parser.add_argument("--cells", nargs="+", required=True)
    args = parser.parse_args()
    manifest = reuse_step(
        source_root=args.source_root,
        destination_root=args.destination_root,
        source_gen_key=args.source_gen_key,
        destination_gen_key=args.destination_gen_key,
        pairs_tag=args.pairs_tag,
        cells=args.cells,
        expect_pairs=args.expect_pairs,
        mode=args.mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
