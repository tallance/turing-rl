#!/usr/bin/env python3
"""Copy a previously validated evaluation step into a fresh result root."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable, Iterable


PairKey = tuple[str, str, int]
PairKeyReader = Callable[[Path, int], set[PairKey]]
EXPECTED_JUDGES = {
    "gemma4-12b": {
        "model": "google/gemma-4-12B-it",
        "num_endpoints": 8,
        "concurrency_per_endpoint": 4,
    },
    "qwen35-9b": {
        "model": "Qwen/Qwen3.5-9B",
        "num_endpoints": 8,
        "concurrency_per_endpoint": 32,
    },
}
EXPECTED_JUDGE_SAMPLING = {"repetition_penalty": 1.1, "temperature": 0.6}
EXPECTED_GENERATION_ARGS = [
    "--temperature",
    "0.7",
    "--top_p",
    "0.8",
    "--top_k",
    "20",
    "--max_tokens",
    "1024",
    "--vllm_truncate_prompt_tokens",
    "12500",
    "--vllm_max_model_len",
    "13524",
]
ALLOWED_DESTINATION_ENTRIES = {"hydra", "logs", "provenance", "work"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(path: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    files = sorted(files)
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


def sha256_tree(path: Path) -> str:
    return sha256_paths(path, (item for item in path.rglob("*") if item.is_file()))


def sha256_cell_artifacts(path: Path) -> str:
    files = [path / "run_metadata.json", *sorted((path / "reward").glob("reward-*.jsonl"))]
    return sha256_paths(path, files)


def copy_cell_artifacts(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True)
    shutil.copy2(source / "run_metadata.json", destination / "run_metadata.json")
    shutil.copytree(source / "reward", destination / "reward")


def read_pair_keys(path: Path, expect_pairs: int) -> set[PairKey]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required to validate pair parquet keys") from exc
    frame = pd.read_parquet(path, columns=["user_id", "post_id", "target_idx"])
    keys = {
        (str(row.user_id), str(row.post_id), int(row.target_idx))
        for row in frame.itertuples(index=False)
    }
    if len(frame) != expect_pairs or len(keys) != expect_pairs:
        raise ValueError(
            f"pair parquet expected {expect_pairs} unique rows; rows={len(frame)} unique={len(keys)}"
        )
    return keys


def validate_destination_root(destination_root: Path) -> None:
    if not destination_root.exists():
        return
    unexpected = sorted(
        item.name
        for item in destination_root.iterdir()
        if item.name not in ALLOWED_DESTINATION_ENTRIES
    )
    if unexpected:
        raise ValueError(f"existing evaluation payload in destination root: {unexpected}")


def validate_generation_metadata(
    *, source_root: Path, source_gen_key: str, expected_eval_parquet: Path
) -> tuple[dict[str, object], Path]:
    path = source_root / "raw/generator" / source_gen_key / "gen_metadata.json"
    if not path.is_file():
        raise ValueError(f"missing generation metadata: {path}")
    metadata = json.loads(path.read_text())
    expected = {
        "gen_key": source_gen_key,
        "checkpoint_dir": "",
        "base_model": True,
        "eval_expect": "heldout",
        "gen_num": 1,
        "backend": "vllm",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"unexpected generation metadata {key}={metadata.get(key)!r}; expected {value!r}")
    model_id = str(metadata.get("model_id") or "")
    expected_suffix = (
        "/checkpoints/sft/"
        "qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3"
    )
    if not model_id.endswith(expected_suffix):
        raise ValueError(f"unexpected step-0 generator model_id: {model_id}")
    if Path(str(metadata.get("test_parquet") or "")).resolve() != expected_eval_parquet.resolve():
        raise ValueError(f"unexpected generation test_parquet: {metadata.get('test_parquet')}")
    if shlex.split(str(metadata.get("sampling_overrides") or "")) != EXPECTED_GENERATION_ARGS:
        raise ValueError(f"unexpected generation sampling_overrides: {metadata.get('sampling_overrides')}")
    if not metadata.get("slurm_job_id"):
        raise ValueError(f"missing slurm_job_id in {path}")
    return metadata, path


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
    expected_config = EXPECTED_JUDGES.get(cell)
    if expected_config is None:
        raise ValueError(f"no pinned reuse configuration for judge cell {cell}")
    for key in (
        "cell_name",
        "model",
        "thinking_mode",
        "num_endpoints",
        "concurrency_per_endpoint",
        "sampling",
        "json_schema",
        "enable_thinking",
        "max_completion_tokens",
        "disable_openrouter_extras",
        "slurm_job_id",
        "pair_source",
        "n_pairs_total",
    ):
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
    for key, value in expected_config.items():
        if metadata.get(key) != value:
            raise ValueError(f"unexpected {key} for {cell}: {metadata.get(key)!r}; expected {value!r}")
    if json.loads(str(metadata["sampling"])) != EXPECTED_JUDGE_SAMPLING:
        raise ValueError(f"unexpected sampling for {cell}: {metadata['sampling']}")
    exact_fields = {
        "json_schema": "1",
        "enable_thinking": "1" if mode == "on" else "0",
        "max_completion_tokens": "8192",
        "disable_openrouter_extras": "1",
    }
    for key, value in exact_fields.items():
        if str(metadata[key]) != value:
            raise ValueError(f"unexpected {key} for {cell}: {metadata[key]!r}; expected {value!r}")
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
    expected_eval_parquet: Path,
    mode: str = "on",
    pair_key_reader: PairKeyReader = read_pair_keys,
) -> dict[str, object]:
    cells = tuple(cells)
    if not cells:
        raise ValueError("at least one cell is required")
    source_root = source_root.resolve()
    destination_root = destination_root.resolve()
    expected_eval_parquet = expected_eval_parquet.resolve()
    validate_destination_root(destination_root)
    pair_source = source_root / "raw/pairs" / f"gen_{source_gen_key}_{pairs_tag}.parquet"
    if not pair_source.is_file():
        raise ValueError(f"missing source pair parquet: {pair_source}")
    pair_keys = pair_key_reader(pair_source, expect_pairs)
    if len(pair_keys) != expect_pairs:
        raise ValueError(f"pair parquet expected {expect_pairs} unique keys; got {len(pair_keys)}")
    generation_metadata, generation_metadata_path = validate_generation_metadata(
        source_root=source_root,
        source_gen_key=source_gen_key,
        expected_eval_parquet=expected_eval_parquet,
    )

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
        if keys != pair_keys:
            raise ValueError(f"pair parquet key set differs for {cell}")
        if reference_keys is None:
            reference_keys = keys
        elif keys != reference_keys:
            raise ValueError(f"key set differs for {cell}")
        cell_sources[cell] = cell_source
        cell_hashes[cell] = sha256_cell_artifacts(cell_source)
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

        destination_cell_hashes = {
            cell: sha256_cell_artifacts(staged_step / "sweep" / cell / mode) for cell in cells
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
            "generation_metadata_source": str(generation_metadata_path),
            "generation_metadata_sha256": sha256_file(generation_metadata_path),
            "generation_metadata": generation_metadata,
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
    parser.add_argument("--expected-eval-parquet", type=Path, required=True)
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
        expected_eval_parquet=args.expected_eval_parquet,
        mode=args.mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
