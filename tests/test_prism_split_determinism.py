"""Regression test: PRISM two-stage split reproduces paper Table 3 exactly and is deterministic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]

RUN1_BUILD = REPO_ROOT / "data" / "prism" / "full_s42_history"
RUN1_SPLIT = REPO_ROOT / "data" / "prism" / "full_s42_history_sft40_grpo60_test10"
RUN2_BUILD = REPO_ROOT / "data" / "prism" / "full_s42_history_run2"
RUN2_SPLIT = REPO_ROOT / "data" / "prism" / "full_s42_history_sft40_grpo60_test10_run2"

BUILD_FILES = ("train.parquet", "val.parquet", "test.parquet")
SPLIT_FILES = ("sft/train.parquet", "grpo/train.parquet", "grpo/val.parquet", "test.parquet")

EXPECTED_TABLE3 = {
    "sft":        {"users": 464, "rows": 3272},
    "grpo_train": {"users": 696, "rows": 4174},
    "grpo_val":   {"users": 696, "rows": 705},
    "heldout":    {"users": 128, "rows": 880},
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require(path: Path) -> None:
    if not path.exists():
        pytest.skip(
            f"Missing {path}. Run the two Slurm jobs first: "
            f"sbatch scripts/slurm/build_prism_full_s42.sh && "
            f"sbatch scripts/slurm/split_prism_full_s42.sh"
        )


def test_paper_table3_counts_match():
    metadata_path = RUN1_SPLIT / "split_metadata.json"
    _require(metadata_path)
    metadata = json.loads(metadata_path.read_text())

    for split, expected in EXPECTED_TABLE3.items():
        got = metadata["counts"][split]
        assert got == expected, f"{split}: expected {expected}, got {got}"

    assert metadata["user_overlap"] == {"grpo_sft": 0, "grpo_heldout": 0, "sft_heldout": 0}


@pytest.mark.parametrize("relpath", BUILD_FILES)
def test_build_stage_is_byte_deterministic(relpath: str):
    p1, p2 = RUN1_BUILD / relpath, RUN2_BUILD / relpath
    _require(p1)
    _require(p2)
    assert _sha256(p1) == _sha256(p2), f"build parquet differs across runs: {relpath}"


@pytest.mark.parametrize("relpath", SPLIT_FILES)
def test_split_stage_is_byte_deterministic(relpath: str):
    p1, p2 = RUN1_SPLIT / relpath, RUN2_SPLIT / relpath
    _require(p1)
    _require(p2)
    assert _sha256(p1) == _sha256(p2), f"split parquet differs across runs: {relpath}"
