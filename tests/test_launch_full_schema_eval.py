from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_full_schema_eval.sh"


def _dry_env(tmp_path: Path, *, offset: int = 0, batch_size: int = 8) -> dict[str, str]:
    eval_root = tmp_path / "eval"
    pairs = eval_root / "raw" / "pairs"
    pairs.mkdir(parents=True)
    for step in (0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320):
        (pairs / f"gen_9b-full5ep-step{step}_880.parquet").touch()
    return {
        **os.environ,
        "DRY": "1",
        "SKIP_SPLIT_GUARD": "1",
        "REPO": str(ROOT),
        "PY": sys.executable,
        "EVAL_ROOT": str(eval_root),
        "OFFSET": str(offset),
        "BATCH_SIZE": str(batch_size),
    }


def test_first_batch_is_qwen9_and_queue_bounded(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_dry_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 9  # eight GPU cells plus one continuation
    assert all("qwen35-9b" in line for line in planned[:8])
    assert "NEXT_OFFSET=8" in planned[-1]
    assert sum("--dependency=afterok:" in line for line in planned) == 8
    assert all(" -- " in line for line in planned)


def test_model_major_boundary_is_qwen9_then_gemma12(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_dry_env(tmp_path, offset=8, batch_size=8),
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    cells = planned[:8]
    assert sum("qwen35-9b" in line for line in cells) == 3
    assert sum("gemma4-12b" in line for line in cells) == 5
    assert all("CONCURRENCY=32" in line for line in cells[:3])
    assert all("CONCURRENCY=4" in line for line in cells[3:])


def test_skip_split_guard_is_dry_run_only(tmp_path):
    env = _dry_env(tmp_path)
    env["DRY"] = "0"
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert "allowed only with DRY=1" in result.stderr
