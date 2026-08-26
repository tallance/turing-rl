from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_frac10_test50_eval.sh"
STEPS = (0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60)


def _env(tmp_path: Path) -> dict[str, str]:
    eval_root = tmp_path / "results" / "train10pct-test50pct"
    source = tmp_path / "data" / "dataset" / "test.parquet"
    source.parent.mkdir(parents=True)
    source.touch()
    return {
        **os.environ,
        "DRY": "1",
        "REPO": str(ROOT),
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_INPUT_DATA_ROOT": str(tmp_path / "data"),
        "TURING_RL_GENERATED_DATA_ROOT": str(tmp_path / "data"),
        "TURING_RL_RUN_ROOT": str(eval_root),
        "EVAL_ROOT": str(eval_root),
        "SOURCE_EVAL_PARQUET": str(source),
        "EVAL_PARQUET": str(source.parent / "eval_subsets" / "test_seed42_n440.parquet"),
        "PY": sys.executable,
    }


def test_prepare_phase_submits_subset_then_dependent_controller(tmp_path):
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_env(tmp_path),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 2
    assert "--job-name=te_t10t50_prepare" in planned[0]
    assert "scripts/slurm/sample_eval_subset.sh" in planned[0]
    assert "--job-name=te_t10t50_continue" in planned[1]
    assert "--dependency=afterok:" in planned[1]
    assert all(" -- " in line for line in planned)


def test_judge_phase_uses_distinct_names_and_440_pair_sets(tmp_path):
    env = _env(tmp_path)
    env.update({"PHASE": "judge", "SKIP_SPLIT_GUARD": "1"})
    pairs = Path(env["EVAL_ROOT"]) / "raw" / "pairs"
    pairs.mkdir(parents=True)
    for step in STEPS:
        (pairs / f"gen_9b-train10pct-step{step}_440.parquet").touch()

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 2
    assert "--job-name=te_t10t50_qwen35-9b_0" in planned[0]
    assert "gen_9b-train10pct-step0_440.parquet" in planned[0]
    assert "--job-name=te_t10t50_continue" in planned[1]
