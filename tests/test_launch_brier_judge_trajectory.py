from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_brier_judge_trajectory.sh"
STEPS = (64, 128, 192, 256, 320)


def _env(tmp_path: Path) -> dict[str, str]:
    source_eval = tmp_path / "source-eval"
    pairs = source_eval / "raw" / "pairs"
    pairs.mkdir(parents=True)
    for step in (0, *STEPS):
        (pairs / f"gen_9b-full5ep-step{step}_880.parquet").write_text(f"pairs-{step}\n")

    baseline = tmp_path / "source-judge" / "on"
    (baseline / "reward").mkdir(parents=True)
    (baseline / "reward" / "reward-18447-1.jsonl").write_text('{"pair": 1}\n')
    (baseline / "run_metadata.json").write_text('{"slurm_job_id": "18447"}\n')

    model = tmp_path / "judge" / "hf_dense"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")

    eval_root = tmp_path / "trajectory"
    return {
        **os.environ,
        "DRY": "1",
        "PY": sys.executable,
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_RUN_ROOT": str(eval_root),
        "EVAL_ROOT": str(eval_root),
        "SOURCE_EVAL_ROOT": str(source_eval),
        "BASELINE_CELL_ROOT": str(baseline),
        "MODEL": str(model),
    }


def test_submits_only_epoch_boundaries_in_one_afterok_chain(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 5
    assert [f"step{step}_880.parquet" in line for step, line in zip(STEPS, planned)] == [True] * 5
    assert all("CELL_NAME=judge-9b-brier" in line for line in planned)
    assert all("TP=1,REPLICAS=8" in line for line in planned)
    assert all("scripts/slurm/judge_sweep_cell.sh" in line for line in planned)
    assert sum("--dependency=afterok:" in line for line in planned) == 4
    assert not any("step0_880.parquet" in line for line in planned)


def test_reuses_step0_and_records_exact_source_hashes(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    eval_root = Path(env["EVAL_ROOT"])
    reused = eval_root / "raw/9b-full5ep-step0/sweep/judge-9b-brier/on"
    assert (reused / "reward/reward-18447-1.jsonl").read_text() == '{"pair": 1}\n'
    assert (eval_root / "raw/pairs/gen_9b-full5ep-step0_880.parquet").read_text() == "pairs-0\n"

    provenance = json.loads((eval_root / "provenance/baseline_reuse.json").read_text())
    assert provenance["source_slurm_job_id"] == "18447"
    assert provenance["source_cell_root"] == env["BASELINE_CELL_ROOT"]
    assert len(provenance["source_tree_sha256"]) == 64
    assert len(provenance["pair_sha256"]) == 64


def test_refuses_stale_scoring_output(tmp_path: Path) -> None:
    env = _env(tmp_path)
    stale = (
        Path(env["EVAL_ROOT"])
        / "raw/9b-full5ep-step64/sweep/judge-9b-brier/on/reward"
    )
    stale.mkdir(parents=True)
    (stale / "old.jsonl").write_text("stale\n")

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr


def test_refuses_a_missing_model_or_epoch_pair_set(tmp_path: Path) -> None:
    env = _env(tmp_path)
    model = Path(env["MODEL"])
    for path in model.iterdir():
        path.unlink()
    model.rmdir()
    missing_model = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )
    assert missing_model.returncode != 0
    assert "missing trained judge model" in missing_model.stderr

    env = _env(tmp_path / "pairs-case")
    Path(env["SOURCE_EVAL_ROOT"], "raw/pairs/gen_9b-full5ep-step192_880.parquet").unlink()
    missing_pairs = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )
    assert missing_pairs.returncode != 0
    assert "missing pair set" in missing_pairs.stderr
