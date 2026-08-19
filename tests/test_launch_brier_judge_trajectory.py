from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_brier_judge_trajectory.sh"
STEPS = (64, 128, 192, 256, 320)


def _env(tmp_path: Path, *, mode: str = "on", confirm_off: bool = False) -> dict[str, str]:
    source_eval = tmp_path / "source-eval"
    pairs = source_eval / "raw" / "pairs"
    pairs.mkdir(parents=True)
    for step in range(0, 321, 32):
        (pairs / f"gen_9b-full5ep-step{step}_880.parquet").write_text(f"pairs-{step}\n")

    baseline = tmp_path / "source-judge" / mode
    (baseline / "reward").mkdir(parents=True)
    (baseline / "reward" / "reward-12345-1.jsonl").write_text('{"pair": 1}\n')
    (baseline / "run_metadata.json").write_text('{"slurm_job_id": "12345"}\n')

    model = tmp_path / "judge" / "hf_dense"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}\n")

    eval_root = tmp_path / ("trajectory-thinking-off" if mode == "off" else "trajectory")
    env = {
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
        "THINKING_MODE": mode,
    }
    if confirm_off:
        env["CONFIRM_THINKING_OFF"] = "1"
    return env


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
    assert (reused / "reward/reward-12345-1.jsonl").read_text() == '{"pair": 1}\n'
    assert (eval_root / "raw/pairs/gen_9b-full5ep-step0_880.parquet").read_text() == "pairs-0\n"

    provenance = json.loads((eval_root / "provenance/baseline_reuse.json").read_text())
    assert provenance["source_slurm_job_id"] == "12345"
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


def test_requires_explicit_judge_model_and_step0_baseline(tmp_path: Path) -> None:
    missing_model_env = _env(tmp_path / "missing-model")
    missing_model_env.pop("MODEL")
    missing_model = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=missing_model_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_model.returncode != 0
    assert "set MODEL" in missing_model.stderr

    missing_baseline_env = _env(tmp_path / "missing-baseline")
    missing_baseline_env.pop("BASELINE_CELL_ROOT")
    missing_baseline = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=missing_baseline_env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert missing_baseline.returncode != 0
    assert "set BASELINE_CELL_ROOT" in missing_baseline.stderr


def test_thinking_off_requires_explicit_confirmation(tmp_path: Path) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_env(tmp_path, mode="off"),
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "CONFIRM_THINKING_OFF=1" in result.stderr
    assert "[DRY] sbatch" not in result.stderr


def test_thinking_off_requires_a_mode_labeled_root(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off", confirm_off=True)
    env["EVAL_ROOT"] = str(tmp_path / "trajectory")
    env["TURING_RL_RUN_ROOT"] = env["EVAL_ROOT"]

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode != 0
    assert "thinking-off" in result.stderr


def test_thinking_off_reuses_only_off_baseline_and_scopes_outputs(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off", confirm_off=True)
    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    reused = Path(env["EVAL_ROOT"]) / "raw/9b-full5ep-step0/sweep/judge-9b-brier/off"
    assert (reused / "reward/reward-12345-1.jsonl").is_file()
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 5
    assert all("THINKING_MODE=off" in line for line in planned)


def test_refuses_cross_mode_baseline(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off", confirm_off=True)
    wrong = Path(env["BASELINE_CELL_ROOT"]).parent / "on"
    Path(env["BASELINE_CELL_ROOT"]).rename(wrong)
    env["BASELINE_CELL_ROOT"] = str(wrong)

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode != 0
    assert "cross-mode baseline" in result.stderr


def test_refuses_duplicate_submission_before_outputs_exist(tmp_path: Path) -> None:
    env = _env(tmp_path)

    first = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )
    second = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "already claimed for submission" in second.stderr
    assert "[DRY] sbatch" not in second.stderr


def test_can_submit_all_ten_nonzero_checkpoints_in_one_off_chain(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off", confirm_off=True)
    env["STEPS"] = "32 64 96 128 160 192 224 256 288 320"

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode == 0, result.stderr
    planned = [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]
    assert len(planned) == 10
    assert sum("--dependency=afterok:" in line for line in planned) == 9
    assert all("THINKING_MODE=off" in line for line in planned)
