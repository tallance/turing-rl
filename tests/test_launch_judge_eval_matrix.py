from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_judge_eval_matrix.sh"


def _env(tmp_path: Path, *, mode: str | None = None, confirm: bool = False) -> dict[str, str]:
    pairs = tmp_path / "pairs.parquet"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_text("pairs\n")

    models: dict[str, str] = {}
    for key in (
        "JUDGE_4B_DIRECTIONAL_MODEL",
        "JUDGE_4B_GRADED_MODEL",
        "JUDGE_9B_DIRECTIONAL_MODEL",
        "JUDGE_9B_GRADED_MODEL",
    ):
        model = tmp_path / key.lower()
        model.mkdir()
        (model / "config.json").write_text("{}\n")
        models[key] = str(model)

    env = {
        **os.environ,
        **models,
        "DRY": "1",
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_RUN_ROOT": str(tmp_path / "thinking-on-eval"),
        "EVAL_ROOT": str(tmp_path / "thinking-on-eval"),
        "PAIRS": str(pairs),
    }
    if mode is not None:
        env["THINKING_MODE"] = mode
    if confirm:
        env["CONFIRM_THINKING_OFF"] = "1"
    return env


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _planned(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]


def test_defaults_to_thinking_on_for_all_nine_comparison_cells(tmp_path: Path) -> None:
    result = _run(_env(tmp_path))

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == 9
    assert all("THINKING_MODE=on" in line for line in planned)
    assert sum("--dependency=afterany:" in line for line in planned) == 8


def test_thinking_off_requires_explicit_confirmation(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off")
    env["EVAL_ROOT"] = str(tmp_path / "thinking-off-eval")
    env["TURING_RL_RUN_ROOT"] = env["EVAL_ROOT"]

    result = _run(env)

    assert result.returncode != 0
    assert "CONFIRM_THINKING_OFF=1" in result.stderr
    assert not _planned(result)


def test_thinking_off_requires_a_mode_labeled_run_root(tmp_path: Path) -> None:
    result = _run(_env(tmp_path, mode="off", confirm=True))

    assert result.returncode != 0
    assert "thinking-off" in result.stderr
    assert not _planned(result)


def test_confirmed_thinking_off_is_mode_scoped_and_complete(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="off", confirm=True)
    env["EVAL_ROOT"] = str(tmp_path / "judge-thinking-off-eval")
    env["TURING_RL_RUN_ROOT"] = env["EVAL_ROOT"]

    result = _run(env)

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == 9
    assert all("THINKING_MODE=off" in line for line in planned)
    for cell in (
        "judge-9b-graded-step52",
        "judge-4b-graded-step52",
        "judge-4b-directional-step52",
        "qwen35-27b",
        "gemma4-31b",
        "gemma4-12b",
        "qwen35-9b",
        "qwen35-4b",
        "judge-9b-directional-step52",
    ):
        assert any(f"CELL_NAME={cell}" in line for line in planned)


def test_refuses_stale_output_in_the_selected_mode(tmp_path: Path) -> None:
    env = _env(tmp_path)
    reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b/on/reward"
    reward.mkdir(parents=True)
    (reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr
    assert not _planned(result)
