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


def _set_eval_root(env: dict[str, str], tmp_path: Path, leaf: str) -> None:
    """Point the launcher at ``tmp_path/leaf``.

    Two hazards this exists to close.

    The guard substrings must come from ``leaf`` alone. pytest derives ``tmp_path`` from the
    test function name, so a test named ..._single_token_... yields a path that can satisfy a
    "must name the style" check on its own and turn a negative test tautological; assert the
    ambient path is inert instead of trusting the naming convention.

    And the root must stay *writable*. Under an unwritable base the launcher dies at the
    ``mkdir -p $EVAL_ROOT/provenance`` claim, so a rejection test would pass even with the
    guard removed. A writable base means the only reason a guard test can fail is the guard.
    """
    for substring in ("single-token", "thinking-off"):
        assert substring not in str(tmp_path), (
            f"ambient tmp_path already contains {substring!r}: {tmp_path}"
        )
    root = str(tmp_path / leaf)
    env["EVAL_ROOT"] = root
    env["TURING_RL_RUN_ROOT"] = root


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
    assert all("JUDGE_PROMPT_STYLE=full" in line for line in planned)
    assert sum("--dependency=afterok:" in line for line in planned) == 8
    assert not any("--dependency=afterany:" in line for line in planned)


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


def test_refuses_duplicate_submission_before_outputs_exist(tmp_path: Path) -> None:
    env = _env(tmp_path)

    first = _run(env)
    second = _run(env)

    assert first.returncode == 0, first.stderr
    assert second.returncode != 0
    assert "already claimed for submission" in second.stderr
    assert not _planned(second)


def test_single_token_style_requires_a_matching_eval_root(tmp_path: Path) -> None:
    """A single-token cell written into a full-schema EVAL_ROOT would be silently mistaken
    for a full-schema number when the comparison table is assembled."""
    env = _env(tmp_path, mode="off", confirm=True)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-thinking-off-eval")

    result = _run(env)

    assert result.returncode != 0
    assert "single_token" in result.stderr
    assert not _planned(result)


def test_single_token_style_rejects_the_default_thinking_on_mode(tmp_path: Path) -> None:
    """THINKING_MODE defaults to on, but the single-token judge always serves with thinking
    disabled. Accepting the default would label the mode directory, the provenance record and
    the submission line "on" for a cell that ran thinking-off."""
    env = _env(tmp_path)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-single-token-judge")
    assert "THINKING_MODE" not in env

    result = _run(env)

    assert result.returncode != 0
    assert "JUDGE_PROMPT_STYLE=single_token" in result.stderr
    assert "THINKING_MODE=on" in result.stderr
    assert not _planned(result)


def test_single_token_style_rejects_an_explicit_thinking_on_mode(tmp_path: Path) -> None:
    env = _env(tmp_path, mode="on")
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-single-token-judge")

    result = _run(env)

    assert result.returncode != 0
    assert "requires THINKING_MODE=off" in result.stderr
    assert not _planned(result)


def test_single_token_style_accepts_confirmed_thinking_off(tmp_path: Path) -> None:
    """The only accepted single-token submission: thinking-off, confirmed, and named for both
    constraints."""
    env = _env(tmp_path, mode="off", confirm=True)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-single-token-judge-thinking-off")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == 9
    assert all("THINKING_MODE=off" in line for line in planned)
    assert all("JUDGE_PROMPT_STYLE=single_token" in line for line in planned)


def test_unknown_prompt_style_is_rejected(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env["JUDGE_PROMPT_STYLE"] = "freeform"

    result = _run(env)

    assert result.returncode != 0
    assert "full|single_token" in result.stderr
    assert not _planned(result)


def test_default_prompt_style_reward_dir_is_unchanged(tmp_path: Path) -> None:
    """JUDGE_PROMPT_STYLE unset must stay byte-identical to the pre-existing path: the
    stale-output pre-check must still key off $SWEEP_ROOT/$cell/$THINKING_MODE/reward with
    no style segment folded in, or every existing cell path would be silently orphaned."""
    env = _env(tmp_path)
    reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b/on/reward"
    reward.mkdir(parents=True)
    (reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr
    assert "single_token" not in result.stderr
    assert not _planned(result)


def test_single_token_style_does_not_collide_with_the_full_schema_reward_dir(tmp_path: Path) -> None:
    """The style must be folded into the per-cell reward directory: stale output sitting
    under the OLD style-less path must not block a single_token run against the same
    EVAL_ROOT/THINKING_MODE/cell (they no longer share a directory)."""
    env = _env(tmp_path, mode="off", confirm=True)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-single-token-judge-thinking-off")

    old_reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b/off/reward"
    old_reward.mkdir(parents=True)
    (old_reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == 9


def test_single_token_style_reward_dir_is_itself_guarded(tmp_path: Path) -> None:
    """The other half of the fold-in: stale output under the NEW style-scoped path must
    still be caught, or a retried single_token cell could silently append to old output."""
    env = _env(tmp_path, mode="off", confirm=True)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    _set_eval_root(env, tmp_path, "2026-08-26-single-token-judge-thinking-off")

    new_reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b/off/single_token/reward"
    new_reward.mkdir(parents=True)
    (new_reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr
    assert not _planned(result)
