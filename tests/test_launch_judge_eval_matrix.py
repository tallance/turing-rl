from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_judge_eval_matrix.sh"

GRPO_MODEL_KEYS = (
    "JUDGE_4B_DIRECTIONAL_MODEL",
    "JUDGE_4B_GRADED_MODEL",
    "JUDGE_9B_DIRECTIONAL_MODEL",
    "JUDGE_9B_GRADED_MODEL",
)
CE_MODEL_KEYS = ("JUDGE_4B_CE_MODEL", "JUDGE_9B_CE_MODEL", "JUDGE_GEMMA12B_CE_MODEL")


def _make_models(tmp_path: Path, keys: tuple[str, ...]) -> dict[str, str]:
    """Materialise one complete (config.json present) model directory per env var key."""
    models: dict[str, str] = {}
    for key in keys:
        model = tmp_path / key.lower()
        model.mkdir(exist_ok=True)
        (model / "config.json").write_text("{}\n")
        models[key] = str(model)
    return models


def _env(tmp_path: Path, *, mode: str | None = None, confirm: bool = False) -> dict[str, str]:
    pairs = tmp_path / "pairs.parquet"
    pairs.parent.mkdir(parents=True, exist_ok=True)
    pairs.write_text("pairs\n")

    models = _make_models(tmp_path, GRPO_MODEL_KEYS)

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
    # This fixture stands for a default invocation, and it inherits os.environ. An ambient
    # CELLS would silently narrow every matrix a test asserts on.
    env.pop("CELLS", None)
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


def _single_token_env(tmp_path: Path, *, leaf: str = "2026-08-26-single-token-judge-thinking-off") -> dict[str, str]:
    """The one accepted single_token shape: thinking-off, confirmed, both constraints named,
    and the two CE checkpoints present. Callers break exactly the thing under test."""
    env = _env(tmp_path, mode="off", confirm=True)
    env["JUDGE_PROMPT_STYLE"] = "single_token"
    env.update(_make_models(tmp_path, CE_MODEL_KEYS))
    _set_eval_root(env, tmp_path, leaf)
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


def _planned_matrix(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Reconstruct the resolved MATRIX, in submission order, from the planned sbatch lines.

    One row per cell as ``"<cell> <model> <tp> <replicas> <concurrency>"`` -- the same shape
    as a MATRIX heredoc line, so the expected value in a test reads as the literal table.
    """
    rows: list[str] = []
    for line in _planned(result):
        _, _, exports = line.partition("--export=")
        assert exports, f"planned line carries no --export: {line}"
        fields = dict(
            item.split("=", 1) for item in exports.split(" ", 1)[0].split(",") if "=" in item
        )
        rows.append(
            " ".join(
                fields[key]
                for key in ("CELL_NAME", "MODEL", "TP", "REPLICAS", "CONCURRENCY")
            )
        )
    return rows


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
    env = _single_token_env(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == len(SINGLE_TOKEN_CELLS)
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
    env = _single_token_env(tmp_path)

    old_reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b-st/off/reward"
    old_reward.mkdir(parents=True)
    (old_reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == len(SINGLE_TOKEN_CELLS)


def test_single_token_style_reward_dir_is_itself_guarded(tmp_path: Path) -> None:
    """The other half of the fold-in: stale output under the NEW style-scoped path must
    still be caught, or a retried single_token cell could silently append to old output."""
    env = _single_token_env(tmp_path)

    new_reward = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b-st/off/single_token/reward"
    new_reward.mkdir(parents=True)
    (new_reward / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr
    assert not _planned(result)


def test_full_arm_matrix_is_the_literal_nine_rows(tmp_path: Path) -> None:
    """The full arm is the pre-existing behaviour and must not drift. Pinning the resolved
    table -- cells, models, TP, replicas, concurrency, and submission order -- means a later
    edit to the single_token arm cannot quietly reshape this one."""
    env = _env(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == [
        f"judge-9b-graded-step52 {env['JUDGE_9B_GRADED_MODEL']} 1 8 32",
        f"judge-4b-graded-step52 {env['JUDGE_4B_GRADED_MODEL']} 1 8 32",
        f"judge-4b-directional-step52 {env['JUDGE_4B_DIRECTIONAL_MODEL']} 1 8 32",
        "qwen35-27b Qwen/Qwen3.5-27B 8 1 32",
        "gemma4-31b google/gemma-4-31B-it 8 1 4",
        "gemma4-12b google/gemma-4-12B-it 1 8 4",
        "qwen35-9b Qwen/Qwen3.5-9B 1 8 32",
        "qwen35-4b Qwen/Qwen3.5-4B 1 8 32",
        f"judge-9b-directional-step52 {env['JUDGE_9B_DIRECTIONAL_MODEL']} 1 8 32",
    ]


def test_single_token_arm_matrix_is_the_literal_seven_rows(tmp_path: Path) -> None:
    """The seven cells the design calls for. The zero-shot serving shapes are copied from
    the full arm (gemma4-31b at TP=8 and concurrency 4 is a node constraint, not a protocol
    one) and the two CE rows mirror the full arm's trained 4B/9B rows. Every cell name ends
    in -st: the arms merge into one table keyed by cell name plus a prompt_style column."""
    env = _single_token_env(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == [
        f"judge-9b-ce-st {env['JUDGE_9B_CE_MODEL']} 1 8 32",
        f"judge-4b-ce-st {env['JUDGE_4B_CE_MODEL']} 1 8 32",
        # Concurrency 4, not 32: copied from the zero-shot gemma4-12b row, because it is a
        # serving constraint of the model rather than of the training.
        f"judge-gemma12b-ce-st {env['JUDGE_GEMMA12B_CE_MODEL']} 1 8 4",
        "qwen35-27b-st Qwen/Qwen3.5-27B 8 1 32",
        "gemma4-31b-st google/gemma-4-31B-it 8 1 4",
        "gemma4-12b-st google/gemma-4-12B-it 1 8 4",
        "qwen35-9b-st Qwen/Qwen3.5-9B 1 8 32",
        "qwen35-4b-st Qwen/Qwen3.5-4B 1 8 32",
    ]


def test_the_two_arms_share_no_cell_name(tmp_path: Path) -> None:
    """prompt_style is a column in the merged table, not part of the cell name, so a name
    used by both arms would leave that row unattributable to an arm by name alone."""
    full = _run(_env(tmp_path / "full"))
    single = _run(_single_token_env(tmp_path / "single"))

    assert full.returncode == 0, full.stderr
    assert single.returncode == 0, single.stderr
    full_cells = {row.split(" ", 1)[0] for row in _planned_matrix(full)}
    single_cells = {row.split(" ", 1)[0] for row in _planned_matrix(single)}
    assert full_cells and single_cells
    assert full_cells.isdisjoint(single_cells)


def test_full_arm_checks_exactly_the_four_grpo_models(tmp_path: Path) -> None:
    """Breaking any one of the four must stop the submission, naming the offender. Each
    iteration gets its own EVAL_ROOT so a run that got far enough to claim one cannot make
    the next iteration fail for the wrong reason."""
    for index, key in enumerate(GRPO_MODEL_KEYS):
        env = _env(tmp_path)
        root = str(tmp_path / f"thinking-on-eval-{index}")
        env["EVAL_ROOT"] = root
        env["TURING_RL_RUN_ROOT"] = root
        missing = tmp_path / f"absent-{key.lower()}"
        env[key] = str(missing)
        assert not missing.exists()

        result = _run(env)

        assert result.returncode != 0, f"{key} unchecked: {result.stderr}"
        assert "trained judge model is incomplete" in result.stderr
        assert str(missing) in result.stderr
        assert not _planned(result)


def test_full_arm_does_not_require_the_ce_models(tmp_path: Path) -> None:
    """The CE checkpoints belong to the other arm and are never served here, so their
    absence must not block a full-schema submission."""
    env = _env(tmp_path)
    for key in CE_MODEL_KEYS:
        absent = tmp_path / f"absent-{key.lower()}"
        env[key] = str(absent)
        assert not absent.exists()

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == 9
    assert not any("-ce-st" in line for line in _planned(result))


def test_single_token_arm_does_not_require_the_grpo_models(tmp_path: Path) -> None:
    """The four GRPO judges are the full arm's; a single_token run neither names nor serves
    them. Dropping the overrides entirely leaves the launcher's hardcoded cluster defaults,
    which do not exist on this machine -- so the run can only succeed if they go unchecked."""
    env = _single_token_env(tmp_path)
    for key in GRPO_MODEL_KEYS:
        env.pop(key, None)
    for default in (
        "/home/lancewicki/projects/turing-rl/results/2026-08-14-judge-4b-eval-v2",
        "/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval",
    ):
        assert not Path(default).exists(), f"cluster path present locally, test is vacuous: {default}"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == len(SINGLE_TOKEN_CELLS)


def test_single_token_arm_checks_both_ce_models(tmp_path: Path) -> None:
    """The mirror of the full arm's check: a missing CE checkpoint must fail at submit
    rather than at the cell, and must name the path it looked for."""
    for index, key in enumerate(CE_MODEL_KEYS):
        env = _single_token_env(
            tmp_path, leaf=f"2026-08-26-single-token-judge-thinking-off-{index}"
        )
        missing = tmp_path / f"absent-{key.lower()}"
        env[key] = str(missing)
        assert not missing.exists()

        result = _run(env)

        assert result.returncode != 0, f"{key} unchecked: {result.stderr}"
        assert "trained judge model is incomplete" in result.stderr
        assert str(missing) in result.stderr
        assert not _planned(result)


# --- CELLS: submitting a subset of the arm's matrix -------------------------------------

FULL_CELLS = (
    "judge-9b-graded-step52",
    "judge-4b-graded-step52",
    "judge-4b-directional-step52",
    "qwen35-27b",
    "gemma4-31b",
    "gemma4-12b",
    "qwen35-9b",
    "qwen35-4b",
    "judge-9b-directional-step52",
)
SINGLE_TOKEN_CELLS = (
    "judge-9b-ce-st",
    "judge-4b-ce-st",
    "judge-gemma12b-ce-st",
    "qwen35-27b-st",
    "gemma4-31b-st",
    "gemma4-12b-st",
    "qwen35-9b-st",
    "qwen35-4b-st",
)


def _cell_names(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [row.split(" ", 1)[0] for row in _planned_matrix(result)]


def _claim_metadata(env: dict[str, str]) -> str:
    claim = Path(env["EVAL_ROOT"]) / "provenance/judge_eval_matrix_submission.claim"
    return (claim / "metadata.txt").read_text()


def test_cells_unset_plans_the_whole_matrix_for_both_styles(tmp_path: Path) -> None:
    """The property that matters most: the default is unchanged. Both arms plan their whole
    matrix, in matrix order, and an explicitly empty CELLS means "no selection" rather than
    "select nothing"."""
    full_env = _env(tmp_path / "full")
    single_env = _single_token_env(tmp_path / "single")
    assert "CELLS" not in full_env and "CELLS" not in single_env

    full = _run(full_env)
    single = _run(single_env)

    assert full.returncode == 0, full.stderr
    assert single.returncode == 0, single.stderr
    assert _cell_names(full) == list(FULL_CELLS)
    assert _cell_names(single) == list(SINGLE_TOKEN_CELLS)

    empty_full = _run({**_env(tmp_path / "full2"), "CELLS": ""})
    empty_single = _run({**_single_token_env(tmp_path / "single2"), "CELLS": "  "})

    assert empty_full.returncode == 0, empty_full.stderr
    assert empty_single.returncode == 0, empty_single.stderr
    assert _cell_names(empty_full) == list(FULL_CELLS)
    assert _cell_names(empty_single) == list(SINGLE_TOKEN_CELLS)


def test_cells_with_one_name_plans_exactly_that_cell(tmp_path: Path) -> None:
    """The one-cell serving smoke. A lone cell has nothing to chain behind, so it must carry
    no dependency."""
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b-st"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == ["qwen35-4b-st Qwen/Qwen3.5-4B 1 8 32"]
    assert "--dependency=" not in _planned(result)[0]
    assert "THINKING_MODE=off" in _planned(result)[0]
    assert "JUDGE_PROMPT_STYLE=single_token" in _planned(result)[0]


def test_cells_selects_within_the_full_arm_too(tmp_path: Path) -> None:
    """CELLS is a property of the launcher, not of one arm. Dropping the GRPO overrides
    leaves the hardcoded cluster defaults, which do not exist here, so this also shows the
    model check following the selection rather than the arm."""
    env = _env(tmp_path)
    env["CELLS"] = "gemma4-12b qwen35-27b"
    for key in GRPO_MODEL_KEYS:
        env.pop(key, None)
    for default in (
        "/home/lancewicki/projects/turing-rl/results/2026-08-14-judge-4b-eval-v2",
        "/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval",
    ):
        assert not Path(default).exists(), f"cluster path present locally, test is vacuous: {default}"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == [
        "qwen35-27b Qwen/Qwen3.5-27B 8 1 32",
        "gemma4-12b google/gemma-4-12B-it 1 8 4",
    ]


def test_cells_keeps_matrix_order_and_serving_shape(tmp_path: Path) -> None:
    """The caller picks which cells run, never their order or their shape. Requested here in
    reverse matrix order and with a duplicate, so a filter that walked the caller's list
    instead of the matrix would plan a visibly different chain. gemma4-31b-st must keep its
    TP=8 / concurrency=4 node constraint."""
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b-st gemma4-31b-st judge-9b-ce-st gemma4-31b-st"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == [
        f"judge-9b-ce-st {env['JUDGE_9B_CE_MODEL']} 1 8 32",
        "gemma4-31b-st google/gemma-4-31B-it 8 1 4",
        "qwen35-4b-st Qwen/Qwen3.5-4B 1 8 32",
    ]
    assert sum("--dependency=afterok:" in line for line in _planned(result)) == 2


def test_cells_rejects_a_name_that_is_not_in_the_resolved_matrix(tmp_path: Path) -> None:
    """A typo must be fatal. Submitting the recognised subset and dropping the rest would
    read as a successful partial run, which is far harder to notice than a refusal."""
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b-st qwen35-4b-ts"

    result = _run(env)

    assert result.returncode != 0
    assert "unknown cell 'qwen35-4b-ts'" in result.stderr
    assert not _planned(result)
    # The valid names are listed, so the caller can fix the typo from the message alone.
    valid = result.stderr.partition("valid cells:")[2]
    assert valid, result.stderr
    for cell in SINGLE_TOKEN_CELLS:
        assert cell in valid


def test_cells_rejects_a_name_belonging_to_the_other_arm(tmp_path: Path) -> None:
    """The matrix is resolved per style before CELLS is applied, so the full arm's
    ``qwen35-4b`` is simply not a cell of the single_token arm."""
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b"

    result = _run(env)

    assert result.returncode != 0
    assert "unknown cell 'qwen35-4b'" in result.stderr
    assert "JUDGE_PROMPT_STYLE=single_token" in result.stderr
    assert not _planned(result)


def test_a_zero_shot_cell_under_single_token_does_not_require_the_ce_models(
    tmp_path: Path,
) -> None:
    """The point of the change for the serving smoke: a zero-shot cell serves a HuggingFace
    id, so it must be submittable before this arm's CE checkpoints have been trained."""
    env = _single_token_env(tmp_path)
    for key in CE_MODEL_KEYS:
        absent = tmp_path / f"absent-{key.lower()}"
        env[key] = str(absent)
        assert not absent.exists()
    env["CELLS"] = "qwen35-4b-st"

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert _planned_matrix(result) == ["qwen35-4b-st Qwen/Qwen3.5-4B 1 8 32"]


def test_a_selected_trained_cell_still_requires_its_own_checkpoint(tmp_path: Path) -> None:
    """The other half: the check is scoped to the selected cells, not switched off. Selecting
    the 9B CE cell must still fail on a missing 9B CE checkpoint -- while an absent 4B CE
    checkpoint, whose cell is not selected, must not interfere."""
    env = _single_token_env(tmp_path)
    absent_4b = tmp_path / "absent-judge-4b-ce"
    missing_9b = tmp_path / "absent-judge-9b-ce"
    env["JUDGE_4B_CE_MODEL"] = str(absent_4b)
    env["JUDGE_9B_CE_MODEL"] = str(missing_9b)
    assert not absent_4b.exists() and not missing_9b.exists()
    env["CELLS"] = "judge-9b-ce-st"

    result = _run(env)

    assert result.returncode != 0
    assert "trained judge model is incomplete" in result.stderr
    assert str(missing_9b) in result.stderr
    assert str(absent_4b) not in result.stderr
    assert not _planned(result)


def test_cells_scopes_the_stale_output_guard_to_the_selected_cells(tmp_path: Path) -> None:
    """Output belonging to a cell this submission does not run cannot collide with it, so it
    must not block the subset."""
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b-st"
    unselected = Path(env["EVAL_ROOT"]) / "raw/sweep/gemma4-12b-st/off/single_token/reward"
    unselected.mkdir(parents=True)
    (unselected / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == 1


def test_cells_still_guards_stale_output_for_a_selected_cell(tmp_path: Path) -> None:
    env = _single_token_env(tmp_path)
    env["CELLS"] = "qwen35-4b-st"
    selected = Path(env["EVAL_ROOT"]) / "raw/sweep/qwen35-4b-st/off/single_token/reward"
    selected.mkdir(parents=True)
    (selected / "stale.jsonl").write_text("{}\n")

    result = _run(env)

    assert result.returncode != 0
    assert "refusing stale output" in result.stderr
    assert not _planned(result)


def test_a_subset_claims_the_whole_run_root(tmp_path: Path) -> None:
    """Deliberate, and the reason a smoke needs its own run root: the claim answers "has a
    matrix been submitted into this root". A one-cell smoke followed by the whole matrix into
    the same root would re-run the smoke cell and append to its JSONL, so the second
    submission is refused."""
    env = _single_token_env(tmp_path)

    smoke = _run({**env, "CELLS": "qwen35-4b-st"})
    whole = _run(env)

    assert smoke.returncode == 0, smoke.stderr
    assert len(_planned(smoke)) == 1
    assert whole.returncode != 0
    assert "already claimed for submission" in whole.stderr
    assert not _planned(whole)


def test_the_claim_records_the_cell_selection(tmp_path: Path) -> None:
    """A subset run root must not later be mistaken for a whole-matrix one."""
    subset_env = _single_token_env(tmp_path / "subset")
    subset_env["CELLS"] = "qwen35-4b-st"
    whole_env = _single_token_env(tmp_path / "whole")

    subset = _run(subset_env)
    whole = _run(whole_env)

    assert subset.returncode == 0, subset.stderr
    assert whole.returncode == 0, whole.stderr
    assert "cells=qwen35-4b-st\n" in _claim_metadata(subset_env)
    assert "cells=all\n" in _claim_metadata(whole_env)
