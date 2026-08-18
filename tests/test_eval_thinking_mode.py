"""Evaluation launchers must let the caller choose the thinking mode, and keep it.

Judges trained thinking-OFF have to be scored OFF and judges trained ON scored ON, with one
inference policy shared by every model inside a comparison. Both launchers previously pinned
`THINKING_MODE=on`, which silently makes any OFF-trained judge a cross-mode measurement.
"""

from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
LAUNCHERS = ("launch_full_schema_eval.sh", "launch_brier_judge_trajectory.sh")


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_does_not_pin_thinking_mode(name):
    text = (SCRIPTS / name).read_text()

    assert "THINKING_MODE=on," not in text, f"{name} pins the mode in its sbatch --export"
    assert "THINKING_MODE=${THINKING_MODE:-on}" in text, f"{name} must accept a caller override"


@pytest.mark.parametrize("name", LAUNCHERS)
def test_launcher_rejects_a_bogus_mode(name):
    text = (SCRIPTS / name).read_text()

    assert 'case "$THINKING_MODE" in on|off)' in text, f"{name} must validate the mode"


@pytest.mark.parametrize("name", LAUNCHERS)
def test_output_paths_are_mode_scoped(name):
    """judge_sweep_cell.sh writes to $CELL_NAME/$THINKING_MODE; the stale-output guard must agree.

    A guard still checking `/on/` would wave through a rerun that then collides with, or is
    silently mistaken for, the other family's results.
    """
    text = (SCRIPTS / name).read_text()

    assert "/on/reward" not in text, f"{name} has a mode-blind stale-output guard"
    assert "$THINKING_MODE/reward" in text


def test_full_schema_eval_exports_the_mode_to_its_continuation():
    """continue_full_schema_eval.sh re-invokes the launcher with only OFFSET overridden.

    Everything else arrives via --export=ALL, so a set-but-unexported mode reverts to the default
    after the first batch and splits one sweep across two thinking modes.
    """
    text = (SCRIPTS / "launch_full_schema_eval.sh").read_text()
    export_line = next(
        line for line in text.splitlines() if line.startswith("export ") and "JOB_PREFIX" in line
    )

    assert "THINKING_MODE" in export_line


def test_brier_trajectory_refuses_a_cross_mode_baseline():
    """The step-0 cell is copied in rather than recomputed, so its mode must match the sweep's."""
    text = (SCRIPTS / "launch_brier_judge_trajectory.sh").read_text()

    assert 'baseline_mode=$(basename "$BASELINE_CELL_ROOT")' in text
    assert '[ "$baseline_mode" = "$THINKING_MODE" ]' in text


def test_brier_trajectory_copies_the_baseline_into_a_mode_scoped_directory():
    text = (SCRIPTS / "launch_brier_judge_trajectory.sh").read_text()

    assert 'cell / os.environ["THINKING_MODE"]' in text, "baseline copy destination is mode-blind"


def test_no_launcher_anywhere_pins_the_thinking_mode():
    """Repo-wide guard, so this cannot be reintroduced by a launcher that does not exist yet.

    `scripts/launch_judge_eval.sh` currently pins THINKING_MODE=on, but it lives on the fork's
    branch (worktree-judge-4b-eval) and is theirs to change -- editing it here would manufacture
    the same-file conflict CLAUDE.md asks us to avoid. See
    docs/agent-comms/2026-08-12-judge-only-rlvr/handoff-eval-thinking-mode.md. This test enforces
    the requirement automatically the moment that file reaches a shared branch, rather than
    relying on the handoff note being read.
    """
    offenders = [
        path.name
        for path in sorted(SCRIPTS.glob("launch_*.sh"))
        if "THINKING_MODE=on," in path.read_text()
    ]

    assert not offenders, (
        f"these launchers pin the thinking mode: {offenders}. Evaluation mode must be a "
        "caller-supplied parameter -- a judge trained thinking-OFF scored ON is a cross-mode "
        "measurement."
    )
