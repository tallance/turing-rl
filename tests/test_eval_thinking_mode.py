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
