"""The three places that name a sweep cell's output directory must agree.

``scripts/slurm/judge_sweep_cell.sh`` creates it, ``run_judge_sweep_cell.cell_output_dirs``
writes the dumps into it, and ``launch_judge_eval_matrix.sh``'s stale-output guard
inspects it. They disagreed before: the guard checked a style-scoped path while the
writer used a style-less one, so the guard watched a directory nothing wrote and a
single_token rerun appended into the full-schema cell.

The bash block is EXECUTED here, not pattern-matched, and the guard is exercised by
running the real launcher against the directory that block produces.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from scripts.run_judge_sweep_cell import cell_output_dirs

ROOT = Path(__file__).resolve().parents[1]
CELL_SCRIPT = ROOT / "scripts" / "slurm" / "judge_sweep_cell.sh"
LAUNCHER = ROOT / "scripts" / "launch_judge_eval_matrix.sh"

_BLOCK_RE = re.compile(
    r"^# --- BEGIN mode-dir ---$(.*?)^# --- END mode-dir ---$",
    re.MULTILINE | re.DOTALL,
)


def _mode_dir_block() -> str:
    match = _BLOCK_RE.search(CELL_SCRIPT.read_text())
    assert match, "judge_sweep_cell.sh lost its mode-dir markers"
    return match.group(1)


def _mode_dir(sweep_root: str, cell: str, mode: str, style: str) -> str:
    """Run the sbatch script's own MODE_DIR computation and return what it produced."""
    script = (
        f'SWEEP_ROOT={sweep_root}\nCELL_NAME={cell}\nTHINKING_MODE={mode}\n'
        f'JUDGE_PROMPT_STYLE={style}\n{_mode_dir_block()}\nprintf %s "$MODE_DIR"'
    )
    result = subprocess.run(
        ["bash", "-uc", script], text=True, capture_output=True, check=True
    )
    return result.stdout


@pytest.mark.parametrize("style", ["full", "single_token"])
def test_the_sbatch_script_and_the_python_client_agree(tmp_path: Path, style: str) -> None:
    sweep_root = str(tmp_path / "raw" / "sweep")

    from_bash = _mode_dir(sweep_root, "qwen35-4b", "on", style)
    from_python = os.path.dirname(cell_output_dirs(sweep_root, "qwen35-4b", "on", style)["reward"])

    assert from_bash == from_python


def test_full_keeps_the_historical_style_less_path(tmp_path: Path) -> None:
    """Existing result trees have no style segment; adding one would orphan them."""
    sweep_root = str(tmp_path / "raw" / "sweep")

    assert _mode_dir(sweep_root, "qwen35-4b", "on", "full").endswith("/qwen35-4b/on")
    assert _mode_dir(sweep_root, "qwen35-4b", "on", "single_token").endswith(
        "/qwen35-4b/on/single_token"
    )


def _launcher_env(tmp_path: Path, style: str) -> dict[str, str]:
    pairs = tmp_path / "pairs.parquet"
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
    root = tmp_path / ("single-token-eval" if style == "single_token" else "thinking-on-eval")
    return {
        **os.environ,
        **models,
        "DRY": "1",
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_RUN_ROOT": str(root),
        "EVAL_ROOT": str(root),
        "PAIRS": str(pairs),
        "JUDGE_PROMPT_STYLE": style,
    }


@pytest.mark.parametrize("style", ["full", "single_token"])
def test_the_guard_inspects_the_directory_the_cell_actually_writes(
    tmp_path: Path, style: str
) -> None:
    """Round trip: seed stale output at the path the sbatch block computes, then let the
    real launcher decide. If the two ever drift apart the launcher happily proceeds."""
    env = _launcher_env(tmp_path, style)
    sweep_root = os.path.join(env["EVAL_ROOT"], "raw", "sweep")
    mode_dir = Path(_mode_dir(sweep_root, "qwen35-4b", "on", style))
    (mode_dir / "reward").mkdir(parents=True)
    (mode_dir / "reward" / "stale.jsonl").write_text("{}\n")

    result = subprocess.run(
        ["bash", str(LAUNCHER)], env=env, text=True, capture_output=True, check=False
    )

    assert result.returncode != 0, result.stdout
    assert "refusing stale output" in result.stderr


def test_the_cell_script_rejects_an_unknown_style() -> None:
    source = CELL_SCRIPT.read_text()
    assert "JUDGE_PROMPT_STYLE must be full|single_token" in source
    # The client is told explicitly rather than relying on --export=ALL inheritance.
    assert '--prompt_style "$JUDGE_PROMPT_STYLE"' in source


def test_the_launcher_exports_the_style_to_the_cell() -> None:
    assert "JUDGE_PROMPT_STYLE=$JUDGE_PROMPT_STYLE" in LAUNCHER.read_text()
