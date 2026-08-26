"""Executing guards on the judge pair-build launcher.

scripts/launch_judge_pairs.sh is the only path by which a caller can reach
scripts/slurm/judge_train_gen.sh, which is the only path to
scripts/build_judge_train_pairs.py. Anything the builder needs that the launcher cannot
express is unreachable on the cluster, so these run the launcher under DRY=1 and read the
--export list it would hand to sbatch.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_judge_pairs.sh"


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    env = {
        **os.environ,
        "DRY": "1",
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_GENERATED_DATA_ROOT": str(tmp_path / "generated"),
        "OUT_DIR": str(tmp_path / "pairs"),
    }
    env.update(overrides)
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
    return [line for line in result.stdout.splitlines() if line.startswith("[DRY] ")]


def test_default_style_is_full(tmp_path: Path) -> None:
    """PROMPT_STYLE unset must select the builder's own default, so every pair set built
    before this knob existed is reproduced unchanged."""
    result = _run(_env(tmp_path, SPLITS="train val"))

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == 2
    assert all("PROMPT_STYLE=full" in line for line in planned)
    assert not any("single_token" in line for line in planned)


def test_default_style_leaves_the_pre_existing_export_list_intact(tmp_path: Path) -> None:
    """The style is appended, not woven in: SPLIT and OUT_DIR must keep their exact
    positions and spelling or an existing --env invocation changes meaning."""
    env = _env(tmp_path, SPLITS="train")
    result = _run(env)

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert f"--export=ALL,SPLIT=train,OUT_DIR={env['OUT_DIR']}" in planned


def test_single_token_style_reaches_the_job(tmp_path: Path) -> None:
    result = _run(_env(tmp_path, SPLITS="train", PROMPT_STYLE="single_token"))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "PROMPT_STYLE=single_token" in planned


def test_every_split_carries_the_style(tmp_path: Path) -> None:
    """Both splits feed one training run; a style set on only one of them would pair a
    single-token train set against a full-schema val set."""
    result = _run(_env(tmp_path, SPLITS="train val", PROMPT_STYLE="single_token"))

    assert result.returncode == 0, result.stderr
    planned = _planned(result)
    assert len(planned) == 2
    assert all("PROMPT_STYLE=single_token" in line for line in planned)


def test_unknown_prompt_style_is_rejected(tmp_path: Path) -> None:
    """argparse would catch it inside the job -- after the 12h generation step has already
    burned a GPU node. The launcher has to catch it on the login node instead."""
    result = _run(_env(tmp_path, SPLITS="train", PROMPT_STYLE="freeform"))

    assert result.returncode != 0
    assert "full|single_token" in result.stderr
    assert not _planned(result)


def test_an_empty_prompt_style_falls_back_to_full(tmp_path: Path) -> None:
    """`--env PROMPT_STYLE=` is a plausible slip. ${VAR:-default} substitutes on empty as
    well as unset, so it must land on full rather than reaching argparse as ''."""
    result = _run(_env(tmp_path, SPLITS="train", PROMPT_STYLE=""))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "PROMPT_STYLE=full" in planned
