"""Submission-shape guards for the single-token trajectory launcher.

Every failure guarded here is silent at run time: the jobs succeed and a plausible curve
comes out the other end. Scoring the wrong checkpoint, or scoring the full schema into a
directory named single_token, does not raise anywhere downstream.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_single_token_trajectory.sh"

PREFIX = "9b-train10pct-step"
TAG = "440"


def _pair(root: Path, step: int, body: str = "pairs\n") -> Path:
    path = root / "raw" / "pairs" / f"gen_{PREFIX}{step}_{TAG}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def _model(tmp_path: Path, name: str = "ce_dense") -> Path:
    model = tmp_path / name
    model.mkdir(parents=True, exist_ok=True)
    (model / "config.json").write_text("{}\n")
    return model


def _run(tmp_path: Path, *, steps: str, roots: list[Path], judges: str | None = None,
         eval_root: Path | None = None) -> subprocess.CompletedProcess[str]:
    run_root = eval_root or (tmp_path / "run")
    env = {
        **os.environ,
        "DRY": "1",
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "TURING_RL_RUN_ROOT": str(run_root),
        "EVAL_ROOT": str(run_root),
        "SOURCE_ROOTS": " ".join(str(r) for r in roots),
        "STEPS": steps,
    }
    if judges is not None:
        env["JUDGES"] = judges
    return subprocess.run(
        ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True, cwd=str(ROOT)
    )


def _submissions(result: subprocess.CompletedProcess[str]) -> list[str]:
    return [line for line in result.stderr.splitlines() if line.startswith("[DRY] sbatch")]


def test_each_step_resolves_to_whichever_source_root_holds_it(tmp_path: Path) -> None:
    # The real trajectory is split across two roots: the original run and its extension.
    early, late = tmp_path / "early", tmp_path / "late"
    _pair(early, 0)
    _pair(late, 72)
    model = _model(tmp_path)

    result = _run(tmp_path, steps="0 72", roots=[early, late], judges=f"only|{model}")
    assert result.returncode == 0, result.stderr

    manifest = (tmp_path / "run" / "provenance" / "pair_sources.psv").read_text().splitlines()
    assert manifest[0] == "step|source|sha256"
    assert str(early) in manifest[1] and manifest[1].startswith("0|")
    assert str(late) in manifest[2] and manifest[2].startswith("72|")


def test_a_step_no_root_provides_is_fatal(tmp_path: Path) -> None:
    early = tmp_path / "early"
    _pair(early, 0)
    model = _model(tmp_path)

    result = _run(tmp_path, steps="0 12", roots=[early], judges=f"only|{model}")
    assert result.returncode == 2
    assert "no source root provides step 12" in result.stderr
    assert not _submissions(result), "must not submit a partial trajectory"


def test_the_same_step_in_two_roots_with_different_content_is_fatal(tmp_path: Path) -> None:
    # "First root wins" would silently score one checkpoint under another's label.
    early, late = tmp_path / "early", tmp_path / "late"
    _pair(early, 24, body="from the original run\n")
    _pair(late, 24, body="from the extension\n")
    model = _model(tmp_path)

    result = _run(tmp_path, steps="24", roots=[early, late], judges=f"only|{model}")
    assert result.returncode == 2
    assert "exists in two source roots with different content" in result.stderr


def test_a_byte_identical_step_in_two_roots_is_accepted(tmp_path: Path) -> None:
    early, late = tmp_path / "early", tmp_path / "late"
    _pair(early, 24)
    _pair(late, 24)
    model = _model(tmp_path)

    result = _run(tmp_path, steps="24", roots=[early, late], judges=f"only|{model}")
    assert result.returncode == 0, result.stderr


def test_every_submission_pins_single_token_and_thinking_off(tmp_path: Path) -> None:
    # judge_sweep_cell.sh defaults JUDGE_PROMPT_STYLE to "full". An omission here scores the
    # full schema and files it under a run root named for the single-token protocol.
    early = tmp_path / "early"
    for step in (0, 12):
        _pair(early, step)
    model = _model(tmp_path)

    result = _run(tmp_path, steps="0 12", roots=[early], judges=f"only|{model}")
    assert result.returncode == 0, result.stderr

    submissions = _submissions(result)
    assert len(submissions) == 2
    for line in submissions:
        assert "JUDGE_PROMPT_STYLE=single_token" in line
        assert "THINKING_MODE=off" in line


def test_one_independent_chain_per_judge(tmp_path: Path) -> None:
    # Judges must not be chained to each other: one dead judge would strand the rest.
    early = tmp_path / "early"
    for step in (0, 12):
        _pair(early, step)
    first, second = _model(tmp_path, "first"), _model(tmp_path, "second")

    result = _run(tmp_path, steps="0 12", roots=[early],
                  judges=f"cell-a|{first};cell-b|{second}")
    assert result.returncode == 0, result.stderr

    submissions = _submissions(result)
    assert len(submissions) == 4  # 2 judges x 2 steps
    # The first step of each chain has no dependency; the second depends on the first.
    heads = [line for line in submissions if "--dependency" not in line]
    assert len(heads) == 2, "each judge chain must start unconditionally"


def test_steps_are_submitted_as_an_afterok_chain(tmp_path: Path) -> None:
    early = tmp_path / "early"
    for step in (0, 12, 24):
        _pair(early, step)
    model = _model(tmp_path)

    result = _run(tmp_path, steps="0 12 24", roots=[early], judges=f"only|{model}")
    submissions = _submissions(result)
    assert len(submissions) == 3
    assert "--dependency" not in submissions[0]
    assert all("--dependency=afterok:" in line for line in submissions[1:])


def test_a_missing_local_judge_checkpoint_is_fatal_before_any_submission(tmp_path: Path) -> None:
    early = tmp_path / "early"
    _pair(early, 0)

    result = _run(tmp_path, steps="0", roots=[early],
                  judges=f"only|{tmp_path / 'absent'}")
    assert result.returncode == 2
    assert "missing local judge model" in result.stderr
    assert not _submissions(result)


def test_an_already_claimed_run_root_is_refused(tmp_path: Path) -> None:
    early = tmp_path / "early"
    _pair(early, 0)
    model = _model(tmp_path)
    run_root = tmp_path / "run"

    first = _run(tmp_path, steps="0", roots=[early], judges=f"only|{model}",
                 eval_root=run_root)
    assert first.returncode == 0, first.stderr

    second = _run(tmp_path, steps="0", roots=[early], judges=f"only|{model}",
                  eval_root=run_root)
    assert second.returncode == 2
    assert "already claimed" in second.stderr


def test_stale_reward_output_is_refused(tmp_path: Path) -> None:
    early = tmp_path / "early"
    _pair(early, 0)
    model = _model(tmp_path)
    run_root = tmp_path / "run"
    stale = run_root / "raw" / f"{PREFIX}0" / "sweep" / "only" / "off" / "single_token" / "reward"
    stale.mkdir(parents=True)
    (stale / "reward-1.jsonl").write_text("{}\n")

    result = _run(tmp_path, steps="0", roots=[early], judges=f"only|{model}",
                  eval_root=run_root)
    assert result.returncode == 2
    assert "refusing stale output" in result.stderr
