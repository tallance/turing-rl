"""Executing guards on judge_train_gen.sh's REUSE_GENERATIONS path.

The script's three steps (slice, generate, build) are control flow, so static text cannot
tell whether a step actually runs. These run the real script with $PY pointed at a stub that
records its own argv, and read the recording.

Two environment facts the harness works around:
  * the script sources cluster_job_bootstrap.sh, which needs a real SLURM_JOB_ID -- the stub
    root supplies a no-op instead;
  * SPLIT=val sets LIMIT=0, and bash 3.2 (macOS) cannot expand the resulting empty
    LIMIT_ARG array under `set -u`. That is pre-existing and cluster-irrelevant (bash 5),
    so these tests use SPLIT=train, which is also the split reuse was added for.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "scripts" / "slurm" / "judge_train_gen.sh"
LAUNCH = ROOT / "scripts" / "launch_judge_pairs.sh"

STUB = """#!/bin/bash
# Records the argv of every $PY invocation, one line per call, then succeeds.
printf '%s\\n' "$*" >> "$PY_CALLS"
"""


def _harness(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    """A stub root, a stub python, and the minimum environment the script dereferences."""
    code = tmp_path / "code" / "scripts"
    code.mkdir(parents=True)
    (code / "cluster_job_bootstrap.sh").write_text("# no-op: the real one needs SLURM_JOB_ID\n")

    work = tmp_path / "work"
    work.mkdir()

    python = tmp_path / "stub_python"
    python.write_text(STUB)
    python.chmod(0o755)

    calls = tmp_path / "py_calls.log"
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TURING_RL_CODE_ROOT": str(tmp_path / "code"),
        "TURING_RL_WORK_ROOT": str(work),
        "TURING_RL_INPUT_DATA_ROOT": str(tmp_path / "input"),
        "TURING_RL_JOB_PYTHON": str(python),
        "PY_CALLS": str(calls),
        "SPLIT": "train",
    }
    return calls, env


def _run(env: dict[str, str], calls: Path) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    result = subprocess.run(
        ["bash", str(GEN)], env=env, text=True, capture_output=True, check=False
    )
    recorded = calls.read_text().splitlines() if calls.exists() else []
    return result, recorded


def _seed_generations(raw_dir: Path, split: str = "train") -> tuple[Path, Path]:
    """The two files the reuse path requires. Contents are never read -- $PY is a stub."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    pkl = raw_dir / f"{split}_generations.pkl"
    sliced = raw_dir / f"{split}_source_slice.parquet"
    pkl.write_bytes(b"")
    sliced.write_bytes(b"")
    return pkl, sliced


def _step(recorded: list[str], needle: str) -> list[str]:
    return [line for line in recorded if needle in line]


# --- default mode: unchanged -------------------------------------------------------------


def test_default_still_slices_and_generates(tmp_path: Path) -> None:
    """REUSE_GENERATIONS unset must leave the three-step path exactly as it was: every pair
    set built before this knob existed has to be reproducible by setting nothing."""
    calls, env = _harness(tmp_path)
    env["OUT_DIR"] = str(tmp_path / "pairs")

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    assert len(_step(recorded, "slice_judge_source.py")) == 1
    assert len(_step(recorded, "eval.generate_trained")) == 1
    assert len(_step(recorded, "build_judge_train_pairs.py")) == 1
    # Order matters: the builder's `assert not missing` reads the pickle generation writes.
    assert recorded.index(_step(recorded, "slice_judge_source.py")[0]) < recorded.index(
        _step(recorded, "eval.generate_trained")[0]
    )


def test_default_keeps_the_raw_dir_under_out_dir(tmp_path: Path) -> None:
    """The generate path's pickle location is unchanged by the reuse plumbing."""
    calls, env = _harness(tmp_path)
    out_dir = tmp_path / "pairs"
    env["OUT_DIR"] = str(out_dir)

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    (generate,) = _step(recorded, "eval.generate_trained")
    assert f"--output {out_dir}/raw/train_generations.pkl" in generate
    assert (out_dir / "raw").is_dir(), "generate mode must still create the raw dir"


def test_reuse_raw_dir_is_ignored_when_reuse_is_off(tmp_path: Path) -> None:
    """A stale REUSE_RAW_DIR riding in on --export=ALL must not silently redirect a
    generating run's output pickle away from its own OUT_DIR."""
    calls, env = _harness(tmp_path)
    out_dir = tmp_path / "pairs"
    env["OUT_DIR"] = str(out_dir)
    env["REUSE_RAW_DIR"] = str(tmp_path / "elsewhere")

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    (generate,) = _step(recorded, "eval.generate_trained")
    assert f"--output {out_dir}/raw/train_generations.pkl" in generate
    assert "elsewhere" not in generate


# --- reuse mode: skips generation --------------------------------------------------------


def test_reuse_skips_the_slice_and_the_generation(tmp_path: Path) -> None:
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    _seed_generations(raw)
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    assert _step(recorded, "slice_judge_source.py") == []
    assert _step(recorded, "eval.generate_trained") == []
    assert len(_step(recorded, "build_judge_train_pairs.py")) == 1


def test_reuse_reads_the_named_raw_dir_not_a_prefix_of_out_dir(tmp_path: Path) -> None:
    """The whole point of the knob: the style-nested OUT_DIR
    (.../iter1/single_token) and the full-schema raw dir (.../iter1/raw) are different
    directories, and neither is derived from the other."""
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    pkl, sliced = _seed_generations(raw)
    out_dir = tmp_path / "iter1" / "single_token"
    env["OUT_DIR"] = str(out_dir)
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    (build,) = _step(recorded, "build_judge_train_pairs.py")
    assert f"--inference_pkl {pkl}" in build
    assert f"--source_parquet {sliced}" in build
    assert f"--out {out_dir}/train.parquet" in build
    assert not (out_dir / "raw").exists(), "reuse must not create a raw dir it never reads"


def test_reuse_logs_both_directories(tmp_path: Path) -> None:
    """The job's own stdout is the only record that a given run reused rather than sampled,
    and of which pickle it reused."""
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    pkl, _ = _seed_generations(raw)
    out_dir = tmp_path / "iter1" / "single_token"
    env["OUT_DIR"] = str(out_dir)
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, _ = _run(env, calls)

    assert result.returncode == 0, result.stderr
    assert "REUSE existing" in result.stdout
    assert str(pkl) in result.stdout
    assert str(out_dir) in result.stdout


def test_generate_mode_says_so_in_the_log(tmp_path: Path) -> None:
    calls, env = _harness(tmp_path)
    env["OUT_DIR"] = str(tmp_path / "pairs")

    result, _ = _run(env, calls)

    assert result.returncode == 0, result.stderr
    assert "GENERATE fresh" in result.stdout
    assert "REUSE existing" not in result.stdout


# --- reuse mode: fails loudly ------------------------------------------------------------
#
# These assert a specific message AND that generation was not attempted. A guard test that
# only checked a non-zero exit would pass on any unrelated failure -- an unreadable path, a
# typo'd variable -- and so would keep passing with the guard deleted.


def test_reuse_fails_when_the_pickle_is_missing(tmp_path: Path) -> None:
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    _seed_generations(raw)
    pkl = raw / "train_generations.pkl"
    pkl.unlink()
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, recorded = _run(env, calls)

    assert result.returncode != 0
    assert "no generations pickle" in result.stderr
    assert str(pkl) in result.stderr, "the message must name the path it looked for"
    assert recorded == [], "a missing pickle must not fall back to sampling new turns"


def test_reuse_fails_when_the_sliced_source_is_missing(tmp_path: Path) -> None:
    """The builder needs both. Only checking the pickle would let it die later, inside
    argparse or pandas, with a message that does not mention reuse at all."""
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    _seed_generations(raw)
    sliced = raw / "train_source_slice.parquet"
    sliced.unlink()
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, recorded = _run(env, calls)

    assert result.returncode != 0
    assert "no sliced source" in result.stderr
    assert str(sliced) in result.stderr
    assert recorded == []


def test_reuse_reports_both_missing_inputs_at_once(tmp_path: Path) -> None:
    """One resubmission per missing file, on a queue, is the failure mode worth avoiding."""
    calls, env = _harness(tmp_path)
    raw = tmp_path / "iter1" / "raw"
    raw.mkdir(parents=True)
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")
    env["REUSE_GENERATIONS"] = "1"
    env["REUSE_RAW_DIR"] = str(raw)

    result, recorded = _run(env, calls)

    assert result.returncode != 0
    assert "no generations pickle" in result.stderr
    assert "no sliced source" in result.stderr
    assert recorded == []


def test_reuse_without_a_raw_dir_is_rejected(tmp_path: Path) -> None:
    """Defaulting REUSE_RAW_DIR to $OUT_DIR/raw would make a single_token reuse look for the
    pickle under .../single_token/raw, which never holds one."""
    calls, env = _harness(tmp_path)
    out_dir = tmp_path / "iter1" / "single_token"
    env["OUT_DIR"] = str(out_dir)
    env["REUSE_GENERATIONS"] = "1"

    result, recorded = _run(env, calls)

    assert result.returncode != 0
    assert "requires REUSE_RAW_DIR" in result.stderr
    assert str(out_dir) in result.stderr, "the message must show the OUT_DIR it refused to guess from"
    assert recorded == []


def test_a_non_boolean_reuse_flag_is_rejected(tmp_path: Path) -> None:
    """`--env REUSE_GENERATIONS=true` would otherwise compare unequal to 1 and quietly
    generate -- the exact outcome the caller was trying to prevent."""
    calls, env = _harness(tmp_path)
    env["OUT_DIR"] = str(tmp_path / "pairs")
    env["REUSE_GENERATIONS"] = "true"

    result, recorded = _run(env, calls)

    assert result.returncode != 0
    assert "REUSE_GENERATIONS must be 0 or 1" in result.stderr
    assert recorded == []


def test_an_empty_reuse_flag_falls_back_to_generating(tmp_path: Path) -> None:
    """`--env REUSE_GENERATIONS=` must land on the historical behaviour, not on ''."""
    calls, env = _harness(tmp_path)
    env["OUT_DIR"] = str(tmp_path / "pairs")
    env["REUSE_GENERATIONS"] = ""

    result, recorded = _run(env, calls)

    assert result.returncode == 0, result.stderr
    assert len(_step(recorded, "eval.generate_trained")) == 1


# --- the knobs have to survive the trip to the job ---------------------------------------


def test_the_launcher_export_list_still_carries_the_ambient_environment() -> None:
    """Neither knob is named in launch_judge_pairs.sh's --export list. They reach the job
    through its leading ALL, which is what carries cluster_launch.py's `env NAME=VALUE ...`
    into sbatch. Narrowing that to NONE would leave both knobs inert: the job would sample
    fresh turns on a GPU node while the caller believed it was reusing -- and inert knobs
    fail silently, which is why this needs a guard (cf. the RL_CKPT_DIR case in
    tests/test_judge_launchers.py)."""
    code = [l for l in LAUNCH.read_text().splitlines() if not l.strip().startswith("#")]
    (exports,) = [l for l in code if l.strip().startswith("EXPORTS=")]
    assert exports.strip().startswith('EXPORTS="ALL,')


# --- the builder is shared by both modes -------------------------------------------------


def test_both_modes_reach_the_builder_with_the_same_prompt_style(tmp_path: Path) -> None:
    """--prompt-style is the only way to build single-token pairs. Duplicating the builder
    call into each branch is the mutation this catches: the reuse copy could drop it."""
    builds = {}
    for mode in ("generate", "reuse"):
        root = tmp_path / mode
        calls, env = _harness(root)
        env["PROMPT_STYLE"] = "single_token"
        if mode == "reuse":
            raw = root / "iter1" / "raw"
            _seed_generations(raw)
            env["OUT_DIR"] = str(root / "iter1" / "single_token")
            env["REUSE_GENERATIONS"] = "1"
            env["REUSE_RAW_DIR"] = str(raw)
        else:
            env["OUT_DIR"] = str(root / "iter1" / "single_token")

        result, recorded = _run(env, calls)

        assert result.returncode == 0, f"{mode}: {result.stderr}"
        (build,) = _step(recorded, "build_judge_train_pairs.py")
        assert "--prompt-style single_token" in build, mode
        builds[mode] = build

    # Same flags in both modes, differing only in the paths reuse redirects.
    def _flags(line: str) -> list[str]:
        return [token for token in line.split() if token.startswith("--")]

    assert _flags(builds["generate"]) == _flags(builds["reuse"])
