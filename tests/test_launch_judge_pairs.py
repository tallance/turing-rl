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
    env = _env(tmp_path, SPLITS="train", PROMPT_STYLE="single_token")
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "PROMPT_STYLE=single_token" in planned


def test_every_split_carries_the_style(tmp_path: Path) -> None:
    """Both splits feed one training run; a style set on only one of them would pair a
    single-token train set against a full-schema val set."""
    env = _env(tmp_path, SPLITS="train val", PROMPT_STYLE="single_token")
    env["OUT_DIR"] = str(tmp_path / "iter1" / "single_token")

    result = _run(env)

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


def _out_dir(planned: str) -> str:
    field = [f for f in planned.split("--export=")[1].split(" ")[0].split(",") if f.startswith("OUT_DIR=")]
    (only,) = field
    return only[len("OUT_DIR="):]


def _default_env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    """_env pins OUT_DIR; the default-path tests need it unset."""
    env = _env(tmp_path, **overrides)
    del env["OUT_DIR"]
    return env


def test_default_out_dir_for_the_full_style_is_unchanged(tmp_path: Path) -> None:
    """iter1 with no style segment. Every pair set built before the style existed lives
    here, and a new segment would orphan all of them."""
    env = _default_env(tmp_path, SPLITS="train")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert _out_dir(planned) == f"{env['TURING_RL_GENERATED_DATA_ROOT']}/prism/judge/iter1"


def test_default_out_dir_nests_the_single_token_style(tmp_path: Path) -> None:
    """iter1 identifies the data slice; the style is a rendering OF that slice, so it
    nests. Same fold convention as launch_judge_eval_matrix.sh's reward_dir."""
    env = _default_env(tmp_path, SPLITS="train", PROMPT_STYLE="single_token")

    result = _run(env)

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert _out_dir(planned) == f"{env['TURING_RL_GENERATED_DATA_ROOT']}/prism/judge/iter1/single_token"


# The guard reads the OUT_DIR string, and pytest's tmp_path embeds the test function name --
# which for these tests contains "single_token" and silently satisfies the very check under
# test. OUT_DIR is only echoed and exported in DRY mode, never stat'd, so these use a
# synthetic path that no test name can contaminate.
SYNTHETIC = "/synthetic/turing-rl/prism/judge"


def test_single_token_rejects_an_out_dir_that_does_not_name_the_style(tmp_path: Path) -> None:
    """Reusing the full-schema iter1 path out of habit overwrites the full-schema parquet
    AND its .meta.json -- destroying the only record of which style either file holds."""
    env = _env(tmp_path, SPLITS="train", PROMPT_STYLE="single_token")
    env["OUT_DIR"] = f"{SYNTHETIC}/iter1"

    result = _run(env)

    assert result.returncode != 0
    assert "must name the style" in result.stderr
    assert f"{SYNTHETIC}/iter1" in result.stderr, "the message must show the rejected path"
    assert not _planned(result)


def test_single_token_accepts_either_separator_in_the_out_dir(tmp_path: Path) -> None:
    """Being strict about which separator a human types buys nothing for a data directory."""
    for leaf in ("iter1-single-token", "iter1_single_token", "single_token/iter1"):
        env = _env(tmp_path, SPLITS="train", PROMPT_STYLE="single_token")
        env["OUT_DIR"] = f"{SYNTHETIC}/{leaf}"

        result = _run(env)

        assert result.returncode == 0, f"{leaf}: {result.stderr}"
        assert len(_planned(result)) == 1


def test_the_full_style_puts_no_constraint_on_out_dir(tmp_path: Path) -> None:
    """The guard is single_token-only: a full-style build into any path must still run,
    including one that happens to mention the other style."""
    for leaf in ("pairs", "iter1", "single-token-experiment"):
        env = _env(tmp_path, SPLITS="train")
        env["OUT_DIR"] = str(tmp_path / leaf)

        result = _run(env)

        assert result.returncode == 0, f"{leaf}: {result.stderr}"
        (planned,) = _planned(result)
        assert _out_dir(planned) == str(tmp_path / leaf)


def test_an_explicit_out_dir_does_not_require_the_generated_data_root(tmp_path: Path) -> None:
    """The default is resolved lazily. Making TURING_RL_GENERATED_DATA_ROOT unconditional
    would break every caller that passes --env OUT_DIR= without it."""
    env = _env(tmp_path, SPLITS="train")
    del env["TURING_RL_GENERATED_DATA_ROOT"]

    result = _run(env)

    assert result.returncode == 0, result.stderr
    assert len(_planned(result)) == 1


def test_an_empty_prompt_style_falls_back_to_full(tmp_path: Path) -> None:
    """`--env PROMPT_STYLE=` is a plausible slip. ${VAR:-default} substitutes on empty as
    well as unset, so it must land on full rather than reaching argparse as ''."""
    result = _run(_env(tmp_path, SPLITS="train", PROMPT_STYLE=""))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "PROMPT_STYLE=full" in planned
