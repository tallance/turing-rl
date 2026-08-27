"""Executing guards on the judge CE training launcher.

scripts/slurm/sft_variant.sh is a job script: it needs SLURM_JOB_ID, so cluster_launch.sh
cannot run it directly. scripts/launch_judge_ce_train.sh is the launcher that submits it,
and therefore the only sanctioned path to a judge cross-entropy run. These execute the
launcher under DRY=1 and read the --export list it would hand to sbatch.

The three rejections below are the deviations that otherwise run to completion while
measuring something other than a judge: a generator MODEL, the generator's SFT corpus as
DATA, and a checkpoint dir carrying the generator's dataset name.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "launch_judge_ce_train.sh"
JOB_SCRIPT = ROOT / "scripts" / "slurm" / "sft_variant.sh"

SBATCH = f"{ROOT}/scripts/snapshot_sbatch.sh"

JUDGE_MODELS = ("qwen35-4b-judge", "qwen35-9b-judge")
GENERATOR_MODELS = ("qwen3-8b", "qwen35-9b")


def _data(tmp_path: Path) -> str:
    """A real file: the DATA guard stats the path, so a fake one would pass for the wrong
    reason in the accept-cases and for the right one only by accident."""
    path = tmp_path / "ce_train.jsonl"
    path.write_text('{"messages": []}\n')
    return str(path)


def _env(tmp_path: Path, **overrides: str) -> dict[str, str]:
    # Deliberately minimal rather than inheriting os.environ: a stray MODEL/DATA/OUT in the
    # caller's environment would satisfy exactly the guards under test.
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "DRY": "1",
        "TURING_RL_WORK_ROOT": str(ROOT),
        "TURING_RL_CODE_ROOT": str(ROOT),
        "MODEL": "qwen35-9b-judge",
        "DATA": _data(tmp_path),
        "OUT": str(tmp_path / "ckpt" / "judge_ce_9b"),
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


def _exports(planned: str) -> list[str]:
    return planned.split("--export=")[1].split(" ")[0].split(",")


# --------------------------------------------------------------------------------------
# The submitted line
# --------------------------------------------------------------------------------------


def test_a_valid_judge_invocation_plans_the_expected_sbatch_line(tmp_path: Path) -> None:
    """Spelled out in full: this one line is the entire contract with the gateway."""
    env = _env(tmp_path)

    result = _run(env)

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert planned == (
        f"[DRY] {SBATCH} --parsable "
        f"--export=ALL,MODEL=qwen35-9b-judge,VARIANT=bf16_fsdp,"
        f"DATA={env['DATA']},OUT={env['OUT']} "
        f"-- scripts/slurm/sft_variant.sh"
    )


def test_both_judge_aliases_reach_the_job(tmp_path: Path) -> None:
    for model in JUDGE_MODELS:
        result = _run(_env(tmp_path, MODEL=model))

        assert result.returncode == 0, f"{model}: {result.stderr}"
        (planned,) = _planned(result)
        assert f"MODEL={model}" in _exports(planned), model


def test_the_submitted_job_script_exists(tmp_path: Path) -> None:
    """The path is relative to the snapshot root, so a typo would only surface as an sbatch
    failure on the cluster."""
    result = _run(_env(tmp_path))

    (planned,) = _planned(result)
    assert " -- " in planned, "the sbatch script boundary is missing"
    submitted = planned.split(" -- ")[1]
    assert (ROOT / submitted).is_file(), submitted


def test_every_exported_name_is_one_the_job_script_reads(tmp_path: Path) -> None:
    """Forwarding a name sft_variant.sh does not consume is a silent no-op: the run happens,
    with the default the caller thought they had overridden."""
    source = JOB_SCRIPT.read_text()
    (planned,) = _planned(_run(_env(tmp_path, EPOCHS="40", MAX_TRAIN_EXAMPLES="16")))

    names = [field.split("=")[0] for field in _exports(planned) if field != "ALL"]
    assert names, "nothing was exported"
    for name in names:
        assert re.search(rf"\$\{{{name}[:}}]", source), name


def test_the_launcher_does_not_decide_packing(tmp_path: Path) -> None:
    """NOPACK is forced to 1 by the judge aliases inside sft_variant.sh. A value set here
    would be exported through --export=ALL and read before that case runs, and with a
    one-token CE target packing lets an example attend into its neighbour's answer letter."""
    body = "\n".join(
        line for line in LAUNCHER.read_text().splitlines() if not line.strip().startswith("#")
    )
    assert "NOPACK" not in body

    (planned,) = _planned(_run(_env(tmp_path)))
    assert "NOPACK" not in planned


# --------------------------------------------------------------------------------------
# MODEL
# --------------------------------------------------------------------------------------


def test_a_generator_alias_is_rejected(tmp_path: Path) -> None:
    """sft_variant.sh accepts these happily. Under a judge run root, with judge CE data,
    they produce a generator LoRA -- a run that completes and means nothing."""
    for model in GENERATOR_MODELS:
        result = _run(_env(tmp_path, MODEL=model))

        assert result.returncode == 2, model
        assert "judge alias" in result.stderr, model
        assert model in result.stderr, "the message must show the rejected value"
        assert not _planned(result), model


def test_an_empty_or_unset_model_is_rejected(tmp_path: Path) -> None:
    """`${MODEL:-qwen3-8b}` in sft_variant.sh treats empty as unset, so an unresolved
    `--env MODEL=$SOMETHING_UNSET` would default to a generator. It has to die here."""
    for env in (_env(tmp_path, MODEL=""), {k: v for k, v in _env(tmp_path).items() if k != "MODEL"}):
        result = _run(env)

        assert result.returncode == 2
        assert "MODEL is unset or empty" in result.stderr
        assert not _planned(result)


# --------------------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------------------


def test_a_missing_data_is_rejected(tmp_path: Path) -> None:
    """Unset, sft_variant.sh falls back to the generator's prism_full_s42 SFT corpus and
    trains a judge on it."""
    for env in (
        _env(tmp_path, DATA=""),
        {k: v for k, v in _env(tmp_path).items() if k != "DATA"},
    ):
        result = _run(env)

        assert result.returncode == 2
        assert "DATA is unset or empty" in result.stderr
        assert not _planned(result)


def test_a_data_path_that_does_not_exist_is_rejected(tmp_path: Path) -> None:
    """The job checks this too -- after eight GPUs have been allocated."""
    absent = str(tmp_path / "not_built_yet.jsonl")
    assert not os.path.exists(absent)

    result = _run(_env(tmp_path, DATA=absent))

    assert result.returncode == 2
    assert "DATA does not exist" in result.stderr
    assert absent in result.stderr
    assert not _planned(result)


# --------------------------------------------------------------------------------------
# OUT
# --------------------------------------------------------------------------------------


def test_a_missing_out_is_rejected(tmp_path: Path) -> None:
    """sft_variant.sh's per-VARIANT default is
    checkpoints/sft/judge_qwen35_9b_prism_full_s42_bf16_fsdp_nopack -- a judge checkpoint
    labelled with the generator's dataset."""
    for env in (_env(tmp_path, OUT=""), {k: v for k, v in _env(tmp_path).items() if k != "OUT"}):
        result = _run(env)

        assert result.returncode == 2
        assert "OUT is unset or empty" in result.stderr
        assert not _planned(result)


def test_out_reaches_the_job_verbatim(tmp_path: Path) -> None:
    out = str(tmp_path / "judge_ce_9b_ep3")
    result = _run(_env(tmp_path, OUT=out))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert f"OUT={out}" in _exports(planned)


# --------------------------------------------------------------------------------------
# Optional passthroughs
# --------------------------------------------------------------------------------------


def test_optional_passthroughs_are_absent_unless_set(tmp_path: Path) -> None:
    """Empty is unset: forwarding EPOCHS= would reach sft_variant.sh's integer guard as ''
    -- accepted there as "unset" today, but the launcher must not depend on that."""
    for env in (_env(tmp_path), _env(tmp_path, EPOCHS="", MAX_TRAIN_EXAMPLES="")):
        result = _run(env)

        assert result.returncode == 0, result.stderr
        (planned,) = _planned(result)
        assert "EPOCHS" not in planned
        assert "MAX_TRAIN_EXAMPLES" not in planned


def test_optional_passthroughs_are_forwarded_when_set(tmp_path: Path) -> None:
    """The overfit gate is EPOCHS=40 MAX_TRAIN_EXAMPLES=16; without these it is unreachable
    through the sanctioned path."""
    result = _run(_env(tmp_path, EPOCHS="40", MAX_TRAIN_EXAMPLES="16"))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    exports = _exports(planned)
    assert "EPOCHS=40" in exports
    assert "MAX_TRAIN_EXAMPLES=16" in exports


def test_only_one_of_the_two_passthroughs_may_be_set(tmp_path: Path) -> None:
    result = _run(_env(tmp_path, EPOCHS="40"))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "EPOCHS=40" in _exports(planned)
    assert "MAX_TRAIN_EXAMPLES" not in planned


# --------------------------------------------------------------------------------------
# VARIANT
# --------------------------------------------------------------------------------------


def test_the_variant_defaults_to_bf16_fsdp(tmp_path: Path) -> None:
    """qlora_r64 would pass --force_qlora against a judge config that sets use_qlora: false,
    and sft_variant.sh cross-checks the variant against nothing."""
    assert "use_qlora: false" in (
        ROOT / "training" / "sft" / "configs" / "qwen35_9b_judge_lora.yaml"
    ).read_text()

    (planned,) = _planned(_run(_env(tmp_path)))
    assert "VARIANT=bf16_fsdp" in _exports(planned)


def test_an_explicit_variant_is_forwarded(tmp_path: Path) -> None:
    result = _run(_env(tmp_path, VARIANT="bf16_fa2"))

    assert result.returncode == 0, result.stderr
    (planned,) = _planned(result)
    assert "VARIANT=bf16_fa2" in _exports(planned)


def test_an_unknown_variant_is_rejected(tmp_path: Path) -> None:
    """sft_variant.sh's own `bad VARIANT=` exit happens inside the allocated job."""
    result = _run(_env(tmp_path, VARIANT="bf16-fsdp"))

    assert result.returncode == 2
    assert "VARIANT must be" in result.stderr
    assert not _planned(result)


# --------------------------------------------------------------------------------------
# Gateway conventions
# --------------------------------------------------------------------------------------


def test_the_launcher_requires_the_cluster_launch_environment(tmp_path: Path) -> None:
    """TURING_RL_WORK_ROOT unset must fail loudly rather than cd'ing nowhere and submitting
    from whatever directory the caller happened to be in."""
    env = {k: v for k, v in _env(tmp_path).items() if k != "TURING_RL_WORK_ROOT"}

    result = _run(env)

    assert result.returncode != 0
    assert "cluster_launch.sh" in result.stderr
    assert not _planned(result)


def test_a_real_submission_needs_the_gateway_path(tmp_path: Path) -> None:
    """DRY=0 without TURING_RL_CODE_ROOT must not fall through to a bare `sbatch`."""
    env = {k: v for k, v in _env(tmp_path).items() if k != "TURING_RL_CODE_ROOT"}
    env["DRY"] = "0"

    result = _run(env)

    assert result.returncode == 2
    assert "cluster_launch.sh" in result.stderr
