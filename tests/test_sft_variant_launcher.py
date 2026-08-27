"""Guards on scripts/slurm/sft_variant.sh, which now launches the judge CE run too.

The launcher cannot be executed here (it sources the cluster bootstrap and calls torchrun),
but its configuration resolution is a pure, side-effect-free block fenced by
`# >>> resolve-config` / `# <<< resolve-config`. These tests extract that block verbatim and
run it under bash, so the resolved DATA/OUT/PY/ARGS are the real ones the job would use.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from training.sft.lora_sft import MODEL_MAP


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "slurm" / "sft_variant.sh"

BEGIN = "# >>> resolve-config"
END = "# <<< resolve-config"

# The HF exports sit OUTSIDE the resolve-config fence (they are side effects, which that
# block excludes by construction), so they carry their own fence and their own harness.
HF_BEGIN = "# >>> hf-env"
HF_END = "# <<< hf-env"

REPO = "/repo"
GEN_ROOT = "/generated"
GENERATOR_DATA = f"{GEN_ROOT}/sft/prism_full_s42_sft_cot.jsonl"

QWEN35_PY = "/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python"
QWEN35_LAYER_CLS = "Qwen3_5DecoderLayer"

VARIANTS = ("qlora_r64", "bf16_fsdp", "bf16_fa2")
GENERATOR_MODELS = ("qwen3-8b", "qwen35-9b")
JUDGE_MODELS = ("qwen35-4b-judge", "qwen35-9b-judge")


def _text(path: Path) -> str:
    return path.read_text()


def _resolution_block() -> str:
    text = _text(SCRIPT)
    start = text.index(BEGIN)
    end = text.index(END)
    assert start < end
    block = text[start:end]
    # A block that no longer resolves anything would make every assertion below vacuous.
    for token in ("DATA=", "OUT=", "case \"$MODEL\" in", "ARGS=("):
        assert token in block, f"resolution block no longer contains {token!r}"
    return block


class ResolveError(RuntimeError):
    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        super().__init__(f"rc={result.returncode}\n{result.stdout}\n{result.stderr}")
        self.result = result


def _resolve(**env: str) -> dict[str, object]:
    """Run the launcher's real resolution block and report what it decided."""
    probe = "\n".join(
        [
            "",
            'printf "%s\\n" "DATA=$DATA" "OUT=$OUT" "PY=$PY" "STEM=$STEM" \\',
            '  "FSDP_LAYER_CLS=$FSDP_LAYER_CLS" "NOPACK=$NOPACK" "WANDB_NAME=$WANDB_NAME"',
            'printf "ARG=%s\\n" "${ARGS[@]}"',
            "",
        ]
    )
    script = "set -uo pipefail\n" + _resolution_block() + probe
    result = subprocess.run(
        ["bash", "-c", script],
        # A deliberately minimal env: inheriting the caller's would let a stray DATA/OUT/
        # MODEL leak in and mask a missing default.
        env={
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "REPO": REPO,
            "TURING_RL_GENERATED_DATA_ROOT": GEN_ROOT,
            **env,
        },
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ResolveError(result)

    resolved: dict[str, object] = {"ARGS": []}
    for line in result.stdout.splitlines():
        key, _, value = line.partition("=")
        if key == "ARG":
            resolved["ARGS"].append(value)  # type: ignore[union-attr]
        else:
            resolved[key] = value
    return resolved


def _arg_value(resolved: dict[str, object], flag: str) -> str:
    args: list[str] = resolved["ARGS"]  # type: ignore[assignment]
    return args[args.index(flag) + 1]


# --------------------------------------------------------------------------------------
# The regression guard that matters most: an existing generator run must not retarget.
# --------------------------------------------------------------------------------------


def test_generator_defaults_are_byte_identical_to_today():
    """DATA and OUT with none of the new variables set. Literal expected strings on
    purpose — deriving them from the script would make the test agree with any drift."""
    expected_out = {
        ("qwen3-8b", "qlora_r64"): "/repo/checkpoints/sft/qwen3_8b_prism_full_s42_qlora_r64",
        ("qwen3-8b", "bf16_fsdp"): "/repo/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp",
        ("qwen3-8b", "bf16_fa2"): "/repo/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fa2",
        ("qwen35-9b", "qlora_r64"): "/repo/checkpoints/sft/qwen35_9b_prism_full_s42_qlora_r64",
        ("qwen35-9b", "bf16_fsdp"): "/repo/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp",
        ("qwen35-9b", "bf16_fa2"): "/repo/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fa2",
    }
    for (model, variant), out in expected_out.items():
        resolved = _resolve(MODEL=model, VARIANT=variant)
        assert resolved["DATA"] == GENERATOR_DATA, (model, variant)
        assert resolved["OUT"] == out, (model, variant)
        assert resolved["NOPACK"] == "0", (model, variant)
        assert "--no_packing" not in resolved["ARGS"], (model, variant)
        assert _arg_value(resolved, "--data_path") == GENERATOR_DATA
        assert _arg_value(resolved, "--output_dir") == out


def test_the_unset_model_default_is_still_qwen3_8b():
    """MODEL is the one selector with a default; a caller who never set it must keep it."""
    resolved = _resolve(VARIANT="bf16_fsdp")
    assert resolved["STEM"] == "qwen3_8b"
    assert resolved["FSDP_LAYER_CLS"] == "Qwen3DecoderLayer"
    assert resolved["PY"] == "/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python"
    assert resolved["OUT"] == "/repo/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp"


def test_generator_nopack_and_run_tag_suffixes_still_apply_in_order():
    """The overridable OUT sits upstream of these two suffixes, so they must still land."""
    resolved = _resolve(MODEL="qwen35-9b", VARIANT="bf16_fsdp", NOPACK="1", RUN_TAG="ep3")
    assert resolved["OUT"] == "/repo/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_ep3"
    assert resolved["WANDB_NAME"] == "sft-bf16-fsdp-nopack-ep3"


def test_generator_wandb_names_are_unchanged():
    for variant, name in zip(VARIANTS, ("sft-qlora-r64", "sft-bf16-fsdp", "sft-bf16-fa2")):
        assert _resolve(MODEL="qwen3-8b", VARIANT=variant)["WANDB_NAME"] == name


# --------------------------------------------------------------------------------------
# Overrides
# --------------------------------------------------------------------------------------


def test_data_is_overridable():
    resolved = _resolve(MODEL="qwen35-9b-judge", VARIANT="bf16_fsdp", DATA="/judge/ce.jsonl")
    assert resolved["DATA"] == "/judge/ce.jsonl"
    assert _arg_value(resolved, "--data_path") == "/judge/ce.jsonl"


def test_out_is_overridable():
    resolved = _resolve(MODEL="qwen35-9b-judge", VARIANT="bf16_fsdp", OUT="/ckpt/judge_ce_9b")
    assert resolved["OUT"].startswith("/ckpt/judge_ce_9b")
    assert _arg_value(resolved, "--output_dir") == resolved["OUT"]
    assert "prism_full_s42" not in resolved["OUT"], (
        "an overridden OUT must not carry the generator dataset name"
    )


# --------------------------------------------------------------------------------------
# Judge aliases
# --------------------------------------------------------------------------------------


def test_judge_aliases_take_the_qwen35_env_and_decoder_class():
    """Both are Qwen3.5: the 4.57.6 transformers in turing-rl-train cannot load
    model_type=qwen3_5, and Qwen3DecoderLayer is not a class FSDP could wrap here."""
    reference = _resolve(MODEL="qwen35-9b", VARIANT="bf16_fsdp")
    assert reference["PY"] == QWEN35_PY
    assert reference["FSDP_LAYER_CLS"] == QWEN35_LAYER_CLS
    for model in JUDGE_MODELS:
        resolved = _resolve(MODEL=model, VARIANT="bf16_fsdp")
        assert resolved["PY"] == reference["PY"], model
        assert resolved["FSDP_LAYER_CLS"] == reference["FSDP_LAYER_CLS"], model


def test_judge_aliases_force_no_packing_even_when_the_caller_disables_it():
    """The CE target is a single A/B token at the end of the example. Under sdpa, trl's
    packing=True leaks attention across packed conversations (isolation is delegated to
    FlashAttention varlen, absent in sdpa), so an example could read its neighbour's
    answer letter. NOPACK must not be the caller's to forget or to switch off."""
    for model in JUDGE_MODELS:
        for env in ({}, {"NOPACK": "0"}, {"NOPACK": "1"}):
            resolved = _resolve(MODEL=model, VARIANT="bf16_fsdp", **env)
            assert resolved["NOPACK"] == "1", (model, env)
            assert "--no_packing" in resolved["ARGS"], (model, env)


def test_judge_no_packing_survives_an_explicit_out():
    """Forcing NOPACK inside the MODEL case must not be undone by the OUT override that
    runs after it."""
    resolved = _resolve(MODEL="qwen35-4b-judge", VARIANT="bf16_fsdp", OUT="/ckpt/j4b")
    assert "--no_packing" in resolved["ARGS"]


def test_judge_checkpoints_cannot_land_on_a_generator_path():
    """Every generator path the launcher can produce, against every judge path it can
    produce. A judge stem colliding with a generator stem would resume from — or clobber —
    a generator checkpoint under `--resume_from_checkpoint auto`."""
    generator_outs = {
        _resolve(MODEL=model, VARIANT=variant, **nopack)["OUT"]
        for model in GENERATOR_MODELS
        for variant in VARIANTS
        for nopack in ({}, {"NOPACK": "1"})
    }
    judge_outs = {
        _resolve(MODEL=model, VARIANT=variant)["OUT"]
        for model in JUDGE_MODELS
        for variant in VARIANTS
    }
    assert len(judge_outs) == len(JUDGE_MODELS) * len(VARIANTS)
    assert not (generator_outs & judge_outs)
    for out in judge_outs:
        assert os.path.basename(out).startswith("judge_"), out


def test_judge_stems_are_distinct_from_every_generator_stem():
    generator_stems = {_resolve(MODEL=m, VARIANT="bf16_fsdp")["STEM"] for m in GENERATOR_MODELS}
    judge_stems = {_resolve(MODEL=m, VARIANT="bf16_fsdp")["STEM"] for m in JUDGE_MODELS}
    assert len(judge_stems) == len(JUDGE_MODELS)
    assert not (generator_stems & judge_stems)


def test_judge_model_names_are_aliases_lora_sft_actually_accepts():
    """--model is passed straight through to lora_sft.py, whose argparse pins
    choices=MODEL_MAP. A stem-style or invented name dies after the 8 GPUs are allocated."""
    for model in JUDGE_MODELS + GENERATOR_MODELS:
        resolved = _resolve(MODEL=model, VARIANT="bf16_fsdp")
        passed = _arg_value(resolved, "--model")
        assert passed == model
        assert passed in MODEL_MAP, passed
        # lora_sft.py derives its yaml from the same string.
        config = model.replace("-", "_").replace(".", "_") + "_lora.yaml"
        assert (ROOT / "training" / "sft" / "configs" / config).exists(), config


def test_unknown_model_is_still_rejected():
    # MODEL="" is deliberately absent: `${MODEL:-qwen3-8b}` treats empty as unset, which is
    # the pre-existing convention here and not something this change touches.
    for model in ("qwen35-4b", "judge_qwen35_9b", "qwen35-9b-JUDGE"):
        try:
            _resolve(MODEL=model, VARIANT="bf16_fsdp")
        except ResolveError as exc:
            assert exc.result.returncode == 2, model
        else:
            raise AssertionError(f"MODEL={model!r} was accepted")


# --------------------------------------------------------------------------------------
# EPOCHS / MAX_TRAIN_EXAMPLES passthroughs
# --------------------------------------------------------------------------------------

# Spelled out rather than derived from the script: a derived expectation would agree with
# any drift, which is exactly what "byte-identical to today" has to rule out.
BASELINE_ARGS = {
    ("qwen3-8b", "bf16_fsdp"): [
        "--model", "qwen3-8b",
        "--data_path", GENERATOR_DATA,
        "--output_dir", "/repo/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp",
        "--max_seq_length", "8192",
        "--resume_from_checkpoint", "auto",
        "--report_to", "wandb",
        "--no_torch_compile",
        "--no_qlora",
        "--attn_implementation", "sdpa",
        "--fsdp", "full_shard auto_wrap",
        "--fsdp_transformer_layer_cls", "Qwen3DecoderLayer",
    ],
    ("qwen35-4b-judge", "bf16_fsdp"): [
        "--model", "qwen35-4b-judge",
        "--data_path", GENERATOR_DATA,
        "--output_dir", "/repo/checkpoints/sft/judge_qwen35_4b_prism_full_s42_bf16_fsdp_nopack",
        "--max_seq_length", "8192",
        "--resume_from_checkpoint", "auto",
        "--report_to", "wandb",
        "--no_torch_compile",
        "--no_qlora",
        "--attn_implementation", "sdpa",
        "--fsdp", "full_shard auto_wrap",
        "--fsdp_transformer_layer_cls", QWEN35_LAYER_CLS,
        "--no_packing",
    ],
}

GENERATOR_PAIR = ("qwen3-8b", "bf16_fsdp")
JUDGE_PAIR = ("qwen35-4b-judge", "bf16_fsdp")

# Nothing that shells out to a GPU node should accept these.
BAD_INT_VALUES = ("3O", "0", "-1", "1.5", "3 4", " 3", "+3", "1e3", "3;echo pwned", "abc")


def test_unset_overrides_leave_the_arg_list_byte_identical():
    """The whole point of the passthrough is that it is invisible until asked for."""
    for pair, expected in BASELINE_ARGS.items():
        model, variant = pair
        assert _resolve(MODEL=model, VARIANT=variant)["ARGS"] == expected, pair


def test_smoke_alone_still_emits_its_own_cap():
    """Splitting the SMOKE append in two must not change what SMOKE=1 produces."""
    resolved = _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1], SMOKE="1")
    assert resolved["ARGS"] == BASELINE_ARGS[GENERATOR_PAIR] + [
        "--exit_after_trainer_build", "--max_train_examples", "64",
    ]


def test_smoke_still_precedes_no_packing_on_a_judge_alias():
    """--no_packing was appended after the SMOKE flags; keep that order."""
    base = BASELINE_ARGS[JUDGE_PAIR]
    assert base[-1] == "--no_packing"
    resolved = _resolve(MODEL=JUDGE_PAIR[0], VARIANT=JUDGE_PAIR[1], SMOKE="1")
    assert resolved["ARGS"] == base[:-1] + [
        "--exit_after_trainer_build", "--max_train_examples", "64", "--no_packing",
    ]


def test_epochs_appends_exactly_num_epochs():
    resolved = _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1], EPOCHS="40")
    assert resolved["ARGS"] == BASELINE_ARGS[GENERATOR_PAIR] + ["--num_epochs", "40"]


def test_max_train_examples_appends_exactly():
    resolved = _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1],
                        MAX_TRAIN_EXAMPLES="16")
    assert resolved["ARGS"] == BASELINE_ARGS[GENERATOR_PAIR] + ["--max_train_examples", "16"]


def test_the_overfit_gate_invocation_resolves_end_to_end():
    """MODEL=qwen35-4b-judge EPOCHS=40 MAX_TRAIN_EXAMPLES=16: many epochs over a handful of
    examples, still with packing off so no example can read its neighbour's answer."""
    resolved = _resolve(MODEL=JUDGE_PAIR[0], VARIANT=JUDGE_PAIR[1], EPOCHS="40",
                        MAX_TRAIN_EXAMPLES="16")
    assert resolved["ARGS"] == BASELINE_ARGS[JUDGE_PAIR] + [
        "--num_epochs", "40", "--max_train_examples", "16",
    ]


def test_max_train_examples_wins_over_smoke_and_is_passed_once():
    """argparse would take the last of a duplicated flag, but a job whose log shows the cap
    twice is unreadable. The explicit value wins; SMOKE keeps its early exit."""
    resolved = _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1], SMOKE="1",
                        MAX_TRAIN_EXAMPLES="16")
    args: list[str] = resolved["ARGS"]  # type: ignore[assignment]
    assert args.count("--max_train_examples") == 1
    assert _arg_value(resolved, "--max_train_examples") == "16"
    assert "64" not in args
    assert "--exit_after_trainer_build" in args


def test_non_positive_integers_are_rejected_before_the_gpus_are_allocated():
    """EPOCHS=3O (letter O) must die at submit, not after a node allocation, and must never
    be silently dropped."""
    for name in ("EPOCHS", "MAX_TRAIN_EXAMPLES"):
        for value in BAD_INT_VALUES:
            try:
                _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1], **{name: value})
            except ResolveError as exc:
                assert exc.result.returncode == 2, (name, value, exc.result.stdout)
                assert name in exc.result.stdout, (name, value)
            else:
                raise AssertionError(f"{name}={value!r} was accepted")


def test_one_epoch_and_one_example_are_accepted():
    """The lower bound is 1, not 2 — an off-by-one in the guard would bite the smallest gate."""
    resolved = _resolve(MODEL=GENERATOR_PAIR[0], VARIANT=GENERATOR_PAIR[1], EPOCHS="1",
                        MAX_TRAIN_EXAMPLES="1")
    assert resolved["ARGS"] == BASELINE_ARGS[GENERATOR_PAIR] + [
        "--num_epochs", "1", "--max_train_examples", "1",
    ]


def test_the_new_flags_are_ones_lora_sft_actually_accepts():
    """Same failure mode as an invented --model: argparse rejects it after the allocation."""
    source = (ROOT / "training" / "sft" / "lora_sft.py").read_text()
    for flag in ("--num_epochs", "--max_train_examples"):
        assert f'"{flag}"' in source, flag


def test_the_new_env_vars_are_documented_in_the_header():
    header = _text(SCRIPT).split(BEGIN)[0]
    for name in ("EPOCHS", "MAX_TRAIN_EXAMPLES"):
        assert name in header, name


# --------------------------------------------------------------------------------------
# Shared invariants
# --------------------------------------------------------------------------------------


def test_max_seq_length_is_not_lowered_below_the_judge_prompt_budget():
    """lora_sft.py defaults to 5120 and TRL truncates from the RIGHT, where the single A/B
    target sits. Judge prompts run past 5k, so a lower value would silently delete the
    supervised target on the longest rows."""
    for model in JUDGE_MODELS + GENERATOR_MODELS:
        resolved = _resolve(MODEL=model, VARIANT="bf16_fsdp")
        assert int(_arg_value(resolved, "--max_seq_length")) >= 8192, model


def test_the_bf16_fsdp_variant_wraps_the_resolved_decoder_class():
    """The judge arms are only useful through a variant that consumes FSDP_LAYER_CLS."""
    resolved = _resolve(MODEL="qwen35-9b-judge", VARIANT="bf16_fsdp")
    assert _arg_value(resolved, "--fsdp") == "full_shard auto_wrap"
    assert _arg_value(resolved, "--fsdp_transformer_layer_cls") == QWEN35_LAYER_CLS


RESOLVED_NAMES = ("DATA", "OUT", "DEFAULT_OUT", "PY", "STEM", "FSDP_LAYER_CLS", "NOPACK",
                  "RUN_TAG", "WANDB_NAME", "ARGS", "MODEL", "VARIANT", "SMOKE", "EPOCHS",
                  "MAX_TRAIN_EXAMPLES")


def test_the_fence_is_well_formed_and_covers_the_whole_invocation():
    text = _text(SCRIPT)
    assert text.count(BEGIN) == 1 and text.count(END) == 1
    assert text.index("cluster_job_bootstrap.sh") < text.index(BEGIN)
    assert text.index(END) < text.index("torch.distributed.run")


def test_nothing_outside_the_fence_resolves_configuration():
    """These tests execute only the fenced block, so a later assignment to one of the
    resolved names would change the job while every assertion above kept passing."""
    text = _text(SCRIPT)
    inside = range(text.index(BEGIN), text.index(END))
    offset = 0
    stray = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        start, offset = offset, offset + len(line)
        stripped = line.strip().removeprefix("export ")
        if start in inside or stripped.startswith("#"):
            continue
        for name in RESOLVED_NAMES:
            if stripped.startswith(f"{name}=") or stripped.startswith(f"{name}+="):
                stray.append(f"sft_variant.sh:{number}: {line.strip()}")
    assert not stray, f"resolution outside the tested fence: {stray}"


# --------------------------------------------------------------------------------------
# HF hub environment
# --------------------------------------------------------------------------------------


def _hf_env_block() -> str:
    text = _text(SCRIPT)
    start, end = text.index(HF_BEGIN), text.index(HF_END)
    assert start < end
    block = text[start:end]
    # A block that stopped setting this would make every assertion below vacuous.
    assert "HF_HUB_OFFLINE=" in block, "hf-env block no longer sets HF_HUB_OFFLINE"
    return block


def _resolve_hf_env(**env: str) -> dict[str, str]:
    """Execute the launcher's real HF export block and report what the job would see.

    The probe reads the values back from a CHILD process, not from the block's own shell:
    training runs in `$PY -m torch.distributed.run`, so a bare assignment that lost its
    `export` would leave transformers online while the parent shell still looked correct.
    """
    probe = (
        "\nbash -c 'printf \"%s\\n\" \"HF_HOME=${HF_HOME:-<unset>}\" "
        '"HF_HUB_CACHE=${HF_HUB_CACHE:-<unset>}" '
        '"HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}"\'\n'
    )
    result = subprocess.run(
        ["bash", "-c", "set -uo pipefail\n" + _hf_env_block() + probe],
        # Minimal env for the same reason as _resolve: a stray HF_HUB_OFFLINE inherited from
        # the caller would make the default look right no matter what the script does.
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **env},
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ResolveError(result)
    return dict(line.partition("=")[::2] for line in result.stdout.splitlines())


def test_hf_hub_offline_defaults_on_and_reaches_the_training_process():
    """Job 19315 died one minute into an 8-GPU allocation because an ONLINE resolve of a
    fully-cached Qwen3.5 failed: the cache uses nonstandard shard names that the Hub's file
    list does not contain. Every model this launcher trains is pre-cached."""
    resolved = _resolve_hf_env()
    assert resolved["HF_HUB_OFFLINE"] == "1"
    # The cache location is what makes the offline default safe; it must stay put.
    assert resolved["HF_HOME"] == "/home/lancewicki/data/hf_cache"
    assert resolved["HF_HUB_CACHE"] == "/home/lancewicki/data/hf_cache"


def test_hf_hub_offline_is_overridable_for_a_genuine_first_download():
    for value in ("0", "1"):
        assert _resolve_hf_env(HF_HUB_OFFLINE=value)["HF_HUB_OFFLINE"] == value


def test_hf_hub_offline_is_set_only_inside_its_fence_and_before_the_training_call():
    """A later unconditional `export HF_HUB_OFFLINE=` would override both the default and
    the caller's value while the tests above kept passing."""
    text = _text(SCRIPT)
    assert text.count(HF_BEGIN) == 1 and text.count(HF_END) == 1
    inside = range(text.index(HF_BEGIN), text.index(HF_END))
    assert text.index(HF_END) < text.index("torch.distributed.run")
    offset = 0
    stray = []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        start, offset = offset, offset + len(line)
        stripped = line.strip().removeprefix("export ")
        if start in inside or stripped.startswith("#"):
            continue
        if stripped.startswith("HF_HUB_OFFLINE="):
            stray.append(f"sft_variant.sh:{number}: {line.strip()}")
    assert not stray, f"HF_HUB_OFFLINE set outside the tested fence: {stray}"


def test_the_offline_default_records_why_it_cannot_be_removed():
    """Without the nonstandard shard name written down, this reads like a stray line that a
    future reader deletes, and the next judge run dies in its allocation again."""
    assert "model.safetensors-00001-of-00002.safetensors" in _hf_env_block()
    # Split at HF_BEGIN, not at BEGIN: the hf-env fence itself precedes the resolve-config
    # one, so `split(BEGIN)[0]` would be satisfied by the export and never fail.
    header = _text(SCRIPT).split(HF_BEGIN)[0]
    assert "HF_HUB_OFFLINE" in header, "the override is undocumented in the header"


def test_launcher_clears_the_stale_v2_proxy_vars():
    assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY" in _text(SCRIPT)


def test_launcher_never_calls_sbatch_directly():
    for line in _text(SCRIPT).splitlines():
        if not line.strip().startswith("#"):
            assert not line.strip().startswith("sbatch ")


def test_generated_data_never_resolves_through_the_source_snapshot():
    """Inside a job $REPO/data is the immutable source snapshot, where a generated jsonl
    does not exist. Generated datasets come from TURING_RL_GENERATED_DATA_ROOT."""
    for number, line in enumerate(_text(SCRIPT).splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        assert "$REPO/data/" not in line, f"sft_variant.sh:{number}"
    assert "TURING_RL_DATA_ROOT" not in _text(SCRIPT).replace(
        "TURING_RL_GENERATED_DATA_ROOT", ""
    )
