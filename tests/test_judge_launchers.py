"""Static guards on the judge Slurm launchers."""

import os
import re
import subprocess
import sys

ROOT =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(ROOT, "scripts", "slurm", "judge_format_probe.sh")
GEN = os.path.join(ROOT, "scripts", "slurm", "judge_train_gen.sh")
TRAIN = os.path.join(ROOT, "scripts", "slurm", "judge_grpo_train.sh")
ALL = (PROBE, GEN, TRAIN)


def _text(path: str) -> str:
    with open(path) as handle:
        return handle.read()


def test_every_launcher_clears_the_stale_v2_proxy_vars():
    for path in ALL:
        assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY" in _text(path)


def test_every_launcher_runs_from_the_snapshot_roots():
    for path in ALL:
        text = _text(path)
        assert "TURING_RL_WORK_ROOT" in text
        assert "cluster_job_bootstrap.sh" in text


def test_no_launcher_calls_sbatch_directly():
    for path in ALL:
        for line in _text(path).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert not stripped.startswith("sbatch "), f"{path}: direct sbatch is forbidden"


def test_generation_pins_the_documented_sampling():
    text = _text(GEN)
    assert "GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.7}" in text
    assert "GEN_TOP_P=${GEN_TOP_P:-0.8}" in text
    assert "GEN_TOP_K=${GEN_TOP_K:-20}" in text
    assert "GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}" in text


def test_generation_defaults_to_the_sft_ep3_checkpoint():
    assert "merged_ep3" in _text(GEN)


def test_generation_slices_the_source_before_generating():
    """Generating over the full split and discarding 90% in the builder wastes ~15k
    generations of a 12h single-GPU job whose pickle is only written at the end."""
    text = _text(GEN)
    assert "scripts/slice_judge_source.py" in text
    generate_at = text.index("eval.generate_trained")
    slice_at = text.index("scripts/slice_judge_source.py")
    assert slice_at < generate_at, "the slice must be written before generation runs"
    # Both the generator and the builder must read the SLICED file, not the full split.
    assert '--test_parquet "$SLICED_PARQUET"' in text
    assert '--source_parquet "$SLICED_PARQUET"' in text
    assert '--test_parquet "$SOURCE_PARQUET"' not in text


def test_generation_records_the_prompt_budget_measurement():
    """The .meta.json prompt-length fields are what data.max_prompt_length must be set from."""
    assert "--prompt_budget_tokens" in _text(GEN)


def _builder_invocation(text: str) -> str:
    """The `$PY ... build_judge_train_pairs.py` command, joined across its backslashes."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "build_judge_train_pairs.py" in line and not line.strip().startswith("#"):
            block = [line]
            while block[-1].rstrip().endswith("\\") and index + len(block) < len(lines):
                block.append(lines[index + len(block)])
            return "\n".join(block)
    raise AssertionError("judge_train_gen.sh no longer invokes build_judge_train_pairs.py")


def test_generation_threads_the_prompt_style_to_the_builder():
    """--prompt-style is the only way to build single-token pairs, and the builder is
    reachable on the cluster only through this job script."""
    text = _text(GEN)
    assert "PROMPT_STYLE=${PROMPT_STYLE:-full}" in text, "default must stay full"
    assert '--prompt-style "$PROMPT_STYLE"' in _builder_invocation(text)


def test_generation_spells_every_builder_flag_the_way_the_builder_declares_it():
    """This invocation mixes dashes (--prompt-style) and underscores (--slice_lo) because
    the builder's own parser does. Guessing the wrong separator is an unrecognized-argument
    exit AFTER the 12h generation step has already run."""
    flags = set(re.findall(r"--[A-Za-z0-9][A-Za-z0-9_-]*", _builder_invocation(_text(GEN))))
    # LIMIT_ARG is expanded, not spelled, at the call site.
    flags.add("--limit")
    help_text = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "build_judge_train_pairs.py"), "--help"],
        capture_output=True, text=True, check=True, cwd=ROOT,
    ).stdout
    unknown = sorted(f for f in flags if f not in help_text)
    assert not unknown, f"judge_train_gen.sh passes flags the builder does not declare: {unknown}"


def test_probe_uses_the_freeform_capable_script():
    text = _text(PROBE)
    assert "scripts/probe_judge_format.py" in text
    assert "freeform" in text


def test_training_names_the_judge_config_and_arm():
    text = _text(TRAIN)
    assert "qwen35_judge_grpo" in text
    assert "JUDGE_REWARD_ARM" in text
    assert "REWARD_METRIC" not in text, "judge training must not inherit the generator reward switch"


def test_training_uses_the_qwen35_verl09_environment():
    assert "turing-rl-rl-qwen35" in _text(TRAIN)


def test_training_exports_the_judge_thinking_env_var():
    """Only that the export is present. The var is inert on the training path -- what
    actually turns thinking on there is data.apply_chat_template_kwargs.enable_thinking,
    locked by tests/test_judge_grpo_config.py."""
    assert "PERSONA_JUDGE_ENABLE_THINKING=1" in _text(TRAIN)


def test_training_passes_the_repo_config_dir_to_hydra():
    """Without --config-dir, Hydra resolves --config-name against veRL's own packaged
    configs and the job dies instantly with "Cannot find primary config"."""
    text = _text(TRAIN)
    assert "--config-dir training/grpo/configs" in text
    assert 'hydra.run.dir="$TURING_RL_HYDRA_DIR"' in text
    assert "hydra.job.chdir=false" in text


def test_training_disables_mtp_layers_on_the_actor():
    """New key, so it needs Hydra's `+`; copied verbatim from the proven 9B recipe."""
    assert (
        "+actor_rollout_ref.model.override_config.text_config.mtp_num_hidden_layers=0"
        in _text(TRAIN)
    )


def test_training_never_re_enables_the_v1_controller():
    """veRL 0.9 V1 ignores the reward config; the yaml pins use_v1=false.

    A launcher override flipping it back on would silently disable the judge reward and
    score every rollout 0 without erroring.
    """
    assert "trainer.use_v1=True" not in _text(TRAIN)
    assert "trainer.use_v1=true" not in _text(TRAIN)


LAUNCH = os.path.join(ROOT, "scripts", "launch_judge_pairs.sh")


def test_pair_launcher_exists_and_submits_through_the_gateway():
    """cluster_launch.sh runs its argument on the login node, so a JOB script cannot be
    handed to it directly — it needs this launcher to sbatch judge_train_gen.sh."""
    text = _text(LAUNCH)
    assert "snapshot_sbatch.sh" in text
    assert "scripts/slurm/judge_train_gen.sh" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not stripped.startswith("sbatch "), "direct sbatch is forbidden"


def test_pair_launcher_is_not_itself_a_job_script():
    """It runs on the login node: no #SBATCH directives, no bootstrap needing SLURM_JOB_ID.

    Checks line starts specifically — the file's header comment mentions "#SBATCH" while
    explaining why this launcher exists, and that prose is not a directive.
    """
    text = _text(LAUNCH)
    directives = [l for l in text.splitlines() if l.startswith("#SBATCH")]
    assert not directives, f"launcher must not carry job directives: {directives}"
    sourced = [
        l for l in text.splitlines()
        if "cluster_job_bootstrap.sh" in l and not l.strip().startswith("#")
    ]
    assert not sourced, f"launcher must not source the job bootstrap: {sourced}"


def test_pair_launcher_rejects_an_unknown_split():
    assert "SPLITS entries must be train or val" in _text(LAUNCH)


def test_pair_launcher_clears_the_stale_v2_proxy_vars():
    assert "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY" in _text(LAUNCH)


PROBE_LAUNCH = os.path.join(ROOT, "scripts", "launch_judge_format_probe.sh")


def test_probe_job_wires_the_eval_csv_dump():
    """probe_judge_format.py --dump_csv is the ONLY producer of the CSV that
    scripts/analyze_judge_training.py consumes. The job script must pass it through, and must
    pin the dumped regime -- otherwise the dump silently takes whichever regime ran last
    (freeform by default), while the published comparison is forced-schema."""
    text = _text(PROBE)
    assert "--dump_csv" in text
    assert "--dump_regime" in text
    assert 'DUMP_REGIME=${DUMP_REGIME:-json_schema}' in text


def _code_lines(path: str) -> list[str]:
    """Non-comment lines only. Header prose routinely names the thing it warns against."""
    return [l for l in _text(path).splitlines() if not l.strip().startswith("#")]


def test_probe_launcher_serializes_the_judges():
    """Three 8-GPU servers at once would take the entire 24-GPU QOS allowance."""
    text = _text(PROBE_LAUNCH)
    code = "\n".join(_code_lines(PROBE_LAUNCH))
    assert "afterany" in code, "chain the judges so one node is held at a time"
    assert "afterok" not in code, "one judge failing to serve must not cancel the rest"
    assert "snapshot_sbatch.sh" in text
    assert "scripts/slurm/judge_format_probe.sh" in text


def test_probe_serves_the_judge_as_a_separate_job():
    """A job that sources cluster_job_bootstrap.sh cannot invoke another script that also
    sources it: both derive the work dir from job-$SLURM_JOB_ID and the second dies with
    "runtime work directory already exists" (jobs 15951-15953). The server needs its own job
    id, so the launcher submits it separately and the probe waits on the endpoint file."""
    launch = _text(PROBE_LAUNCH)
    probe = _text(PROBE)
    assert "scripts/slurm/judge_serve_9b_replicas.sh" in launch
    assert "--dependency=after:" in launch, "probe starts once the server has STARTED"
    assert "RL_JUDGE_JOB_ID" in launch and "RL_JUDGE_JOB_ID" in probe
    # The probe must NOT background the serve script itself.
    assert "judge_serve_9b_replicas.sh" not in probe


def test_probe_job_requests_no_gpu():
    """It is an HTTP client; the server holds the node."""
    assert not [l for l in _text(PROBE).splitlines() if l.startswith("#SBATCH") and "gres=gpu" in l]


def test_probe_tears_down_the_server():
    assert "scancel" in _text(PROBE)


def test_probe_launcher_is_not_itself_a_job_script():
    text = _text(PROBE_LAUNCH)
    assert not [l for l in text.splitlines() if l.startswith("#SBATCH")]


DATA_CONSUMERS = (PROBE, GEN, TRAIN, LAUNCH, PROBE_LAUNCH)


def test_generated_data_paths_never_resolve_through_the_source_snapshot():
    """Inside a job, $REPO/data symlinks to the IMMUTABLE SOURCE SNAPSHOT.

    The snapshot carries only committed python modules, so a generated parquet is invisible
    there — $REPO/data/... silently points at a file that does not exist. Generated datasets
    must be addressed via TURING_RL_GENERATED_DATA_ROOT (the state root). Caught when the
    Phase 0 probe resolved its pair set to .../work/launcher-*/data/prism/... and would have
    failed after holding an 8-GPU node.
    """
    for path in DATA_CONSUMERS:
        for number, line in enumerate(_text(path).splitlines(), 1):
            if line.strip().startswith("#"):
                continue
            assert "$REPO/data/" not in line, (
                f"{os.path.basename(path)}:{number} addresses generated data through the "
                f"source snapshot; use $TURING_RL_GENERATED_DATA_ROOT"
            )


def test_no_script_uses_the_deprecated_ambiguous_data_root():
    """CLAUDE.md retires TURING_RL_DATA_ROOT in favour of the input/generated split."""
    for path in DATA_CONSUMERS:
        assert "TURING_RL_DATA_ROOT" not in _text(path)


TRAIN_LAUNCH = os.path.join(ROOT, "scripts", "launch_judge_train.sh")


def test_train_launcher_submits_through_the_gateway():
    text = _text(TRAIN_LAUNCH)
    assert "snapshot_sbatch.sh" in text
    assert "scripts/slurm/judge_grpo_train.sh" in text
    assert not [l for l in text.splitlines() if l.startswith("#SBATCH")]


def test_train_launcher_validates_the_reward_arm():
    assert "JUDGE_REWARD_ARM must be directional or graded" in _text(TRAIN_LAUNCH)


def test_extra_overrides_never_enter_the_comma_delimited_export_list():
    """EXTRA_OVERRIDES carries spaces; Slurm's --export list is comma-delimited and does not
    survive them. It must ride the environment via the leading ALL instead."""
    code = "\n".join(_code_lines(TRAIN_LAUNCH))
    assert "export EXTRA_OVERRIDES=" in code
    assert "EXTRA_OVERRIDES=$EXTRA" not in code.replace("export EXTRA_OVERRIDES=\"$EXTRA\"", "")


JUDGE_LAUNCHERS = {
    os.path.join(ROOT, "scripts", "launch_judge_train.sh"): TRAIN,
    os.path.join(ROOT, "scripts", "launch_judge_pairs.sh"): GEN,
    os.path.join(ROOT, "scripts", "launch_judge_format_probe.sh"): PROBE,
}


def test_every_documented_env_knob_is_actually_read_somewhere():
    """A `--env NAME=` in a launcher's usage block is a promise to the caller.

    RL_CKPT_DIR broke that promise: launch_judge_train.sh documented it, jobs 16244/16245
    were submitted WITH it set, and judge_grpo_train.sh read only CKPT_DIR -- so the
    checkpoints silently went somewhere else. An inert knob fails quietly, which is why it
    needs a static guard rather than a runtime one.
    """
    documented = re.compile(r"--env ([A-Z][A-Z0-9_]*)=")
    inert = []
    for launcher, job in JUDGE_LAUNCHERS.items():
        consumers = _text(launcher) + _text(job)
        for name in documented.findall(_text(launcher)):
            # A knob is live if something dereferences it, not merely if the name appears
            # in the usage comment it was read out of.
            if f"${{{name}" not in consumers and f"${name}" not in consumers:
                inert.append(f"{os.path.basename(launcher)}: {name}")
    assert not inert, f"documented but never read: {inert}"


def test_training_reads_the_checkpoint_dir_under_the_generator_name():
    """rl_generator_train_9b.sh:46 established RL_CKPT_DIR as the external name; the judge
    job must not invent a second one."""
    assert "RL_CKPT_DIR" in _text(TRAIN)


def test_judge_training_disables_the_generator_presence_penalty():
    """A 37-field ordered JSON is maximally repetitive; a presence penalty fights it.

    Measured at step 1 vs the penalty-free probe: 2B 0.460->0.090, 4B 0.735->0.523,
    9B 0.855->0.594. The 2B landed on the GRPO bootstrapping edge and collapsed to 0.000.
    Overridable, so the old behaviour is still reachable for a comparison run.
    """
    assert "export PERSONA_VLLM_PRESENCE_PENALTY=${PERSONA_VLLM_PRESENCE_PENALTY:-0}" in _text(TRAIN)


def test_the_shared_runtime_patch_default_is_left_alone():
    """verl_runtime_patch.py's 0.5 is the GENERATOR recipe (REPO_TRAINING_PRESENCE_PENALTY).

    Neutralising the penalty there would silently change generator RL, which is a different
    experiment that wants it. The judge opt-out belongs in the judge job script only.
    """
    patch = os.path.join(ROOT, "training", "grpo", "verl_runtime_patch.py")
    assert '_float_env("PERSONA_VLLM_PRESENCE_PENALTY", 0.5)' in _text(patch)


def test_train_launcher_can_chain_an_arm_behind_another_run():
    """The 0/1 arm is chained behind the graded arm of the same size with afterok.

    afterany would launch a second 8-GPU run to rediscover the same OOM. An unset
    JUDGE_DEPENDENCY must expand to no argument at all, not an empty --dependency=.
    """
    code = "\n".join(_code_lines(TRAIN_LAUNCH))
    assert 'DEP="--dependency=${JUDGE_DEPENDENCY}"' in code
    assert 'DEP=""' in code
    assert "$DEP --export=" in code


def test_overfit_mode_keeps_the_subset_side_balanced_and_batch_sized():
    """8 pairs x 2 orders = 16 rows; the batch must fit and stay divisible by the agent-loop
    worker count (preflight 17/26)."""
    text = _text(TRAIN_LAUNCH)
    assert "scripts/build_judge_overfit.py" in text
    assert "data.train_batch_size=$rows" in text
    assert "rows=$((OVERFIT_PAIRS * 2))" in text


def test_valsmoke_mode_validates_on_a_real_heldout_slice():
    """The whole point of valsmoke: overfit mode redirects data.val_files at its own 16-row
    TRAINING subset, so its "validation" numbers are training numbers. A cheap thinking ON/OFF
    A/B needs a genuine held-out reading, so valsmoke must leave data.val_files alone and pass a
    real val slice through VAL_FILE instead.
    """
    # Code only: the block's comments legitimately discuss data.val_files to explain the contrast.
    code = "\n".join(_code_lines(TRAIN_LAUNCH))
    valsmoke = code.split('if [ "$MODE" = valsmoke ]; then', 1)[1]
    text = _text(TRAIN_LAUNCH)

    assert "data.val_files" not in valsmoke, "valsmoke must not redirect validation at train rows"
    assert 'VAL_FILE=$DATA_DIR/val_smoke${VALSMOKE_PAIRS}.parquet' in valsmoke
    assert '--src "$DATA_DIR/val.parquet"' in valsmoke
    # Same pair-wise slicer as overfit, so both A/B orders survive and human_is_b stays balanced.
    assert "scripts/build_judge_overfit.py" in valsmoke


def test_valsmoke_measures_the_base_model_before_any_gradient_step():
    text = _text(TRAIN_LAUNCH)
    valsmoke = text.split('if [ "$MODE" = valsmoke ]; then', 1)[1]

    assert "trainer.val_before_train=True" in valsmoke
    assert "trainer.test_freq=1" in valsmoke
    assert "trainer.total_epochs=1" in valsmoke
    assert "trainer.save_freq=-1" in valsmoke, "a smoke must not write checkpoints"


def test_unknown_mode_is_rejected_rather_than_treated_as_full():
    """A typo'd MODE previously fell through to `full`, starting a ~9h run instead of a check."""
    code = "\n".join(_code_lines(TRAIN_LAUNCH))
    assert "full|overfit|valsmoke)" in code


def test_tmpdir_is_shared_so_the_compiler_cache_stays_warm():
    """TMPDIR must stay shared: TorchInductor caches under tempfile.gettempdir().

    A per-job TMPDIR (6bb1b18) was meant to dodge an Errno 16 pymp teardown collision. It did not
    -- 18915 hit the same collision inside its own per-job dir -- and it relocated the compiler
    cache, making every run a cold compile: +0.80 GB on the step-1 memory baseline against ~0.05 GB
    of margin. 18858 OOMed on step 2 as a result.
    """
    text = _text(TRAIN)

    assert "TMPDIR=/home/lancewicki/tmp/build\n" in text, (
        "TMPDIR must stay shared so TorchInductor reuses the warm compiler cache"
    )
