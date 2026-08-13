"""Static guards on the judge Slurm launchers."""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    """Three 8-GPU probes at once would take the entire 24-GPU QOS allowance."""
    text = _text(PROBE_LAUNCH)
    code = "\n".join(_code_lines(PROBE_LAUNCH))
    assert "afterany" in code, "chain the jobs so one node is held at a time"
    assert "afterok" not in code, "one judge failing to serve must not cancel the rest"
    assert "snapshot_sbatch.sh" in text
    assert "scripts/slurm/judge_format_probe.sh" in text


def test_probe_launcher_is_not_itself_a_job_script():
    text = _text(PROBE_LAUNCH)
    assert not [l for l in text.splitlines() if l.startswith("#SBATCH")]
