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


def test_training_enables_judge_thinking():
    assert "PERSONA_JUDGE_ENABLE_THINKING=1" in _text(TRAIN)


def test_training_never_re_enables_the_v1_controller():
    """veRL 0.9 V1 ignores the reward config; the yaml pins use_v1=false.

    A launcher override flipping it back on would silently disable the judge reward and
    score every rollout 0 without erroring.
    """
    assert "trainer.use_v1=True" not in _text(TRAIN)
    assert "trainer.use_v1=true" not in _text(TRAIN)
