"""Regression guard for the judge GRPO config.

Locks the values that make this a *judge* run rather than a generator run: the local
reward function, thinking-on, the long-prompt budget, and — most importantly — a
target_modules list that never touches the Gated-DeltaNet backbone.
"""

import os

import yaml

CFG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "training", "grpo", "configs", "qwen35_judge_grpo.yaml",
)

# Gated-DeltaNet backbone projections. LoRA on any of these is destructive.
GDN_MODULES = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj")


def _load():
    with open(CFG) as handle:
        return yaml.safe_load(handle)


def test_uses_the_local_judge_reward():
    c = _load()
    assert c["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"
    assert c["custom_reward_function"]["name"] == "compute_score"


def test_thinking_is_enabled():
    assert _load()["data"]["apply_chat_template_kwargs"]["enable_thinking"] is True


def test_target_modules_never_touch_the_deltanet_backbone():
    modules = _load()["actor_rollout_ref"]["model"]["target_modules"]
    assert isinstance(modules, list), "must be an explicit list, never the base's all-linear"
    assert set(modules) == {
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    }
    for banned in GDN_MODULES:
        assert banned not in modules


def test_vision_and_mtp_are_excluded():
    assert _load()["actor_rollout_ref"]["model"]["exclude_modules"] == ".*(visual|mtp).*"


def test_prompt_and_response_budgets_fit_the_context_window():
    c = _load()
    data = c["data"]
    rollout = c["actor_rollout_ref"]["rollout"]
    assert data["max_prompt_length"] + data["max_response_length"] <= rollout["max_model_len"]
    assert data["max_response_length"] == rollout["response_length"]


def test_optimiser_follows_the_9b_recipe():
    c = _load()
    actor = c["actor_rollout_ref"]["actor"]
    assert float(actor["optim"]["lr"]) == 1e-4
    assert float(actor["kl_loss_coef"]) == 1e-4
    assert actor["use_kl_loss"] is True
    assert float(c["actor_rollout_ref"]["rollout"]["temperature"]) == 1.0


def test_validation_sampling_is_narrower_than_training():
    val = _load()["actor_rollout_ref"]["rollout"]["val_kwargs"]
    assert float(val["temperature"]) == 0.7
    assert float(val["top_p"]) == 0.8
    assert val["top_k"] == 20
    assert val["n"] == 1


def test_fsdp2_is_selected_for_actor_and_ref():
    c = _load()
    assert c["actor_rollout_ref"]["actor"]["strategy"] == "fsdp2"
    assert c["actor_rollout_ref"]["ref"]["strategy"] == "fsdp2"


def test_reward_is_declared_the_way_verl_09_actually_reads_it():
    """veRL 0.9's V1 controller ignores the legacy top-level block.

    A config carrying only `custom_reward_function` runs V1, never calls the reward, and
    scores every rollout 0 without erroring. The working 9B launcher disables V1 and uses
    the nested `reward.custom_reward_function` block; both must be present.
    """
    c = _load()
    assert c["trainer"]["use_v1"] is False
    assert c["reward"]["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"
    assert c["reward"]["custom_reward_function"]["name"] == "compute_score"
    assert c["custom_reward_function"]["path"] == "training/grpo/judge_reward.py"


def test_rollout_overrides_the_8b_generator_hardware_profile():
    """The parent is an 8B generator config; these three are wrong for a judge run."""
    rollout = _load()["actor_rollout_ref"]["rollout"]
    assert rollout["tensor_model_parallel_size"] == 1
    assert float(rollout["gpu_memory_utilization"]) == 0.55
    # Judge prompts (~6k tokens) exceed the 4096 batched-token cap.
    assert rollout["enable_chunked_prefill"] is True


def test_judge_runs_log_to_their_own_wandb_project():
    assert _load()["trainer"]["project_name"] == "grpo-judge"
