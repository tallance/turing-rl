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
    # Interpolation, not equal literals. Two literals agreed at rest but diverged the moment a
    # caller overrode data.max_response_length: the batch allowance rose while vLLM stayed at
    # max_new_tokens 7680 (job 18583). Equality alone could not catch that, since it only ever
    # inspects the committed file, never the overridden run.
    assert rollout["response_length"] == "${data.max_response_length}"


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
    """The parent is an 8B generator config; every value here is wrong for a judge run.

    gpu_memory_utilization and max_num_seqs are MEASURED (capacity probe, job 15926, A100-40GB
    at max_model_len 22,016): 0.55 yields only 2.76x concurrency, 0.70 yields 10.98x. The
    inherited max_num_seqs of 64 would admit ~6x more sequences than physically fit.
    """
    rollout = _load()["actor_rollout_ref"]["rollout"]
    assert rollout["tensor_model_parallel_size"] == 1
    assert float(rollout["gpu_memory_utilization"]) == 0.70
    # Judge prompts (p95 ~10k tokens) far exceed the 4096 batched-token cap.
    assert rollout["enable_chunked_prefill"] is True
    assert rollout["max_num_seqs"] == 16
    # Bounds the chunked-prefill workspace; must NOT track max_model_len.
    assert rollout["max_num_batched_tokens"] == 4096


def test_budgets_match_the_measured_prompt_and_completion_distributions():
    """Both budgets come from measurement, not from round numbers.

    Prompt: TOKENIZED, not estimated. Running the Qwen3.5 tokenizer over all 4,738 rows of the
    iter1 train+val splits gives p50 6,816, p99 9,620, max 10,535, and 11,264 truncates exactly
    0 rows. The previous floor of 12,743 came from the char-based CHARS_PER_TOKEN_ESTIMATE used
    when building pairs, which overshoots the real count by ~21%. Conservative is correct for a
    build-time filter and wrong for sizing this budget, because prompt and response share
    max_model_len -- the surplus was silently taken out of generation.

    Completion: the old floor came from the frozen judge's p90 length (7,452 over 91,398 calls),
    a heuristic. Direct measurement of the 9B at step 0, thinking ON, over 200 held-out rows
    showed 7,680 leaves 12.5% of rollouts with an unclosed <think> -- scoring zero and pulling
    accuracy below chance before any training. 9,216 drops that to 5% and lifts coverage
    0.817 -> 0.945. 10,752 is marginally better still but OOMs in update_actor.

    Re-derive both before pointing this config at a new slice; do not port these numbers.
    """
    config = _load()
    data = config["data"]
    measured_max_prompt_tokens = 10535

    assert data["max_prompt_length"] > measured_max_prompt_tokens, "would truncate prompts"
    assert data["max_response_length"] == 9216
    assert (
        data["max_prompt_length"] + data["max_response_length"]
        < config["actor_rollout_ref"]["rollout"]["max_model_len"]
    ), "saturating the context window exactly is what OOMed at 10752"


def test_judge_runs_log_to_their_own_wandb_project():
    assert _load()["trainer"]["project_name"] == "grpo-judge"


def test_lora_is_merged_before_the_rollout_weight_sync():
    """The one proven Qwen3.5 recipe pins this; without it the rollouts may be the base model.

    veRL's default weight-sync path for a hybrid GDN model is not the merged-dense one. If it
    hands the rollout engine base weights, the run completes and logs plausible curves while
    every sampled verdict came from the untrained model.
    """
    assert _load()["actor_rollout_ref"]["model"]["lora"]["merge"] is True


def test_the_checkpoint_engine_bucket_matches_the_proven_recipe():
    rollout = _load()["actor_rollout_ref"]["rollout"]
    assert rollout["checkpoint_engine"]["update_weights_bucket_megabytes"] == 3072


def test_all_three_micro_batch_knobs_are_pinned_to_one():
    """The 8B generator base leaves ref/rollout log-prob at 8 for far shorter sequences."""
    c = _load()["actor_rollout_ref"]
    assert c["actor"]["ppo_micro_batch_size_per_gpu"] == 1
    assert c["ref"]["log_prob_micro_batch_size_per_gpu"] == 1
    assert c["rollout"]["log_prob_micro_batch_size_per_gpu"] == 1


def test_a_console_logger_exists_so_the_overfit_gate_has_something_to_read():
    """veRL writes no metrics file; with wandb alone judge_overfit_gate.py cannot run."""
    assert "console" in _load()["trainer"]["logger"]


def test_the_budget_records_that_sample_maxima_grow_with_sample_size():
    """The budget is measured, but a measured max is a SAMPLE max.

    Growing the sample from 416 to 1,121 contexts raised the observed max ~10%. The file must
    keep telling the next reader to re-check n_over_budget for any new slice rather than
    trusting this number to hold.
    """
    with open(CFG) as handle:
        text = handle.read()
    # Comment prose wraps across lines with "#" markers; compare on normalized words.
    flat = " ".join(text.lower().replace("#", " ").split())
    assert "n_over_budget" in text
    assert "sample maxima grow with sample size" in flat
