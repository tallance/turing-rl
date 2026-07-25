# tests/test_rl_9b_launcher.py
import pathlib
S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()

def test_lora_target_is_attn_mlp_not_all_linear_excludes_visual_and_mtp():
    assert "all-linear" not in S                       # never LoRA the GDN backbone
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert m in S
    assert "visual" in S and "mtp" in S                # exclude vision tower AND MTP head
    assert "lora_rank=64" in S and "lora_alpha=32" in S

def test_mtp_disabled_via_override_config():
    assert "override_config.text_config.mtp_num_hidden_layers=0" in S

def test_merge_key_is_model_lora_merge():
    # correct veRL key is model.lora.merge (Hydra-appended), NOT rollout.lora.merge
    assert "actor_rollout_ref.model.lora.merge=True" in S
    assert "rollout.lora.merge" not in S

def test_offload_and_cache_clear():
    assert "param_offload=True" in S
    assert "optimizer_offload=True" in S
    assert "actor.fsdp_config.offload_policy=True" in S       # FSDP2-specific offload policy
    assert "ref.fsdp_config.offload_policy=True" in S
    assert "free_cache_engine=True" in S and "enforce_eager=True" in S

def test_checkpoint_engine_override_has_no_plus_prefix():
    # key already exists in current veRL -> `+` would error "already exists"
    assert "checkpoint_engine.update_weights_bucket_megabytes=3072" in S
    assert "+actor_rollout_ref.rollout.checkpoint_engine" not in S

def test_required_fsdp2_and_qwen35_overrides():
    for k in (
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=${RL_NGPUS:-8}",   # env-overridable NGPUS
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.max_model_len=13524",
        "actor_rollout_ref.rollout.calculate_log_probs=True",   # feeds the B0 logprob guard
        "checkpoint_engine.update_weights_bucket_megabytes=3072",
    ):
        assert k in S, f"missing required override: {k}"

def test_uses_merged_9b_and_no_cap():
    assert "merged_ep3" in S
    assert "TURING_JUDGE_SCORE_CLIP_MAX=7" in S or "cap" in S.lower()
