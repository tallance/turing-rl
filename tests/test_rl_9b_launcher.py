# tests/test_rl_9b_launcher.py
import pathlib
S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()
SINGLE_NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b_1node.sh").read_text()
RUN_2NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b.sh").read_text()


def test_judge_concurrency_is_pinned_not_inherited():
    # reward.py reads TURING_JUDGE_MAX_CONCURRENCY *before* PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY,
    # and sbatch --export=ALL carries the submitting shell's env in. Job 13634 inherited a stray
    # TURING_JUDGE_MAX_CONCURRENCY=8 that appears in no committed file and ran 41.5/44.1 h
    # judge-bound. The launcher must therefore SET the var unconditionally -- a `${VAR:-default}`
    # on TURING_JUDGE_MAX_CONCURRENCY itself would re-open exactly that hole.
    assert 'export TURING_JUDGE_MAX_CONCURRENCY="$JUDGE_CONC"' in RUN_2NODE
    assert 'export PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY="$JUDGE_CONC"' in RUN_2NODE
    assert "${TURING_JUDGE_MAX_CONCURRENCY:-" not in RUN_2NODE
    assert 'JUDGE_CONC="${JUDGE_CONC:-64}"' in RUN_2NODE
    # A long timeout is what makes high concurrency safe (reward.py falls back to 400 s).
    assert 'PERSONA_OPENAI_TIMEOUT_SECONDS="${PERSONA_OPENAI_TIMEOUT_SECONDS:-1800}"' in RUN_2NODE

def test_full5_mode_pins_the_whole_cadence_and_keeps_every_checkpoint():
    # The full-dataset 5-epoch run is ~5.7 days and 325 steps. Everything cadence-related is
    # pinned in the MODE arm rather than passed via EXTRA_OVERRIDES, so no run can silently
    # inherit a different value the way 13634 inherited its judge concurrency.
    assert "  full5)" in S
    for k in (
        "trainer.total_epochs=5",
        "trainer.save_freq=32",      # 10 ckpts at 32,64,...,320, ~12.3 h apart
        "trainer.test_freq=32",      # val on the SAME step grid, so every ckpt is scored
        "trainer.val_before_train=True",
    ):
        assert k in S, f"full5 must pin {k}"
    # veRL's default is null (keep all). 13634 ran with 6, which would drop the first four of
    # this run's ten checkpoints.
    assert "trainer.max_actor_ckpt_to_keep=null" in S
    assert "trainer.max_actor_ckpt_to_keep=6" not in S
    # The epoch-end hook fires at multiples of steps_per_epoch (65) and has no offset knob, so
    # leaving it on would add 5 near-duplicate saves 1-5 steps after the save_freq=32 ones.
    assert "export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0" in S
    # The driver validates MODE separately -- an unlisted mode is rejected before the trainer runs.
    assert "overfit|full|epoch1|full5" in RUN_2NODE


def test_full5_does_not_cap_the_dataset():
    # Job 13634 was the HALF run: data.train_max_samples=2048 / val_max_samples=352, passed via
    # EXTRA_OVERRIDES. full5 must run the full split (4174 train / 705 val), so neither cap may
    # be baked into the launcher.
    assert "train_max_samples" not in S
    assert "val_max_samples" not in S


def test_lora_target_is_attn_mlp_not_all_linear_excludes_visual_and_mtp():
    assert "all-linear" not in S                       # never LoRA the GDN backbone
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert m in S
    assert "visual" in S and "mtp" in S                # exclude vision tower AND MTP head
    assert '"actor_rollout_ref.model.exclude_modules=\'.*(visual|mtp).*\'"' in S
    assert "lora_rank=64" in S and "lora_alpha=32" in S

def test_mtp_disabled_via_override_config():
    assert "override_config.text_config.mtp_num_hidden_layers=0" in S

def test_merge_key_is_model_lora_merge():
    # The key exists at the pinned veRL SHA, so Hydra requires an ordinary override.
    assert "actor_rollout_ref.model.lora.merge=True" in S
    assert "+actor_rollout_ref.model.lora.merge=True" not in S
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
        "actor_rollout_ref.rollout.agent.num_workers=${RL_NGPUS:-8}",
        "reward.custom_reward_function.path=training/grpo/reward.py",
        "reward.custom_reward_function.name=compute_score",
        "trainer.use_v1=False",
    ):
        assert k in S, f"missing required override: {k}"


def test_verl_v1_reward_function_uses_new_nested_config_path():
    assert "reward.custom_reward_function.path=training/grpo/reward.py" in S
    assert "reward.custom_reward_function.name=compute_score" in S


def test_arm_b_uses_dataproto_controller_targeted_by_runtime_and_b0_hooks():
    assert "trainer.use_v1=False" in S


def test_rollout_batch_chunks_evenly_across_single_node_agent_workers():
    assert "actor_rollout_ref.rollout.agent.num_workers=${RL_NGPUS:-8}" in S

def test_uses_merged_9b_and_no_cap():
    assert "merged_ep3" in S
    assert "TURING_JUDGE_SCORE_CLIP_MAX=7" in S or "cap" in S.lower()
    assert "import transfer_queue" in S
    assert "TransferQueue==0.1.8" in S


def test_single_node_batch_is_divisible_by_seven_gpu_actor_dp():
    assert 'OVERFIT_TRAIN_BATCH="${OVERFIT_TRAIN_BATCH:-7}"' in SINGLE_NODE
    assert 'OVERFIT_PPO_MINI="${OVERFIT_PPO_MINI:-7}"' in SINGLE_NODE


def test_single_node_tp1_reserves_enough_memory_for_rollout_model_and_cache():
    assert "gpu_memory_utilization=${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}" in S
    assert 'RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"' in SINGLE_NODE


def test_single_node_judge_does_not_overlap_ray_trainer_gpu_ordinals():
    # Ray renumbers the seven trainer resources to physical ordinals 0-6. Keep
    # the judge on the remaining GPU instead of relying on an outer CVD offset.
    assert "CUDA_VISIBLE_DEVICES=7 \\" in SINGLE_NODE
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 RL_NGPUS=7" in SINGLE_NODE
    assert "judge=GPU7 trainer=GPU0-6" in SINGLE_NODE


def test_single_node_judge_cost_controls_keep_defaults_but_allow_debug_overrides():
    assert 'PERSONA_JUDGE_ENABLE_THINKING="${PERSONA_JUDGE_ENABLE_THINKING:-1}"' in SINGLE_NODE
    assert 'PERSONA_JUDGE_MAX_COMPLETION_TOKENS="${PERSONA_JUDGE_MAX_COMPLETION_TOKENS:-8192}"' in SINGLE_NODE


def test_b0_fixed_sequence_probe_disables_batch_reordering():
    assert 'if [ "${B0_ROLLOUT_SYNC:-0}" = "1" ]; then' in S
    assert "OVR+=( trainer.balance_batch=False )" in S
