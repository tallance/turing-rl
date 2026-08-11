# tests/test_rl_9b_launcher.py
import pathlib
import re
import subprocess

S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()
SINGLE_NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b_1node.sh").read_text()
RUN_2NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b.sh").read_text()
CFG_9B = pathlib.Path("training/grpo/configs/qwen3_9b_grpo_turing.yaml").read_text()


def mode_arm(name: str) -> str:
    """Text of one `case "$MODE"` arm in the trainer script.

    Assertions about a mode must be scoped to its own arm: now that more than one mode
    exists, a plain substring check over the whole file cannot tell "full5 caps the
    dataset" from "some other mode does".
    """
    start = S.index(f"  {name})")
    rest = S[start + len(name) + 4:]
    end = re.search(r"^(  \w+\)|esac)", rest, re.M)
    return rest[: end.start()] if end else rest


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
    # be baked into ITS arm. Scoped to the arm because frac10ep10 legitimately caps both.
    full5 = mode_arm("full5")
    assert "train_max_samples" not in full5
    assert "val_max_samples" not in full5


def test_frac10ep10_pins_the_subsets_and_the_per_epoch_cadence():
    # 384 = 6 x 64 exactly (9.2% of 4174). Chosen over the round 10% (417) because the train
    # loader drops the last partial batch, so 417 would rotate 33 different rows out per epoch;
    # 384 means every sample is seen exactly once per epoch and exactly 10 times over the run. save_freq=test_freq=6 keeps ckpts and val on the same epoch grid.
    # 352 = 50% of the 705-row val split, and at seed 42 is the same subset the half-data run
    # used, so val is comparable across the two runs.
    arm = mode_arm("frac10ep10")
    for k in (
        "data.train_max_samples=384",
        "data.val_max_samples=352",
        "trainer.total_epochs=10",
        "trainer.save_freq=6",
        "trainer.test_freq=6",
        "trainer.val_before_train=True",
        "trainer.max_actor_ckpt_to_keep=null",
    ):
        assert k in arm, f"frac10ep10 must pin {k}"
    # save_freq already equals steps_per_epoch, so the epoch-end hook would double-save.
    assert "export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0" in arm
    # Batch size comes from the 9B config, not this arm; setting it here would add a second
    # variable versus full5.
    assert "data.train_batch_size=" not in arm
    assert "overfit|full|epoch1|full5|frac10ep10" in RUN_2NODE


def test_frac10ep10_rejects_extra_overrides_that_collide_with_pinned_keys():
    # Hydra gives the LAST occurrence of a key priority and EXTRA_OVERRIDES is appended after
    # "${OVR[@]}", so an ambient value silently beats everything the arm pins -- the same
    # --export=ALL inheritance that produced 13634. The arm must refuse to launch instead.
    arm = mode_arm("frac10ep10")
    assert "ERROR: EXTRA_OVERRIDES sets" in arm
    assert "exit 5" in arm
    for protected in (
        "data.train_max_samples",
        "data.val_max_samples",
        "trainer.total_epochs",
        "trainer.save_freq",
        "trainer.test_freq",
        "trainer.val_before_train",
        "trainer.max_actor_ckpt_to_keep",
    ):
        assert protected in arm, f"guard must protect {protected}"


def test_frac10ep10_extra_overrides_guard_actually_fires():
    # Functional counterpart to the static test above: extract the real guard text and run it,
    # so a broken `case` pattern cannot pass review by merely containing the right key names.
    start = S.index("    for _protected in")
    guard = S[start : S.index("done ;;", start) + len("done")]
    script = 'set -uo pipefail\nEXTRA_OVERRIDES="$1"\n' + guard
    for value, expected in (
        ("", 0),                                        # nothing set -> proceed
        ("trainer.total_epochs=3", 5),                  # collides with a pinned key
        ("+trainer.total_epochs=3", 5),                 # Hydra append form collides too
        ("data.val_max_samples=99", 5),
        ("trainer.max_actor_ckpt_to_keep=6", 5),        # the 13634 value
        ("actor_rollout_ref.actor.optim.lr=2e-5", 0),   # unpinned key: escape hatch survives
        ("data.train_batch_size=64", 0),                # the arm does not pin it -- must pass
        ("trainer.total_epochs_extra=3", 0),            # near-miss must not false-positive
    ):
        proc = subprocess.run(["bash", "-c", script, "_", value], capture_output=True, text=True)
        assert proc.returncode == expected, f"EXTRA_OVERRIDES={value!r}: {proc.stderr}"


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
    # Qwen3.5-9B at TP=1 loads ~17.7 GB, so 0.40 of a 40 GB A100 (16 GB) cannot hold the
    # weights at all, let alone a KV cache -- that was Arm-B job 11735. The old 0.40 default
    # was only survivable because TP defaulted to 4, which shards the weights. Now that the
    # trainer defaults to the TP=1 both completed 9B runs used, the memory fraction must
    # default to 0.55 alongside it; the two have to move together.
    assert "tensor_model_parallel_size=${RL_ROLLOUT_TP:-1}" in S
    assert "gpu_memory_utilization=${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in S
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


def test_9b_trainer_loads_the_9b_config_not_the_8b_one():
    # Job 15143 trained at lr 1e-5 / kl_loss_coef 1e-3 -- 8B-era values -- because the 9B
    # trainer hardcoded --config-name qwen3_8b_grpo_turing and the real 9B hyperparameters
    # only ever arrived through EXTRA_OVERRIDES at submit time. Launching without that string
    # is silent: 8B defaults are valid values, so nothing errors, the reward curve just stays
    # flat. The recipe must therefore come from a file the 9B run loads by default.
    assert 'GRPO_CONFIG_NAME=${GRPO_CONFIG_NAME:-qwen3_9b_grpo_turing}' in S
    assert '--config-name "$GRPO_CONFIG_NAME"' in S
    # The literal must be gone from the invocation (a comment may still mention the 8B file).
    assert "--config-name qwen3_8b_grpo_turing" not in S


def test_9b_config_pins_the_recipe_both_completed_9b_runs_used():
    # Verbatim from the command lines of 13634 (half) and 14217 (full5), which are identical
    # on all twelve settings. Resolved through veRL's own entry point, the new config
    # reproduces every one of them.
    for k in (
        "kl_loss_coef: 0.0001",     # 8B base says 0.001 -- 10x too much KL pull
        "lr: 0.0001",               # 8B base says 1.0e-5 -- 10x too small a step
        "temperature: 1.0",         # 8B base says 0.6
        "train_batch_size: 64",
        "ppo_mini_batch_size: 64",
        "top_p: 1.0",
        "top_k: -1",
    ):
        assert k in CFG_9B, f"9B config must pin {k}"
    # Validation sampling is narrower than training, and must not fall back to the 8B base's
    # greedy val_kwargs (temperature 0, do_sample False).
    for k in ("temperature: 0.7", "top_p: 0.8", "top_k: 20", "do_sample: true", "n: 1"):
        assert k in CFG_9B, f"9B config must pin val_kwargs {k}"
    # Inherit the shared base rather than fork it: qwen3_8b_grpo.yaml is the SSOT that
    # tests/test_grpo_config.py locks, and 8B runs still load it.
    assert "- qwen3_8b_grpo" in CFG_9B


def test_9b_rollout_defaults_match_what_both_9b_runs_actually_ran():
    # Both runs passed these as env vars every launch; the defaults were 4 and 0.40.
    assert "${RL_ROLLOUT_TP:-1}" in S
    assert "${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in S
