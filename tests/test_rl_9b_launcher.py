# tests/test_rl_9b_launcher.py
import pathlib
import re
import subprocess

import yaml

S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()
SINGLE_NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b_1node.sh").read_text()
RUN_2NODE = pathlib.Path("scripts/slurm/rl_generator_run_9b.sh").read_text()
SERVE = pathlib.Path("scripts/slurm/judge_serve_9b_replicas.sh").read_text()
# The eval-side cell script is the origin of the Gemma serving constants; it is another
# agent's file and is read here only to prove the two copies agree.
SWEEP_CELL = pathlib.Path("scripts/slurm/judge_sweep_cell.sh").read_text()
CFG_9B = pathlib.Path("training/grpo/configs/qwen3_9b_grpo_turing.yaml").read_text()
# Guard the 9B recipe on the parsed tree, not on substrings of the text. Every value below
# appears as a literal in this file (the `defaults:` include supplies none of them), so a
# plain safe_load resolves them -- the same argument tests/test_grpo_config.py makes.
CFG_9B_TREE = yaml.safe_load(CFG_9B)


_ARM_PAT = r"[\w.|-]+\)"


def mode_arm(name: str) -> str:
    """Text of the `case "$MODE"` arm that handles MODE=<name>, in the trainer script.

    Assertions about a mode must be scoped to its own arm: now that more than one mode
    exists, a plain substring check over the whole file cannot tell "full5 caps the
    dataset" from "some other mode does".

    Arms may list several patterns (`frac10ep10|frac10ep20)`), so match the alternation
    rather than an exact `  name)` literal -- and terminate on the same alternation form,
    or an arm that shares a prefix would silently swallow the next one's body.
    """
    m = re.search(rf"^  (?:[\w.-]+\|)*{re.escape(name)}(?:\|[\w.-]+)*\)", S, re.M)
    assert m, f"no case arm handles MODE={name}"
    rest = S[m.end():]
    end = re.search(rf"^(  {_ARM_PAT}|esac)", rest, re.M)
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


def test_frac10_modes_pin_the_subsets_and_the_per_epoch_cadence():
    # 384 = 6 x 64 exactly (9.2% of 4174). Chosen over the round 10% (417) because the train
    # loader drops the last partial batch, so 417 would rotate 33 different rows out per epoch;
    # 384 means every sample is seen exactly once per epoch. Confirmed on the completed
    # 10-epoch run: 384 keys judged exactly 40x (10 epochs x 4 rollouts), no rotation.
    # save_freq=test_freq=6 keeps ckpts and val on the same epoch grid.
    # 352 = 50% of the 705-row val split, and at seed 42 is the same subset the half-data run
    # used, so val is comparable across runs.
    #
    # Both frac10 modes share one arm, so the slice cannot drift between them; only the epoch
    # count differs, and it is derived from the mode name rather than written twice.
    for mode in ("frac10ep10", "frac10ep20"):
        arm = mode_arm(mode)
        for k in (
            "data.train_max_samples=384",
            "data.val_max_samples=352",
            "trainer.total_epochs=$_EPOCHS",
            "trainer.save_freq=6",
            "trainer.test_freq=6",
            "trainer.val_before_train=True",
            "trainer.max_actor_ckpt_to_keep=null",
        ):
            assert k in arm, f"{mode} must pin {k}"
        # save_freq already equals steps_per_epoch, so the epoch-end hook would double-save.
        assert "export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0" in arm
        # Batch size comes from the 9B config, not this arm; setting it here would add a
        # second variable versus full5.
        assert "data.train_batch_size=" not in arm
        assert "_EPOCHS=${MODE#frac10ep}" in arm
    assert "overfit|full|epoch1|full5|frac10ep10|frac10ep20" in RUN_2NODE


def test_frac10_epoch_count_comes_from_the_mode_name():
    # The whole point of sharing one arm: `frac10ep20` must resolve to 20 epochs without a
    # second hard-coded literal that could drift from the mode it is named after.
    script = 'MODE="$1"\n_EPOCHS=${MODE#frac10ep}\necho "$_EPOCHS"\n'
    for mode, expected in (("frac10ep10", "10"), ("frac10ep20", "20")):
        proc = subprocess.run(["bash", "-c", script, "_", mode], capture_output=True, text=True)
        assert proc.stdout.strip() == expected, f"{mode} -> {proc.stdout!r}"


def test_frac10_rejects_extra_overrides_that_collide_with_pinned_keys():
    # Hydra gives the LAST occurrence of a key priority and EXTRA_OVERRIDES is appended after
    # "${OVR[@]}", so an ambient value silently beats everything the arm pins -- the same
    # --export=ALL inheritance that produced 13634. The arm must refuse to launch instead.
    arm = mode_arm("frac10ep20")
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


def test_frac10_extra_overrides_guard_actually_fires():
    # Functional counterpart to the static test above: extract the real guard text and run it,
    # so a broken `case` pattern cannot pass review by merely containing the right key names.
    # MODE is supplied because the guard now names the offending mode in its error message.
    start = S.index("    for _protected in")
    guard = S[start : S.index("done ;;", start) + len("done")]
    script = 'set -uo pipefail\nMODE=frac10ep20\nEXTRA_OVERRIDES="$1"\n' + guard
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
    #
    # Asserted at exact key paths on the parsed tree. The earlier version of this test matched
    # substrings, which cannot distinguish the two settings this file spells the same way:
    # `lr: 0.0001` appears twice (actor, then critic -- and GRPO never reads the critic), and
    # there are two `temperature:` values (1.0 for training rollouts, 0.7 for validation).
    # Corrupting the actor lr to the 8B value that flat-lined job 15143 left the substring
    # form fully green, so the guard did not in fact guard the thing it was written for.
    c = CFG_9B_TREE
    actor = c["actor_rollout_ref"]["actor"]
    rollout = c["actor_rollout_ref"]["rollout"]

    assert actor["optim"]["lr"] == 1e-4      # 8B base says 1.0e-5 -- 10x too small a step
    assert actor["kl_loss_coef"] == 1e-4     # 8B base says 0.001 -- 10x too much KL pull
    assert actor["use_kl_loss"] is True
    assert actor["ppo_mini_batch_size"] == 64
    assert c["data"]["train_batch_size"] == 64

    assert rollout["temperature"] == 1.0     # 8B base says 0.6
    assert rollout["top_p"] == 1.0
    assert rollout["top_k"] == -1

    # Validation sampling is narrower than training, and must not fall back to the 8B base's
    # greedy val_kwargs (temperature 0, do_sample False).
    val = rollout["val_kwargs"]
    assert val["temperature"] == 0.7
    assert val["top_p"] == 0.8
    assert val["top_k"] == 20
    assert val["do_sample"] is True
    assert val["n"] == 1

    # Inherit the shared base rather than fork it: qwen3_8b_grpo.yaml is the SSOT that
    # tests/test_grpo_config.py locks, and 8B runs still load it.
    assert "qwen3_8b_grpo" in c["defaults"]


def test_9b_rollout_defaults_match_what_both_9b_runs_actually_ran():
    # Both runs passed these as env vars every launch; the defaults were 4 and 0.40.
    assert "${RL_ROLLOUT_TP:-1}" in S
    assert "${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}" in S


GEMMA_12B_SNAPSHOT = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"


def judge_arm(name: str) -> str:
    """Text of the `case "$JUDGE"` arm that resolves JUDGE=<name>, in the 2-node driver."""
    m = re.search(rf"^  {re.escape(name)}\)\s", RUN_2NODE, re.M)
    assert m, f"no case arm resolves JUDGE={name}"
    rest = RUN_2NODE[m.end():]
    end = re.search(r"^(  [\w.|-]+\)|esac)", rest, re.M)
    return rest[: end.start()] if end else rest


def test_gemma_judge_resolves_the_shape_the_eval_registry_assigns():
    # gemma-4-12B is ~24GB in bf16, under the 30GB per-GPU budget in
    # configs/judge_sweep_cells.py:tp_for_size, so it gets TP=1 across 8 replicas rather
    # than spanning the node. Getting this backwards would cost ~8x judge throughput on a
    # judge-bound run without failing anything.
    assert "JUDGE=${JUDGE:?set JUDGE=9b|397b|gemma4-12b}" in RUN_2NODE
    assert 'case "$JUDGE" in 9b|397b|gemma4-12b) ;;' in RUN_2NODE
    arm = judge_arm("gemma4-12b")
    assert "JUDGE_MODEL=google/gemma-4-12B-it" in arm
    assert "TP=1" in arm and "DP=8" in arm
    assert "REASONING_PARSER=gemma4" in arm


def test_reasoning_parser_is_pinned_per_family_and_forwarded_to_the_judge_step():
    # The boundary detector is family-specific. A qwen3 parser on a gemma server does not
    # error -- it mis-splits thinking text out of .content, and the reward path then fails to
    # parse with nothing in the log naming the cause. So it must be resolved from JUDGE and
    # passed explicitly into the judge srun, never inherited from the submitting shell.
    assert "REASONING_PARSER=qwen3" in judge_arm("9b")
    assert "REASONING_PARSER=qwen3" in judge_arm("397b")
    assert "REASONING_PARSER=gemma4" in judge_arm("gemma4-12b")
    # forwarded into the srun that launches the server
    assert "REASONING_PARSER=$REASONING_PARSER" in RUN_2NODE
    # and never read back from the ambient environment in the driver
    assert "${REASONING_PARSER:-" not in RUN_2NODE
    # resolved value is echoed, so the log records which parser actually served
    assert "judge serving pinned:" in RUN_2NODE


def test_judge_case_arms_resolve_correctly_when_executed():
    # Functional counterpart: run the real `case` block rather than trusting substrings.
    start = RUN_2NODE.index('case "$JUDGE" in\n  9b)')
    block = RUN_2NODE[start : RUN_2NODE.index("esac", start) + len("esac")]
    script = 'JUDGE="$1"\n' + block + '\necho "$JUDGE_MODEL|$TP|$DP|$REASONING_PARSER"'
    for judge, expected in (
        ("9b", "Qwen/Qwen3.5-9B|1|8|qwen3"),
        ("397b", "Qwen/Qwen3.5-397B-A17B-GPTQ-Int4|8|1|qwen3"),
        ("gemma4-12b", "google/gemma-4-12B-it|1|8|gemma4"),
    ):
        proc = subprocess.run(["bash", "-c", script, "_", judge], capture_output=True, text=True)
        assert proc.stdout.strip() == expected, f"JUDGE={judge}: {proc.stdout!r} {proc.stderr}"


def test_serve_script_takes_the_gemma_branch_only_for_gemma():
    # Gemma needs the CUDA-13 nightly build; the qwen envs cannot serve it. Selecting the
    # wrong interpreter fails loudly, but the remaining three differences do not:
    # the snapshot pin, the multimodal slots, and the memory fraction.
    assert "turing-rl-gemma4-vllm-nightly/bin/vllm" in SERVE
    assert GEMMA_12B_SNAPSHOT in SERVE
    assert '--served-model-name "$MODEL"' in SERVE
    assert '--limit-mm-per-prompt' in SERVE
    assert "GEMMA_GPU_MEMORY_UTILIZATION:-0.90" in SERVE
    assert "FLASHINFER_WORKSPACE_BASE" in SERVE
    # the qwen path must survive untouched alongside it
    assert "-m vllm.entrypoints.openai.api_server" in SERVE
    assert "judge-vllm/bin/python" in SERVE
    assert "turing-rl-train/bin/python" in SERVE
    # and the health gate still matches on the advertised model id, which --served-model-name
    # keeps equal to $MODEL even though gemma is served from a snapshot path
    assert 'grep -qF "\\"$MODEL\\""' in SERVE


def test_serve_script_fails_fast_on_a_missing_gemma_runtime_or_snapshot():
    # Serving OFFLINE from a path means a missing/partial snapshot would otherwise surface as
    # an opaque vLLM stacktrace 20 minutes into warmup.
    assert "ERROR: missing Gemma vLLM" in SERVE
    assert "ERROR: incomplete Gemma snapshot" in SERVE


def test_gemma_snapshot_pin_matches_the_eval_cell_script():
    # Two scripts now serve gemma-4-12B: the eval sweep cell and this training judge. If they
    # drift to different revisions, the training run's rewards stop being comparable to the
    # eval numbers we validate them against, silently.
    assert GEMMA_12B_SNAPSHOT in SWEEP_CELL, "eval cell no longer pins this revision"
    assert "--reasoning-parser gemma4" in SWEEP_CELL


def test_srun_children_reuse_the_parents_runtime_view_instead_of_recreating_it():
    # turing_rl_prepare_runtime keys its work directory off SLURM_JOB_ID alone and hard-fails
    # if it already exists. rl_generator_run_9b.sh prepares the runtime and then sruns BOTH
    # the judge server and the trainer inside the SAME job, so an unconditional bootstrap in
    # either child collides with the parent. Both failure modes were observed:
    #   18499/18500 -- the judge step died ~12 s in
    #   18502       -- the judge came up and published its endpoint, then the trainer died
    # The parent exports TURING_RL_WORK_ROOT, so guard on it: present means reuse, absent
    # means this script is the top-level job and must prepare its own view.
    for name, src in (("judge_serve_9b_replicas.sh", SERVE), ("rl_generator_train_9b.sh", S)):
        assert 'if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then' in src, f"{name} lacks the guard"
        guard_at = src.index('if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then')
        bootstrap_at = src.index("cluster_job_bootstrap.sh")
        assert guard_at < bootstrap_at, f"{name}: bootstrap must be inside the guard"
    # The driver is the top-level script and still prepares one unconditionally.
    assert "cluster_job_bootstrap.sh" in RUN_2NODE
    driver_guard = RUN_2NODE.find('if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then')
    assert driver_guard == -1, "the driver IS the top-level script; it must not guard"


def test_runtime_view_guard_actually_fires_in_both_children():
    # Functional counterpart: run the real guard both ways. A static check would pass even
    # if the condition were inverted.
    for name, src in (("judge_serve_9b_replicas.sh", SERVE), ("rl_generator_train_9b.sh", S)):
        start = src.index('if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then')
        guard = src[start : src.index("fi", start) + 2]
        script = (
            'set -uo pipefail\n'
            'TURING_RL_CODE_ROOT=/nonexistent\n'
            'source() { echo SOURCED; }\n' + guard + '\necho DONE'
        )
        parent = subprocess.run(
            ["bash", "-c", "TURING_RL_WORK_ROOT=/some/work/root\n" + script],
            capture_output=True, text=True,
        )
        assert "SOURCED" not in parent.stdout, f"{name}: must not re-prepare under a parent"
        assert "DONE" in parent.stdout, f"{name}: {parent.stderr}"

        standalone = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
        assert "SOURCED" in standalone.stdout, f"{name}: must prepare its own view standalone"
