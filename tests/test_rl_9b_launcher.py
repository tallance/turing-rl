# tests/test_rl_9b_launcher.py
import os
import pathlib
import re
import subprocess
import sys

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
    # save_freq is derived (6 * SAVE_EVERY_EPOCHS), defaulting to 6 = one epoch; test_freq
    # stays at 6 so every saved ckpt lands on a validated step whatever the save cadence.
    # 352 = 50% of the 705-row val split, and at seed 42 is the same subset the half-data run
    # used, so val is comparable across runs.
    #
    # Both frac10 modes share one arm, so the slice cannot drift between them; only the epoch
    # count differs, and it is derived from the mode name rather than written twice.
    for mode in ("frac10ep3", "frac10ep10", "frac10ep20"):
        arm = mode_arm(mode)
        for k in (
            "data.train_max_samples=384",
            "data.val_max_samples=352",
            "trainer.total_epochs=$_EPOCHS",
            "trainer.save_freq=$_SAVE_FREQ",
            "trainer.test_freq=6",
            "trainer.val_before_train=True",
            "trainer.max_actor_ckpt_to_keep=null",
        ):
            assert k in arm, f"{mode} must pin {k}"
        # save_freq is a whole number of epochs, so the epoch-end hook would double-save.
        assert "export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0" in arm
        # Derived from an epoch count, never a bare literal: 6 steps/epoch is the one place
        # that relationship is written down.
        assert "SAVE_EVERY_EPOCHS=${SAVE_EVERY_EPOCHS:-1}" in arm
        assert "_SAVE_FREQ=$((6 * SAVE_EVERY_EPOCHS))" in arm
        # Batch size comes from the 9B config, not this arm; setting it here would add a
        # second variable versus full5.
        assert "data.train_batch_size=" not in arm
        assert "_EPOCHS=${MODE#frac10ep}" in arm
    assert "overfit|full|epoch1|full5|frac10ep3|frac10ep10|frac10ep20" in RUN_2NODE


def test_frac10_epoch_count_comes_from_the_mode_name():
    # The whole point of sharing one arm: `frac10ep20` must resolve to 20 epochs without a
    # second hard-coded literal that could drift from the mode it is named after.
    script = 'MODE="$1"\n_EPOCHS=${MODE#frac10ep}\necho "$_EPOCHS"\n'
    for mode, expected in (("frac10ep3", "3"), ("frac10ep10", "10"), ("frac10ep20", "20")):
        proc = subprocess.run(["bash", "-c", script, "_", mode], capture_output=True, text=True)
        assert proc.stdout.strip() == expected, f"{mode} -> {proc.stdout!r}"


def assert_judge_is_accepted(judge: str) -> None:
    """Assert `judge` appears in BOTH places JUDGE is validated.

    The :? default message and the explicit case guard are separate lists that must agree;
    a judge missing from either is rejected before its arm is ever reached.
    """
    msg = re.search(r"JUDGE=\$\{JUDGE:\?set JUDGE=([\w.|-]+)\}", RUN_2NODE)
    assert msg, "no JUDGE=${JUDGE:?...} validation message"
    guard = re.search(r'case "\$JUDGE" in ([\w.|-]+)\) ;;', RUN_2NODE)
    assert guard, "no `case $JUDGE` validation guard"
    assert msg.group(1).split("|") == guard.group(1).split("|"), (
        f"the two JUDGE lists disagree: {msg.group(1)!r} vs {guard.group(1)!r}"
    )
    assert judge in guard.group(1).split("|"), f"{judge} is not an accepted JUDGE"


def _save_freq(save_every, epochs):
    """Run the ARM'S OWN save-cadence lines; returns (stdout, exit code).

    Lifted verbatim out of the script rather than reimplemented here. An earlier version of
    this helper transcribed the arithmetic, and deleting the divisibility guard from the
    script left every one of these tests passing -- it was checking the transcription.
    """
    arm = mode_arm("frac10ep20")
    block = re.search(
        r"(SAVE_EVERY_EPOCHS=\$\{SAVE_EVERY_EPOCHS:-1\}.*?exit 5; \})", arm, re.S
    )
    assert block, "the frac10 arm no longer contains a save-cadence block to exercise"
    script = f'MODE=frac10ep20\n_EPOCHS={epochs}\n{block.group(1)}\necho "$_SAVE_FREQ"\n'
    env = None
    if save_every is not None:
        env = {**os.environ, "SAVE_EVERY_EPOCHS": save_every}
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env)
    return proc.stdout.strip(), proc.returncode


def test_save_every_epochs_defaults_to_one_epoch():
    # The default must reproduce the every-epoch cadence both completed frac10 runs used;
    # 20 checkpoints at 6,12,...,120 is what their artifacts on disk actually are.
    for save_every in (None, "1"):
        out, rc = _save_freq(save_every, 20)
        assert (out, rc) == ("6", 0), f"SAVE_EVERY_EPOCHS={save_every} -> {out!r} rc={rc}"


def test_frac10ep3_saves_every_epoch_by_default():
    # frac10ep3 is the short pre-collapse arm: 3 epochs x 6 steps = 18 steps. The default
    # cadence must give a checkpoint per epoch (6, 12, 18) and must land on the final step,
    # or the only policy anyone wants to evaluate is never written.
    out, rc = _save_freq(None, 3)
    assert (out, rc) == ("6", 0)
    assert (6 * 3) % int(out) == 0
    assert (6 * 3) // int(out) == 3, "expected exactly 3 checkpoints"


def test_save_every_epochs_scales_the_grid():
    # 19 GB per checkpoint: every-epoch frac10ep20 is 365 GB and the quota killed job 19509
    # mid-write. N=2 halves it to 10 checkpoints, which is also the cadence eval consumes.
    assert _save_freq("2", 20) == ("12", 0)
    assert _save_freq("4", 20) == ("24", 0)
    assert _save_freq("2", 10) == ("12", 0)


def test_save_every_epochs_refuses_a_grid_that_would_drop_the_final_checkpoint():
    # save_freq must divide the total step count. 20 epochs = 120 steps: N=3 gives 18, and
    # 120 % 18 != 0, so the run would end at step 120 having last saved at 108 -- losing the
    # final policy, which is the one every downstream eval wants.
    assert _save_freq("3", 20)[1] == 5
    assert _save_freq("4", 10)[1] == 5     # 60 % 24 != 0
    assert _save_freq("7", 20)[1] == 5     # 120 % 42 != 0


def test_ce_trained_judge_is_served_from_the_evaluated_checkpoint():
    # The CE judge is a local merged checkpoint, not an HF id. This is the exact path eval
    # cell judge-9b-ce-st scored (0.802 single-token vs 0.541 zero-shot), so the training
    # reward and the published accuracy refer to the same weights.
    assert (
        "JUDGE_MODEL=/home/lancewicki/projects/turing-rl/checkpoints/sft/"
        "judge_qwen35_9b_ce_dense" in RUN_2NODE
    )
    # Same serving shape as the zero-shot 9B: one node, 8 TP=1 replicas.
    ce_arm = re.search(r"^  9b-ce\)(.*?)^  397b\)", RUN_2NODE, re.M | re.S)
    assert ce_arm, "no 9b-ce arm in the judge case block"
    assert "TP=1" in ce_arm.group(1) and "DP=8" in ce_arm.group(1)
    # Absolute, because the judge step runs under the runtime view where checkpoints/ is a
    # symlink and a relative path would depend on the child's cwd.
    assert "JUDGE_MODEL=checkpoints/sft/judge_qwen35_9b_ce_dense" not in RUN_2NODE


def test_ce_judge_passes_both_judge_validation_gates():
    # JUDGE is validated twice -- the :? message and the explicit case guard. A value missing
    # from either is rejected before the judge case block ever runs.
    assert "JUDGE=${JUDGE:?set JUDGE=0.8b|9b|9b-ce|9b-ce2|9b-ce3|397b|gemma4-12b}" in RUN_2NODE
    assert 'case "$JUDGE" in 0.8b|9b|9b-ce|9b-ce2|9b-ce3|397b|gemma4-12b) ;;' in RUN_2NODE
    # And an unknown judge still fails fast rather than serving something unintended.
    guard = ('case "$JUDGE" in 0.8b|9b|9b-ce|9b-ce2|9b-ce3|397b|gemma4-12b) ;; '
             '*) echo "bad JUDGE=$JUDGE" >&2; exit 2 ;; esac')
    assert guard in RUN_2NODE
    for judge, rc in (("9b-ce", 0), ("9b-ce2", 0), ("9b-ce3", 0), ("9b", 0), ("9b-CE", 2), ("ce", 2), ("", 2)):
        proc = subprocess.run(
            ["bash", "-c", f'JUDGE="{judge}"\n{guard}\nexit 0'], capture_output=True, text=True
        )
        assert proc.returncode == rc, f"JUDGE={judge!r} -> rc={proc.returncode}, want {rc}"


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
    # Same default on the two-node driver. It must stay 1: both completed arms ran with
    # thinking on, so a silent flip would make every future run incomparable to them.
    assert 'PERSONA_JUDGE_ENABLE_THINKING="${PERSONA_JUDGE_ENABLE_THINKING:-1}"' in RUN_2NODE
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
    # Assert membership, not the exact list: pinning the whole alternation here made every
    # newly added judge fail this gemma-shape test for an unrelated reason. The list itself
    # is pinned once, in test_ce_judge_passes_both_judge_validation_gates.
    assert_judge_is_accepted("gemma4-12b")
    arm = judge_arm("gemma4-12b")
    assert "JUDGE_MODEL=google/gemma-4-12B-it" in arm
    assert "TP=1" in arm and "DP=8" in arm
    assert "REASONING_PARSER=gemma4" in arm


def test_smallest_judge_reuses_the_qwen_serving_path_unchanged():
    # 0.8B is the first judge below 4B anyone has run here, but it needs no new serving code:
    # same family, same parser, same TP=1/DP=8 shape as the 9B. The risk it carries is
    # whether a model this small can emit the 37-field verdict under training's
    # {"type": "json_object"} mode at all, which is a runtime question the smoke answers --
    # nothing static can. What this test pins is only that we did not accidentally give it
    # the gemma treatment or a whole-node TP.
    arm = judge_arm("0.8b")
    assert "JUDGE_MODEL=Qwen/Qwen3.5-0.8B" in arm
    assert "TP=1" in arm and "DP=8" in arm
    assert "REASONING_PARSER=qwen3" in arm
    assert "gemma" not in arm


def test_judge_smoke_battery_is_model_agnostic():
    # The battery was written for gemma and is reused verbatim for any other judge, so the
    # model must come from the environment rather than being baked in, and the one
    # gemma-only assertion (the offline snapshot pin, which only the gemma serve branch
    # produces) must not fire for models that have no such pin.
    smoke = pathlib.Path("scripts/slurm/gemma4_judge_training_smoke.sh").read_text()
    assert "MODEL=${SMOKE_MODEL:-google/gemma-4-12B-it}" in smoke
    assert "REASONING_PARSER=${SMOKE_PARSER:-gemma4}" in smoke
    assert 'if [ "$REASONING_PARSER" = gemma4 ]; then' in smoke
    # gate 3 guards on its reference dump existing. Note that dump's path names gemma's sweep
    # output regardless of the model under test, so for any other judge the gate still RUNS
    # and measures schema-mode parse rate -- it just is not an equivalence test, which is why
    # the summary labels the rating comparison CROSS-JUDGE rather than passing or failing it.
    assert 'if compgen -G "$EVAL_DUMP" > /dev/null; then' in smoke
    # output must not land in gemma's directory for a non-gemma judge
    assert "results/gemma4-judge-smoke/" not in smoke


def test_reasoning_parser_is_pinned_per_family_and_forwarded_to_the_judge_step():
    # The boundary detector is family-specific. A qwen3 parser on a gemma server does not
    # error -- it mis-splits thinking text out of .content, and the reward path then fails to
    # parse with nothing in the log naming the cause. So it must be resolved from JUDGE and
    # passed explicitly into the judge srun, never inherited from the submitting shell.
    assert "REASONING_PARSER=qwen3" in judge_arm("0.8b")
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
    start = RUN_2NODE.index('case "$JUDGE" in\n  0.8b)')
    block = RUN_2NODE[start : RUN_2NODE.index("esac", start) + len("esac")]
    script = 'JUDGE="$1"\n' + block + '\necho "$JUDGE_MODEL|$TP|$DP|$REASONING_PARSER"'
    for judge, expected in (
        # 0.8b carries a dot, which is a literal in a `case` pattern but not in a glob --
        # worth executing rather than eyeballing.
        ("0.8b", "Qwen/Qwen3.5-0.8B|1|8|qwen3"),
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


def test_driver_pins_env_file_for_the_python_secret_loader():
    # Sourcing .env populates the shell, but shared/load_env.py:get_openai_api_key calls
    # load_local_env() BEFORE looking at the process environment and raises if no .env FILE
    # exists. Its default search resolves relative to its own module path, and .resolve()
    # follows the runtime view's symlinks into the immutable source snapshot, which by design
    # holds no secrets. Job 18570 resumed from global_step_60 correctly and then died 9 min
    # later on the first reward call with "Missing OPENAI_API_KEY/OPENROUTER_API_KEY".
    assert 'export ENV_FILE="$REPO/.env"' in RUN_2NODE
    assert "judge secret file pinned:" in RUN_2NODE
    # Respect an explicit override rather than clobbering it.
    assert 'if [ -z "${ENV_FILE:-}" ] && [ -f "$REPO/.env" ]; then' in RUN_2NODE


def test_env_file_override_actually_reaches_the_loader():
    # Functional: prove ENV_FILE is honoured by the real loader when the module-relative
    # candidate does not exist. Asserting the export string alone would not catch the loader
    # changing its lookup.
    import tempfile
    import textwrap

    with tempfile.TemporaryDirectory() as tmp:
        envfile = pathlib.Path(tmp) / "secrets.env"
        envfile.write_text("OPENAI_API_KEY=sk-test-value\n")
        prog = textwrap.dedent(
            f"""
            import os, sys
            sys.path.insert(0, {str(pathlib.Path.cwd())!r})
            os.environ["ENV_FILE"] = {str(envfile)!r}
            os.environ.pop("OPENAI_API_KEY", None)
            from shared.load_env import get_openai_api_key
            print(get_openai_api_key())
            """
        )
        proc = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True)
        assert proc.returncode == 0, f"loader rejected ENV_FILE: {proc.stderr[-800:]}"
        assert "sk-test-value" in proc.stdout, proc.stdout


# --- single-token judge protocol ---------------------------------------------------------


def test_single_token_style_clears_the_reasoning_parser():
    """The single-token judge decodes ONE token with thinking off, so there is no <think>
    block to split. judge_sweep_cell.sh likewise adds a parser only for thinking-on cells,
    and that is the configuration the seven-cell matrix actually ran."""
    m = re.search(r"^  single_token\)(.*)$", RUN_2NODE, re.M)
    assert m, "no case arm handles JUDGE_PROMPT_STYLE=single_token"
    arm = m.group(1)
    assert 'REASONING_PARSER=""' in arm
    # ...and thinking is turned off for the reward path too, so the run reports what it did.
    assert "PERSONA_JUDGE_ENABLE_THINKING=0" in arm


def test_single_token_style_is_exported_to_the_trainer_step():
    """reward.py reads JUDGE_PROMPT_STYLE; without the export the trainer silently scores
    with the 37-field judge and the run measures the wrong protocol."""
    assert "export JUDGE_PROMPT_STYLE" in RUN_2NODE


def test_unknown_prompt_style_is_rejected_before_any_gpu_is_allocated():
    assert "JUDGE_PROMPT_STYLE must be full|single_token" in RUN_2NODE


def test_prompt_style_defaults_to_full():
    assert "JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE:-full}" in RUN_2NODE


def test_serve_omits_the_parser_flag_when_it_is_empty():
    """An empty REASONING_PARSER must drop --reasoning-parser entirely; passing the flag
    with an empty value is a vLLM startup error, not a no-op."""
    # ${VAR-default}, never ${VAR:-default}: the driver passes REASONING_PARSER="" to mean
    # "no parser", and the colon form treats empty as unset and restores qwen3. Job 19418
    # served with reasoning_parser='qwen3' while its driver printed parser='' two lines up.
    assert "REASONING_PARSER=${REASONING_PARSER-qwen3}" in SERVE
    assert "REASONING_PARSER=${REASONING_PARSER:-" not in SERVE
    assert 'RP=()' in SERVE
    assert '[ -n "$REASONING_PARSER" ] && RP=(--reasoning-parser "$REASONING_PARSER")' in SERVE
    # The flag appears ONCE, building that array -- neither serve invocation (qwen, gemma)
    # still passes it directly, and both consume the array instead.
    assert SERVE.count('--reasoning-parser "$REASONING_PARSER"') == 1
    assert SERVE.count('"${RP[@]}"') == 2
