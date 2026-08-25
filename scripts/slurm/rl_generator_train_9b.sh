#!/bin/bash
#SBATCH --job-name=rl_gen_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/rl_gen_train-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Arm-B 9B GRPO trainer (Qwen3.5-9B) for the RL-generator-vs-fixed-judge probe.
# 9B variant of rl_generator_train.sh: same header, merged-dir guards, DATA_BASE/TRAIN_FILE/
# VAL_FILE, MODE=overfit branch and reward-env inheritance, but a SINGLE 9B OVR array +
# run_verl_main_ppo call (FSDP2 + Qwen3.5/GDN + merged-dense LoRA sync). Invokes
# run_verl_main_ppo directly (option A) with explicit overrides + the reward env inherited
# from the driver. Scancels the judge serve job on exit.
set -uo pipefail
# Prepare a runtime view only when we are the top-level job script -- see the identical
# guard in judge_serve_9b_replicas.sh. turing_rl_prepare_runtime keys its work directory
# off SLURM_JOB_ID and hard-fails if it exists, and rl_generator_run_9b.sh sruns BOTH the
# judge and this trainer inside one job. Job 18502 got the judge up and published its
# endpoint, then died here with "FATAL: runtime work directory already exists".
if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then
  source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO"
# Ray worker subprocesses inherit env vars but NOT cwd/sys.path, so `-m training...`
# resolving in the driver process is not enough — the workers must find the `training`
# package (custom reward + worker_process_setup_hook) via PYTHONPATH. (Repo is not pip-installed.)
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python   # pinned Arm-B env
$PY -c 'import transfer_queue' || {
  echo "ERROR: Arm-B veRL 0.9 env requires TransferQueue==0.1.8" >&2
  exit 2
}

# Teardown the judge when training ends (success or failure).
JUDGE_JOB_ID=${RL_JUDGE_JOB_ID:-}
cleanup() { [ -n "$JUDGE_JOB_ID" ] && scancel "$JUDGE_JOB_ID" 2>/dev/null || true; }
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

MODE=${RL_MODE:?}; JUDGE=${RL_JUDGE:-9b}; RUN_TAG=${RL_RUN_TAG:-${JUDGE}_${MODE}_merged_sft_ref}
CKPT_DIR=${RL_CKPT_DIR:-$REPO/results/grpo/rl-generator/$RUN_TAG/checkpoints}
MERGED_SFT_MODEL_PATH=${MERGED_SFT_MODEL_PATH:-checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
case "$MERGED_SFT_MODEL_PATH" in
  /*) MERGED_SFT_MODEL_DIR="$MERGED_SFT_MODEL_PATH" ;;
  *)  MERGED_SFT_MODEL_DIR="$REPO/$MERGED_SFT_MODEL_PATH" ;;
esac
for required in config.json tokenizer_config.json sft_merge_metadata.json; do
  [ -f "$MERGED_SFT_MODEL_DIR/$required" ] || {
    echo "ERROR: merged SFT model is missing $required under $MERGED_SFT_MODEL_DIR" >&2
    echo "Build it with: $PY scripts/merge_sft_adapter.py" >&2
    exit 2
  }
done
shopt -s nullglob
MERGED_WEIGHT_FILES=("$MERGED_SFT_MODEL_DIR"/model*.safetensors "$MERGED_SFT_MODEL_DIR"/pytorch_model*.bin)
shopt -u nullglob
[ "${#MERGED_WEIGHT_FILES[@]}" -gt 0 ] || {
  echo "ERROR: merged SFT model has no weights under $MERGED_SFT_MODEL_DIR" >&2
  exit 2
}
# The 9B recipe lives in qwen3_9b_grpo_turing.yaml. The 8B config this used to name carries
# 8B-era lr/kl/temperature that no 9B run has ever wanted, and relying on EXTRA_OVERRIDES to
# correct them at submit time is what silently mis-trained job 15143.
GRPO_CONFIG_NAME=${GRPO_CONFIG_NAME:-qwen3_9b_grpo_turing}
DATA_BASE=$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/grpo
TRAIN_FILE=${TRAIN_FILE:-$DATA_BASE/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_BASE/val.parquet}
OVERFIT10=${OVERFIT10:-$DATA_BASE/train_overfit10.parquet}
[ "$MODE" = overfit ] && TRAIN_FILE="$OVERFIT10"
EXP="qwen35-9b-grpo-turing-${RUN_TAG}"

echo "============================================"
echo "RL-gen 9B trainer: MODE=$MODE JUDGE=$JUDGE endpoint=$OPENAI_API_BASE judge_model=$JUDGE_MODEL"
echo "cap=$TURING_JUDGE_SCORE_CLIP_MAX sampling=$PERSONA_JUDGE_SAMPLING thinking=$PERSONA_JUDGE_ENABLE_THINKING"
echo "merged_sft_model=$MERGED_SFT_MODEL_PATH fresh_rl_lora=r64/alpha32 ckpt=$CKPT_DIR reward_dump=$PERSONA_REWARD_DUMP_DIR"
echo "judge_job(to teardown)=$JUDGE_JOB_ID host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# ---- 9B OVR array (single definition; REPLACES the 8B OVR + trainer call) ----
# `+key=...` appends keys not present in qwen3_8b_grpo_turing.yaml (override_config/mtp).
# Keys already present in the base yaml (lora.merge, strategy, chunked_prefill, checkpoint_engine,
# max_model_len, calculate_log_probs) take NO `+` — Hydra errors on `+` for an existing key.
OVR=(
  actor_rollout_ref.model.path="$MERGED_SFT_MODEL_PATH"
  actor_rollout_ref.model.lora_adapter_path=null
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  "actor_rollout_ref.model.exclude_modules='.*(visual|mtp).*'" # preserve regex quotes for Hydra
  +actor_rollout_ref.model.override_config.text_config.mtp_num_hidden_layers=0  # belt: disable MTP on actor/ref
  actor_rollout_ref.model.lora.merge=True                  # key already exists; merged dense sync
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  # FSDP2 + Qwen3.5/GDN requirements (from veRL's 27B FSDP2 recipe) --------------------------------
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${RL_NGPUS:-8}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.offload_policy=True    # FSDP2 offload policy (official 27B recipe)
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  # rollout (vLLM) --------------------------------------------------------------------------------
  actor_rollout_ref.rollout.tensor_model_parallel_size=${RL_ROLLOUT_TP:-1}
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True   # REQUIRED: prompts ~12.5k > 4096 batch cap
  actor_rollout_ref.rollout.max_model_len=13524
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}
  actor_rollout_ref.rollout.calculate_log_probs=True       # feeds the B0 logprob-parity guard
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072   # no + (key exists)
  actor_rollout_ref.rollout.agent.num_workers=${RL_NGPUS:-8}  # rollout batch must chunk evenly in V0
  # veRL 0.9 V1 does not migrate the legacy top-level custom_reward_function block.
  reward.custom_reward_function.path=training/grpo/reward.py
  reward.custom_reward_function.name=compute_score
  # trainer / data --------------------------------------------------------------------------------
  trainer.default_local_dir="$CKPT_DIR"
  trainer.experiment_name="$EXP"
  trainer.project_name="${WANDB_PROJECT:-2026-07-15-rl-generator-vs-fixed-judge}"
  trainer.resume_mode=auto
  # veRL 0.9 keeps the V0 DataProto controller on the new unified engine workers.
  # turing-rl's runtime patches and B0 parity hook target that controller, while
  # V1 uses a separate TransferQueue/KVBatchMeta update path.
  trainer.use_v1=False
  trainer.n_gpus_per_node=${RL_NGPUS:-8}
  trainer.nnodes=1
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
)
# The B0 fixed-sequence delta probe captures the first actor-DP-sized rows before
# and after the trainer assembles its batch. Disable length balancing so those
# rows remain identical across the vLLM and HF scoring paths.
if [ "${B0_ROLLOUT_SYNC:-0}" = "1" ]; then
  OVR+=( trainer.balance_batch=False )
fi
# TURING_JUDGE_SCORE_CLIP_MAX=7 (no cap) and PERSONA_* inherited from the driver, unchanged.
case "$MODE" in
  overfit)
    # Overfit probes run ~50 epochs and only need the reward dumps, not checkpoints.
    # The epoch-end checkpoint hook (PERSONA_ENABLE_EPOCH_END_CHECKPOINTING, default on)
    # otherwise writes a full ckpt EVERY epoch -> blows the ~1TB quota. Disable it for
    # overfit ONLY; full runs (few epochs) keep epoch-end saves. save_freq below already
    # disables regular mid-run saves.
    export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0
    OVR+=(
      data.train_batch_size="${OVERFIT_TRAIN_BATCH:-10}"
      actor_rollout_ref.actor.ppo_mini_batch_size="${OVERFIT_PPO_MINI:-10}"
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${OVERFIT_PPO_MICRO:-1}"
      trainer.total_epochs="${OVERFIT_EPOCHS:-40}"
      trainer.save_freq="${OVERFIT_SAVE_FREQ:-100000}"   # effectively disable mid-run saves for overfit
    ) ;;
  epoch1) OVR+=( trainer.total_epochs=1 ) ;;
  full)   : ;;   # base config (few epochs)
  full5)
    # Full-dataset production run: 4174 train rows / batch 64 = 65 steps/epoch, 325 steps total.
    # Everything cadence-related is PINNED here rather than left to EXTRA_OVERRIDES, for the same
    # reason judge concurrency is pinned in rl_generator_run_9b.sh: job 13634 silently ran with an
    # inherited value and nothing in the repo recorded it.
    #
    # save_freq=32 -> ckpts at 32,64,...,320 (~12.3 h apart). A checkpoint 32 steps INTO each epoch
    # (step = 32 mod 65) is not expressible: save_freq is a plain multiple, and the epoch-end hook
    # (verl_runtime_patch.py:_resolve_epoch_aligned_save_freq) returns exactly steps_per_epoch with
    # no offset. save_freq=32 lands within 5 steps of those targets, so the hook is turned OFF here
    # -- left on it would fire at 65,130,195,260,325, i.e. 5 near-duplicate saves 1-5 steps after
    # the save_freq ones (~95 GB of redundant weights).
    #
    # test_freq=32 puts validation on the SAME step grid as the checkpoints, so every saved ckpt
    # has a val score. Full 705-row val split at ~0.225 req/s is ~52 min per pass, 11 passes.
    export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0
    OVR+=(
      trainer.total_epochs=5
      trainer.save_freq=32
      trainer.test_freq=32
      trainer.val_before_train=True
      # veRL's own default is null (keep all). 13634 was submitted with 6, which would silently
      # delete the first four of this run's ten checkpoints.
      trainer.max_actor_ckpt_to_keep=null
    ) ;;
  frac10ep10|frac10ep20)
    # Low-compute contrast arm to full5: 10% of the train split, one checkpoint and one
    # validation pass per epoch. The question is whether the smaller slice overfits MORE,
    # which reads off train/val divergence WITHIN the run (many views per sample vs full5's 5).
    # This is NOT a matched-compute comparison: even at 20 epochs, 120 steps is ~37% of full5's
    # 325, so a final-val gap between the runs must not be read as coverage-vs-repetition.
    #
    # The epoch count is the mode-name suffix: frac10ep10 -> 10, frac10ep20 -> 20. Everything
    # else is identical, so the two modes cannot drift apart in the parts that define the slice.
    _EPOCHS=${MODE#frac10ep}
    #
    # 384 = 6 x 64, i.e. 9.2% of the 4174-row train split, NOT the round 10% (417).
    # The train loader sets drop_last=True (ray_trainer.py:409), so with 417 rows each epoch
    # would train on 384 and discard 33 -- and because the sampler reshuffles per epoch, a
    # DIFFERENT 33 each time, leaving samples seen 9.21 times per 10 epochs on average instead
    # of 10, and unevenly. 384 divides the batch exactly: nothing is dropped, every sample is
    # seen exactly once per epoch and exactly $_EPOCHS times over the run. Step count is
    # identical either way (6 steps/epoch), so the clean repeat costs nothing. Verified on the
    # completed 10-epoch run: 384 keys judged exactly 40x (10 epochs x 4 rollouts), no rotation.
    #
    # save_freq=test_freq=6 puts checkpoints and validation on the same epoch-aligned grid
    # (6,12,...,6*$_EPOCHS), so every saved ckpt has a val score. Because save_freq already
    # equals steps_per_epoch, the epoch-end hook is turned OFF -- left on it would fire at the
    # same steps and write a duplicate checkpoint per epoch (~19 GB each).
    #
    # 352 = 50% of the 705-row val split. At seed 42 this is the SAME subset the half-data run
    # used (data/.../grpo/val_used352.meta.json), so val is comparable across runs -- confirmed
    # byte-identical parquet (sha256 41611b08...) between the half-data and frac10 runs.
    # Subsampling is native to veRL (rl_dataset.py: rng.choice(total, size, replace=False)
    # under data.shuffle=true); no new parquet files are needed.
    #
    # RESUME: trainer.resume_mode=auto (set above) picks up the latest checkpoint under
    # trainer.default_local_dir, which is derived from RUN_TAG. So re-submitting frac10ep20
    # with the RUN_TAG of a completed frac10ep10 run continues it from global_step_60 rather
    # than starting over, while a fresh RUN_TAG trains from the SFT init. Extending the epoch
    # count is safe for the optimizer: lr_scheduler_type defaults to "constant" with
    # lr_warmup_steps_ratio 0.0, so the LR is flat at 1e-4 and does not depend on
    # total_training_steps. And 6*$_EPOCHS is a multiple of steps_per_epoch, so veRL skips the
    # dataloader-state restore at the boundary and the next epoch iterates from scratch.
    export PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0
    OVR+=(
      data.train_max_samples=384
      data.val_max_samples=352
      trainer.total_epochs=$_EPOCHS
      trainer.save_freq=6
      trainer.test_freq=6
      trainer.val_before_train=True
      trainer.max_actor_ckpt_to_keep=null
    )
    # Hydra gives the LAST occurrence of a key priority, and EXTRA_OVERRIDES is appended after
    # "${OVR[@]}" below. A stray ambient value -- sbatch --export=ALL propagates the submitting
    # shell, the same mechanism behind the 13634 incident -- would therefore silently beat every
    # value pinned above. Refuse to launch instead of running 21 h with a config nobody chose.
    for _protected in data.train_max_samples data.val_max_samples \
                      trainer.total_epochs trainer.save_freq trainer.test_freq \
                      trainer.val_before_train trainer.max_actor_ckpt_to_keep; do
      case " ${EXTRA_OVERRIDES:-} " in
        *" $_protected="*|*"+$_protected="*)
          echo "ERROR: EXTRA_OVERRIDES sets '$_protected', which MODE=$MODE pins." >&2
          echo "       EXTRA_OVERRIDES=${EXTRA_OVERRIDES}" >&2
          echo "       Unpinned keys are still allowed; remove this one and resubmit." >&2
          exit 5 ;;
      esac
    done ;;
esac

echo "+ $PY -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs --config-name $GRPO_CONFIG_NAME hydra.run.dir=$TURING_RL_HYDRA_DIR hydra.job.chdir=false ${OVR[*]} ${EXTRA_OVERRIDES:-}"
$PY -m training.grpo.run_verl_main_ppo \
  --config-dir training/grpo/configs \
  --config-name "$GRPO_CONFIG_NAME" \
  hydra.run.dir="$TURING_RL_HYDRA_DIR" \
  hydra.job.chdir=false \
  "${OVR[@]}" ${EXTRA_OVERRIDES:-}
RC=$?
echo "=== trainer exit: $RC ==="

# Ray keeps its per-worker logs under /tmp/ray on THIS node, and they vanish when the
# allocation is released. When a trainer dies without printing a traceback, the exception is
# in those files and nowhere else: job 18917 exited 1 mid-checkpoint with nothing in either
# Slurm stream, and this is what would have named the cause. It has to live here rather than
# in the driver's trap, because the driver runs on the judge node and Ray runs on this one.
# Best-effort and time-capped: a diagnostic must never be why a job fails, and it must not
# disturb $RC.
if [ "$RC" -ne 0 ]; then
  _raydest="$REPO/logs/ray-${SLURM_JOB_ID:-$$}"
  if mkdir -p "$_raydest" 2>/dev/null; then
    for _s in /tmp/ray/session_*/logs; do
      [ -d "$_s" ] || continue
      timeout 180 cp -r "$_s" "$_raydest/$(basename "$(dirname "$_s")")" 2>/dev/null || true
    done
    echo "=== ray logs saved to $_raydest ==="
  fi
fi

exit $RC
