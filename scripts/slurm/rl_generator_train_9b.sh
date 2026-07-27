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
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=/home/lancewicki/projects/turing-rl
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
DATA_BASE=data/prism/full_s42_history_sft40_grpo60_test10/grpo
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
  actor_rollout_ref.rollout.tensor_model_parallel_size=${RL_ROLLOUT_TP:-4}
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True   # REQUIRED: prompts ~12.5k > 4096 batch cap
  actor_rollout_ref.rollout.max_model_len=13524
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.40}
  actor_rollout_ref.rollout.calculate_log_probs=True       # feeds the B0 logprob-parity guard
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072   # no + (key exists)
  # trainer / data --------------------------------------------------------------------------------
  trainer.default_local_dir="$CKPT_DIR"
  trainer.experiment_name="$EXP"
  trainer.project_name="${WANDB_PROJECT:-2026-07-15-rl-generator-vs-fixed-judge}"
  trainer.resume_mode=auto
  trainer.n_gpus_per_node=${RL_NGPUS:-8}
  trainer.nnodes=1
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
)
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
esac

echo "+ $PY -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs --config-name qwen3_8b_grpo_turing ${OVR[*]} ${EXTRA_OVERRIDES:-}"
$PY -m training.grpo.run_verl_main_ppo \
  --config-dir training/grpo/configs \
  --config-name qwen3_8b_grpo_turing \
  "${OVR[@]}" ${EXTRA_OVERRIDES:-}
RC=$?
echo "=== trainer exit: $RC ==="
exit $RC
