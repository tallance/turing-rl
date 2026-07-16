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
# GRPO trainer for the RL-generator-vs-fixed-judge probe. Invokes run_verl_main_ppo directly
# (option A) with explicit overrides + the reward env inherited from the driver. Scancels the
# judge serve job on exit.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

# Teardown the judge when training ends (success or failure).
JUDGE_JOB_ID=${RL_JUDGE_JOB_ID:-}
cleanup() { [ -n "$JUDGE_JOB_ID" ] && scancel "$JUDGE_JOB_ID" 2>/dev/null || true; }
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

MODE=${RL_MODE:?}; JUDGE=${RL_JUDGE:-9b}; RUN_TAG=${RL_RUN_TAG:-${JUDGE}_${MODE}}
CKPT_DIR=${RL_CKPT_DIR:-$REPO/results/grpo/rl-generator/$RUN_TAG/checkpoints}
SFT_ADAPTER_PATH=${SFT_ADAPTER_PATH:-checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final}
DATA_BASE=data/prism/full_s42_history_sft40_grpo60_test10/grpo
TRAIN_FILE=${TRAIN_FILE:-$DATA_BASE/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_BASE/val.parquet}
OVERFIT10=${OVERFIT10:-$DATA_BASE/train_overfit10.parquet}
[ "$MODE" = overfit ] && TRAIN_FILE="$OVERFIT10"
EXP="qwen3-8b-grpo-turing-${JUDGE}-${MODE}"

echo "============================================"
echo "RL-gen trainer: MODE=$MODE JUDGE=$JUDGE endpoint=$OPENAI_API_BASE judge_model=$JUDGE_MODEL"
echo "cap=$TURING_JUDGE_SCORE_CLIP_MAX sampling=$PERSONA_JUDGE_SAMPLING thinking=$PERSONA_JUDGE_ENABLE_THINKING"
echo "sft_adapter=$SFT_ADAPTER_PATH ckpt=$CKPT_DIR reward_dump=$PERSONA_REWARD_DUMP_DIR"
echo "judge_job(to teardown)=$JUDGE_JOB_ID host=$(hostname) date=$(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Always-on overrides (option A). use_remove_padding=false at all 3 sites (no flash_attn in our env).
# The third site is critic.model.use_remove_padding (yaml line 95), NOT actor_rollout_ref.ref
# (the ref block has no such key; ref inherits actor_rollout_ref.model.use_remove_padding).
OVR=(
  actor_rollout_ref.model.lora_adapter_path="$SFT_ADAPTER_PATH"
  actor_rollout_ref.model.use_remove_padding=false
  actor_rollout_ref.actor.use_remove_padding=false
  critic.model.use_remove_padding=false
  trainer.default_local_dir="$CKPT_DIR"
  trainer.experiment_name="$EXP"
  trainer.project_name="${WANDB_PROJECT:-2026-07-15-rl-generator-vs-fixed-judge}"
  trainer.resume_mode=auto
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
)
case "$MODE" in
  overfit)
    OVR+=(
      data.train_batch_size="${OVERFIT_TRAIN_BATCH:-10}"
      actor_rollout_ref.actor.ppo_mini_batch_size="${OVERFIT_PPO_MINI:-10}"
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${OVERFIT_PPO_MICRO:-1}"
      trainer.total_epochs="${OVERFIT_EPOCHS:-40}"
      trainer.save_freq="${OVERFIT_SAVE_FREQ:-100000}"   # effectively disable mid-run saves for overfit
    ) ;;
  epoch1) OVR+=( trainer.total_epochs=1 ) ;;
  full)   : ;;   # base config (3 epochs, batch 64)
esac

echo "+ $PY -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs --config-name qwen3_8b_grpo_turing ${OVR[*]} ${EXTRA_OVERRIDES:-}"
$PY -m training.grpo.run_verl_main_ppo \
  --config-dir training/grpo/configs \
  --config-name qwen3_8b_grpo_turing \
  "${OVR[@]}" ${EXTRA_OVERRIDES:-}
RC=$?
echo "=== trainer exit: $RC ==="
exit $RC
