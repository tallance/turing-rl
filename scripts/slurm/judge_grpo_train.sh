#!/bin/bash
#SBATCH --job-name=judge_grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=3-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_grpo-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Judge-only GRPO. No judge server: the reward is local and label-verifiable, so unlike
# the generator runs there is nothing to serve and nothing to tear down.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
# Qwen3.5 needs transformers 5.x + veRL 0.9; this is the env both 9B GRPO runs used.
PY=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python
$PY -c 'import transfer_queue' || {
  echo "ERROR: veRL 0.9 env requires TransferQueue==0.1.8" >&2
  exit 2
}

JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:?set JUDGE_MODEL_PATH, e.g. Qwen/Qwen3.5-4B}
export JUDGE_REWARD_ARM=${JUDGE_REWARD_ARM:?set JUDGE_REWARD_ARM to directional or graded}
case "$JUDGE_REWARD_ARM" in
  directional|graded) ;;
  *) echo "ERROR: JUDGE_REWARD_ARM must be directional or graded, got $JUDGE_REWARD_ARM" >&2; exit 2 ;;
esac
export JUDGE_TASK_WEIGHT=${JUDGE_TASK_WEIGHT:-0.9}
export JUDGE_FORMAT_WEIGHT=${JUDGE_FORMAT_WEIGHT:-0.1}
# The judge REASONS before answering; this mirrors what generator RL ran against.
export PERSONA_JUDGE_ENABLE_THINKING=1

DATA_DIR=${DATA_DIR:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1}
TRAIN_FILE=${TRAIN_FILE:-$DATA_DIR/train.parquet}
VAL_FILE=${VAL_FILE:-$DATA_DIR/val.parquet}
RUN_TAG=${JUDGE_RUN_TAG:-$(basename "$JUDGE_MODEL_PATH")_${JUDGE_REWARD_ARM}}
CKPT_DIR=${CKPT_DIR:-$REPO/results/grpo/judge/$RUN_TAG/checkpoints}

echo "=== judge GRPO: model=$JUDGE_MODEL_PATH arm=$JUDGE_REWARD_ARM tag=$RUN_TAG ==="
echo "=== train=$TRAIN_FILE val=$VAL_FILE ckpt=$CKPT_DIR host=$(hostname) date=$(date) ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

OVR=(
  actor_rollout_ref.model.path="$JUDGE_MODEL_PATH"
  # `+` because no parent defines this key. Everything else the 9B recipe pins for a Qwen3.5
  # hybrid (lora.merge, checkpoint_engine bucket size) lives in the yaml, where it cannot be
  # dropped; this one stays here because the `+`-prefixed command-line form is the only
  # syntax proven to work for it (rl_generator_train_9b.sh:96).
  +actor_rollout_ref.model.override_config.text_config.mtp_num_hidden_layers=0
  # TP is the one rollout knob that legitimately varies by model size; everything else
  # (chunked prefill, gpu_memory_utilization, use_v1, reward routing) is pinned in
  # qwen35_judge_grpo.yaml so it cannot be dropped from a submit-time string.
  actor_rollout_ref.rollout.tensor_model_parallel_size=${RL_ROLLOUT_TP:-1}
  actor_rollout_ref.actor.fsdp_config.fsdp_size=${RL_NGPUS:-8}
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.offload_policy=True
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
  trainer.default_local_dir="$CKPT_DIR"
  trainer.experiment_name="qwen35-judge-grpo-$RUN_TAG"
  trainer.project_name=grpo-judge
)

# --config-dir is NOT optional: without it Hydra resolves --config-name against veRL's own
# packaged config directory and the job dies immediately with
# "Cannot find primary config 'qwen35_judge_grpo'". Both working trainers pass it.
echo "+ $PY -u -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs --config-name qwen35_judge_grpo hydra.run.dir=$TURING_RL_HYDRA_DIR hydra.job.chdir=false ${OVR[*]} ${EXTRA_OVERRIDES:-}"
$PY -u -m training.grpo.run_verl_main_ppo \
  --config-dir training/grpo/configs \
  --config-name qwen35_judge_grpo \
  hydra.run.dir="$TURING_RL_HYDRA_DIR" \
  hydra.job.chdir=false \
  "${OVR[@]}" ${EXTRA_OVERRIDES:-}
RC=$?
echo "=== trainer exit: $RC ==="
exit $RC
