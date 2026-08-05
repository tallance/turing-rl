#!/bin/bash
#SBATCH --job-name=rl_gen_1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/rl_gen_1node-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# SINGLE-NODE B0 smoke for the RL-generator-vs-fixed-judge probe: judge + trainer
# share ONE 8-GPU node (fits the 8-GPU headroom under a 24-GPU QOS cap while a
# separate 16-GPU job runs). Judge (frozen 9B, vLLM TP=1) is pinned to GPU 7 and
# backgrounded; the veRL 9B trainer runs on GPUs 0-6 (RL_NGPUS=7, rollout TP=1).
# Ray assigns its seven logical GPU resources as physical ordinals 0-6, so keeping
# that mapping literal avoids silently placing trainer rank 0 on top of the judge.
# Endpoint handed off via a shared file on FSx home + OPENAI_API_BASE; the judge
# process is killed when the trainer exits (trap). 1-node analogue of
# rl_generator_run_9b.sh (which splits judge/trainer across 2 nodes).
#
# Submit: OVERFIT_EPOCHS=8 RUN_TAG=9b_b0_spike \
#           sbatch --export=ALL scripts/slurm/rl_generator_run_9b_1node.sh
#   B0_ROLLOUT_SYNC=1 (default) turns on the Step-3b rollout-sync hook (rollout_sync.json).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"

# wandb: source .env for WANDB_API_KEY + WANDB_BASE_URL, then force online + the
# self-hosted endpoint (same recipe as rl_generator_run_9b.sh / the working SFT runs).
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"
export WANDB_MODE=online
# wandb run dir on node-local tmpfs, not FSx: an FSx stall can wedge wandb's writer so the tail
# never lands even in the LOCAL transaction log (job 13634 lost 2 train steps + 1 validation that
# way, unrecoverable by `wandb sync`). cleanup() copies it back and syncs on exit.
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-${SLURM_JOB_ID:-$$}}"
mkdir -p "$WANDB_DIR"
WANDB_BIN=${WANDB_BIN:-/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/wandb}

# Fixed 9B config for the B0 spike (single node, frozen 9B judge).
JUDGE=9b
MODE=${MODE:-overfit}
JUDGE_MODEL=Qwen/Qwen3.5-9B
RUN_TAG=${RUN_TAG:-9b_b0_spike}
RUN_DIR=$REPO/results/grpo/rl-generator/$RUN_TAG
ENDPOINT_FILE=$RUN_DIR/judge_endpoint.txt
REWARD_DUMP_DIR=$RUN_DIR/reward_dump
CKPT_DIR=$RUN_DIR/checkpoints
mkdir -p "$RUN_DIR" "$REWARD_DUMP_DIR" "$CKPT_DIR" "$REPO/logs"
rm -f "$ENDPOINT_FILE"

# Judge on GPU 7 uses the EXISTING turing-rl-train env vllm (0.18, same as the Arm A
# judge). Trainer on GPUs 0-6 uses the Arm-B env that rl_generator_train_9b.sh points at.
JUDGE_VLLM=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/vllm
PORT=${PORT:-$((8300 + ${SLURM_JOB_ID:-0} % 400))}

echo ">> RL-gen 1-node B0 run: JUDGE=$JUDGE MODEL=$JUDGE_MODEL MODE=$MODE job=${SLURM_JOB_ID:-none}"
echo ">> node=$(hostname) judge=GPU7 trainer=GPU0-6 port=$PORT run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- judge step on GPU 7 (concurrent, backgrounded; frozen 9B judge, thinking-on) ---
CUDA_VISIBLE_DEVICES=7 \
  HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$JUDGE_VLLM" serve "$JUDGE_MODEL" \
    --tensor-parallel-size 1 --data-parallel-size 1 --port "$PORT" \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
    --download-dir /home/lancewicki/data/hf_cache &
JUDGE_PID=$!
save_wandb() {
  [ -d "$WANDB_DIR" ] || return 0
  cp -r "$WANDB_DIR" "$REPO/wandb/joblocal-${SLURM_JOB_ID:-$$}" 2>/dev/null || true
  for d in "$WANDB_DIR"/run-*; do
    [ -d "$d" ] || continue
    timeout 600 "$WANDB_BIN" sync "$d" 2>&1 | tail -2 || true
  done
}
cleanup() { save_wandb; kill "$JUDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

# --- wait for the judge to serve OUR model on /v1/models (up to 30 min warmup) ---
echo ">> waiting for judge /v1/models (up to 30 min warmup)..."
ok=0
for t in $(seq 1 900); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$JUDGE_MODEL\""; then
    ok=1; echo ">> judge ready after $((t*2))s"; break
  fi
  kill -0 "$JUDGE_PID" 2>/dev/null || { echo "ERROR: judge process died during warmup" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge" >&2; exit 4; }

# Single node: trainer reaches the judge on localhost. Publish + export the endpoint.
ENDPOINT="http://localhost:$PORT/v1"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
export OPENAI_API_BASE="$ENDPOINT"
echo ">> judge endpoint: $(cat "$ENDPOINT_FILE")"

# --- reward env for the trainer step (same recipe as rl_generator_run_9b.sh) ---
export REWARD_METRIC=turing
export JUDGE_MODEL
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export PERSONA_JUDGE_ENABLE_THINKING="${PERSONA_JUDGE_ENABLE_THINKING:-1}"
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS="${PERSONA_JUDGE_MAX_COMPLETION_TOKENS:-8192}"
export PERSONA_JUDGE_DUMP_RATE=1.0
export PERSONA_REWARD_DUMP_DIR="$REWARD_DUMP_DIR"
export PERSONA_EVAL_JUDGE_MODEL="$JUDGE_MODEL"
export PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY="${PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY:-128}"
export PERSONA_OPENAI_MAX_RETRIES="${PERSONA_OPENAI_MAX_RETRIES:-3}"
export WANDB_PROJECT="${WANDB_PROJECT:-2026-07-15-rl-generator-vs-fixed-judge}"
export MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"   # extra Hydra overrides (e.g. kl_loss_coef) -> trainer step
export RL_MODE="$MODE" RL_JUDGE="$JUDGE" RL_RUN_TAG="$RUN_TAG" RL_RUN_DIR="$RUN_DIR" RL_CKPT_DIR="$CKPT_DIR"
export RL_JUDGE_JOB_ID=""   # no separate judge job; teardown handled here by killing the judge process
export OVERFIT_EPOCHS="${OVERFIT_EPOCHS:-8}"   # short B0 spike (train script reads OVERFIT_EPOCHS)
# veRL requires train_batch_size * rollout.n to be divisible by actor DP size. The single-node
# layout gives the trainer 7 GPUs, so 7 prompts * n=4 = 28 is divisible by 7.
export OVERFIT_TRAIN_BATCH="${OVERFIT_TRAIN_BATCH:-7}"
export OVERFIT_PPO_MINI="${OVERFIT_PPO_MINI:-7}"
# TP=1 places the full ~17.7GB rollout model on each GPU, so 0.40 cannot leave any KV/GDN cache.
export RL_ROLLOUT_GPU_MEMORY_UTILIZATION="${RL_ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"

# --- trainer step on GPUs 0-6 (foreground). RL_NGPUS/RL_ROLLOUT_TP make the 9B
#     trainer use 7 GPUs + rollout TP=1 (env-overridable in rl_generator_train_9b.sh). ---
# Run from node-local disk, not FSx: bash reads a script LAZILY and keeps the handle open for the
# life of the job, so a transient FSx stale handle can kill a multi-day run mid-execution (job
# 13634 died exactly that way, after finishing all 32 steps). Same node here, so a plain copy works.
TRAIN_SH_LOCAL=/tmp/rl_gen_train-${SLURM_JOB_ID:-$$}.sh
cp "$REPO/scripts/slurm/rl_generator_train_9b.sh" "$TRAIN_SH_LOCAL" || exit 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 RL_NGPUS=7 RL_ROLLOUT_TP=1 B0_ROLLOUT_SYNC="${B0_ROLLOUT_SYNC:-1}" \
  bash "$TRAIN_SH_LOCAL"
RC=$?
echo "=== trainer step exit: $RC ; tearing down judge process ==="
kill "$JUDGE_PID" 2>/dev/null || true
exit $RC
