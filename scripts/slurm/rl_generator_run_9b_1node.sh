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
# separate 16-GPU job runs). Judge (frozen 9B, vLLM TP=1) is pinned to GPU 0 and
# backgrounded; the veRL 9B trainer runs on GPUs 1-7 (RL_NGPUS=7, rollout TP=1).
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

# Judge on GPU 0 uses the EXISTING turing-rl-train env vllm (0.18, same as the Arm A
# judge). Trainer on GPUs 1-7 uses the Arm-B env that rl_generator_train_9b.sh points at.
JUDGE_VLLM=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/vllm
PORT=${PORT:-$((8300 + ${SLURM_JOB_ID:-0} % 400))}

echo ">> RL-gen 1-node B0 run: JUDGE=$JUDGE MODEL=$JUDGE_MODEL MODE=$MODE job=${SLURM_JOB_ID:-none}"
echo ">> node=$(hostname) judge=GPU0 trainer=GPU1-7 port=$PORT run_dir=$RUN_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# --- judge step on GPU 0 (concurrent, backgrounded; frozen 9B judge, thinking-on) ---
CUDA_VISIBLE_DEVICES=0 \
  HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "$JUDGE_VLLM" serve "$JUDGE_MODEL" \
    --tensor-parallel-size 1 --data-parallel-size 1 --port "$PORT" \
    --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
    --download-dir /home/lancewicki/data/hf_cache &
JUDGE_PID=$!
cleanup() { kill "$JUDGE_PID" 2>/dev/null || true; }
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
export PERSONA_JUDGE_ENABLE_THINKING=1
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
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

# --- trainer step on GPUs 1-7 (foreground). RL_NGPUS/RL_ROLLOUT_TP make the 9B
#     trainer use 7 GPUs + rollout TP=1 (env-overridable in rl_generator_train_9b.sh). ---
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 RL_NGPUS=7 RL_ROLLOUT_TP=1 B0_ROLLOUT_SYNC="${B0_ROLLOUT_SYNC:-1}" \
  bash scripts/slurm/rl_generator_train_9b.sh
RC=$?
echo "=== trainer step exit: $RC ; tearing down judge process ==="
kill "$JUDGE_PID" 2>/dev/null || true
exit $RC
