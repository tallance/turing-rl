#!/bin/bash
#SBATCH --job-name=rl_gen
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/rl_gen-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Single ATOMIC 2-node GRPO run for the RL-generator-vs-fixed-judge probe.
# Slurm allocates BOTH nodes together (no idle-hold). node0 = DP judge
# (judge_serve_9b_replicas.sh), node1 = veRL trainer (rl_generator_train.sh).
# Endpoint handed off via a shared file on FSx home; the judge srun step is
# killed when the trainer srun step finishes.
#
# Submit: JUDGE=9b MODE=overfit OVERFIT_EPOCHS=2 sbatch --export=ALL scripts/slurm/rl_generator_run.sh
#   JUDGE = 9b | 397b     MODE = overfit | full | epoch1
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"

JUDGE=${JUDGE:?set JUDGE=9b|397b}
MODE=${MODE:?set MODE=overfit|full|epoch1}
case "$JUDGE" in 9b|397b) ;; *) echo "bad JUDGE=$JUDGE" >&2; exit 2 ;; esac
case "$MODE" in overfit|full|epoch1) ;; *) echo "bad MODE=$MODE" >&2; exit 2 ;; esac
case "$JUDGE" in
  9b)   JUDGE_MODEL=Qwen/Qwen3.5-9B;                  TP=1; DP=8 ;;
  397b) JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4; TP=8; DP=1 ;;
esac

# Two allocated nodes: node0 -> judge, node1 -> trainer.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" -ge 2 ] || { echo "ERROR: need 2 nodes, got '${NODES[*]:-none}'" >&2; exit 2; }
NODE_JUDGE=${NODES[0]}; NODE_TRAIN=${NODES[1]}

RUN_TAG=${RUN_TAG:-${JUDGE}_${MODE}}
RUN_DIR=$REPO/results/grpo/rl-generator/$RUN_TAG
ENDPOINT_FILE=$RUN_DIR/judge_endpoint.txt
REWARD_DUMP_DIR=$RUN_DIR/reward_dump
CKPT_DIR=$RUN_DIR/checkpoints
mkdir -p "$RUN_DIR" "$REWARD_DUMP_DIR" "$CKPT_DIR" "$REPO/logs"
rm -f "$ENDPOINT_FILE"

echo ">> RL-gen atomic run: JUDGE=$JUDGE MODEL=$JUDGE_MODEL MODE=$MODE job=$SLURM_JOB_ID"
echo ">> nodes: judge=$NODE_JUDGE trainer=$NODE_TRAIN  run_dir=$RUN_DIR"

# --- judge step on node0 (concurrent, backgrounded) ---
MODEL=$JUDGE_MODEL TP=$TP DP=$DP JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  srun --nodes=1 --ntasks=1 --nodelist="$NODE_JUDGE" --gres=gpu:8 --overlap \
  bash scripts/slurm/judge_serve_9b_replicas.sh &
JUDGE_PID=$!
cleanup() { kill "$JUDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

# --- wait for the judge to publish its endpoint (written only after model-verified health) ---
echo ">> waiting for judge endpoint (up to 60 min warmup)..."
ok=0
for t in $(seq 1 1800); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$JUDGE_PID" 2>/dev/null || { echo "ERROR: judge step died before publishing endpoint" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge endpoint" >&2; exit 4; }
ENDPOINT=$(cat "$ENDPOINT_FILE")
echo ">> judge endpoint: $ENDPOINT"

# --- reward env for the trainer step (inherited via srun --export=ALL / default) ---
export REWARD_METRIC=turing
export JUDGE_MODEL
export OPENAI_API_BASE="$ENDPOINT"
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
export SFT_ADAPTER_PATH="${SFT_ADAPTER_PATH:-checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final}"
export RL_MODE="$MODE" RL_JUDGE="$JUDGE" RL_RUN_TAG="$RUN_TAG" RL_RUN_DIR="$RUN_DIR" RL_CKPT_DIR="$CKPT_DIR"
export RL_JUDGE_JOB_ID=""   # no separate judge job; teardown handled here by killing the judge srun step

# --- trainer step on node1 (foreground) ---
srun --nodes=1 --ntasks=1 --nodelist="$NODE_TRAIN" --gres=gpu:8 --overlap \
  bash scripts/slurm/rl_generator_train.sh
RC=$?
echo "=== trainer step exit: $RC ; tearing down judge step ==="
kill "$JUDGE_PID" 2>/dev/null || true
exit $RC
