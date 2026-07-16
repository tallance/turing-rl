#!/bin/bash
# RL-generator run driver (login node). Submits the DP judge serve job + the trainer job,
# wires the judge endpoint + reward env into the trainer, then exits. Trainer owns judge teardown.
#
# Usage: JUDGE=9b MODE=overfit bash scripts/slurm/rl_generator_run.sh
#   JUDGE = 9b | 397b     MODE = overfit | full | epoch1
# Optional: RUN_TAG, SFT_ADAPTER_PATH, WANDB_PROJECT, and any OVERFIT_* / EXTRA_OVERRIDES (passed through).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"

JUDGE=${JUDGE:?set JUDGE=9b|397b}
MODE=${MODE:?set MODE=overfit|full|epoch1}
case "$JUDGE" in 9b|397b) ;; *) echo "bad JUDGE=$JUDGE" >&2; exit 2 ;; esac
case "$MODE" in overfit|full|epoch1) ;; *) echo "bad MODE=$MODE" >&2; exit 2 ;; esac

case "$JUDGE" in
  9b)   JUDGE_MODEL=Qwen/Qwen3.5-9B;                       TP=1; DP=8 ;;
  397b) JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4;      TP=8; DP=1 ;;
esac

RUN_TAG=${RUN_TAG:-${JUDGE}_${MODE}}
RUN_DIR=$REPO/results/grpo/rl-generator/$RUN_TAG
ENDPOINT_FILE=$RUN_DIR/judge_endpoint.txt
REWARD_DUMP_DIR=$RUN_DIR/reward_dump
CKPT_DIR=$RUN_DIR/checkpoints
mkdir -p "$RUN_DIR" "$REWARD_DUMP_DIR" "$CKPT_DIR" "$REPO/logs"
# Fresh endpoint handshake each run.
rm -f "$ENDPOINT_FILE"

echo ">> RL-generator run: JUDGE=$JUDGE MODEL=$JUDGE_MODEL MODE=$MODE RUN_TAG=$RUN_TAG"
echo ">> run dir: $RUN_DIR"

# 1) Submit the judge serve job (Task 7 script; parametrized by MODEL/TP/DP/JUDGE_ENDPOINT_FILE).
JUDGE_JOB=$(MODEL=$JUDGE_MODEL TP=$TP DP=$DP JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  sbatch --parsable --export=ALL scripts/slurm/judge_serve_9b_replicas.sh)
echo ">> judge serve job: $JUDGE_JOB (waiting for endpoint file $ENDPOINT_FILE)"

# 2) Wait for the serve script to publish the endpoint (it writes only after a model-verified health check).
for t in $(seq 1 1800); do   # up to 60 min warmup (397B load is slow)
  [ -s "$ENDPOINT_FILE" ] && break
  # abort if the judge job left the queue without publishing
  squeue -j "$JUDGE_JOB" -h >/dev/null 2>&1 || { echo "judge job $JUDGE_JOB gone before endpoint" >&2; }
  if ! squeue -j "$JUDGE_JOB" -h 2>/dev/null | grep -q .; then
    echo "ERROR: judge job $JUDGE_JOB no longer queued/running and no endpoint published" >&2
    exit 3
  fi
  sleep 2
done
[ -s "$ENDPOINT_FILE" ] || { echo "TIMEOUT waiting for judge endpoint; scancel $JUDGE_JOB" >&2; scancel "$JUDGE_JOB"; exit 4; }
ENDPOINT=$(cat "$ENDPOINT_FILE")
echo ">> judge endpoint: $ENDPOINT"

# 3) Reward env for the trainer (inherited via --export=ALL).
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
# Trainer-role vars:
export RL_MODE="$MODE" RL_JUDGE="$JUDGE" RL_RUN_TAG="$RUN_TAG"
export RL_RUN_DIR="$RUN_DIR" RL_CKPT_DIR="$CKPT_DIR" RL_JUDGE_JOB_ID="$JUDGE_JOB"

# 4) Submit the trainer (owns judge teardown via its own EXIT trap on RL_JUDGE_JOB_ID).
TRAIN_JOB=$(sbatch --parsable --export=ALL scripts/slurm/rl_generator_train.sh)
echo ">> trainer job: $TRAIN_JOB  (judge=$JUDGE_JOB will be scancel'd by the trainer at exit)"
echo ">> monitor: squeue -j $JUDGE_JOB,$TRAIN_JOB ; logs in $REPO/logs/"
