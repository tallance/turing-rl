#!/bin/bash
# Orchestrate the GRPO smoke against the 8B frozen judge:
#   1. sbatch judge_serve_8b.sh (1-GPU, Qwen3-8B)
#   2. wait for /v1/models on the assigned node
#   3. sbatch grpo_smoke_8b.sh with JUDGE_HOST exported
#   4. wait for trainer to finish, then scancel judge
#
# Parallel to launch_grpo_smoke.sh (the 397B orchestrator). This script and the
# _8b sbatches it drives are completely independent — the 397B run keeps going.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "============================================"
echo "[launch-8b] submitting 8B judge server"
echo "============================================"
JUDGE_JOB=$("$SBATCH" --parsable -- "$REPO/scripts/slurm/judge_serve_8b.sh")
[ -z "$JUDGE_JOB" ] && { echo "judge sbatch returned no job id" >&2; exit 2; }
echo "judge job id: $JUDGE_JOB"
JUDGE_LOG="$LOGS/judge_serve_8b-$JUDGE_JOB.out"

cleanup() {
  echo "[launch-8b] scancel judge $JUDGE_JOB"
  scancel "$JUDGE_JOB" 2>/dev/null || true
}

echo "[launch-8b] waiting for judge job to reach RUNNING..."
JUDGE_NODE=""
for i in $(seq 1 2880); do   # 2880 * 5s = 4h
  STATE_AND_NODE=$(squeue -h -j "$JUDGE_JOB" -o '%T %N' 2>/dev/null || true)
  if [ -z "$STATE_AND_NODE" ]; then
    echo "[launch-8b] judge $JUDGE_JOB vanished from queue"
    [ -f "$JUDGE_LOG" ] && tail -50 "$JUDGE_LOG"
    exit 3
  fi
  STATE=${STATE_AND_NODE%% *}
  NODE=${STATE_AND_NODE##* }
  if [ "$STATE" = "RUNNING" ] && [ -n "$NODE" ] && [ "$NODE" != "(None)" ]; then
    JUDGE_NODE=$NODE
    echo "[launch-8b] judge running on $JUDGE_NODE (after ${i}*5s)"
    break
  fi
  if [ $((i % 60)) -eq 0 ]; then
    echo "[launch-8b] still queued: state=$STATE reason=$(squeue -h -j "$JUDGE_JOB" -o '%R' 2>/dev/null) elapsed=$((i*5))s"
  fi
  sleep 5
done
[ -z "$JUDGE_NODE" ] && { echo "[launch-8b] judge did not start within 4h" >&2; cleanup; exit 4; }

trap cleanup EXIT

# 8B on 1 GPU warms in ~1-2 min vs the 397B's ~15-25 min. Allow up to 10 min.
echo "[launch-8b] polling http://$JUDGE_NODE:8123/v1/models (up to 10 min for 8B warmup)..."
READY=0
for i in $(seq 1 120); do
  if curl -sf --max-time 5 "http://$JUDGE_NODE:8123/v1/models" >/dev/null 2>&1; then
    echo "[launch-8b] /v1/models OK after ${i}*5s"
    READY=1
    break
  fi
  if ! squeue -h -j "$JUDGE_JOB" -o '%T' 2>/dev/null | grep -q .; then
    echo "[launch-8b] judge job died while warming up"
    [ -f "$JUDGE_LOG" ] && tail -120 "$JUDGE_LOG"
    exit 5
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "[launch-8b] /v1/models never responded" >&2; [ -f "$JUDGE_LOG" ] && tail -200 "$JUDGE_LOG"; exit 6; }

echo "============================================"
echo "[launch-8b] submitting GRPO trainer (JUDGE_HOST=$JUDGE_NODE)"
echo "============================================"
TRAINER_JOB=$("$SBATCH" --parsable \
  --export=ALL,JUDGE_HOST="$JUDGE_NODE",JUDGE_PORT=8123 \
  -- \
  "$REPO/scripts/slurm/grpo_smoke_8b.sh")
[ -z "$TRAINER_JOB" ] && { echo "trainer sbatch returned no job id" >&2; exit 7; }
echo "trainer job id: $TRAINER_JOB"
TRAINER_LOG="$LOGS/grpo_smoke_8b-$TRAINER_JOB.out"

echo "[launch-8b] waiting for trainer to complete..."
while squeue -h -j "$TRAINER_JOB" -o '%T' 2>/dev/null | grep -q .; do
  sleep 30
done
TRAINER_STATE=$(sacct -j "$TRAINER_JOB" --format=State -n -X 2>/dev/null | head -1 | awk '{print $1}')
echo "[launch-8b] trainer state: $TRAINER_STATE"

echo "============================================"
echo "[launch-8b] trainer log tail:"
echo "============================================"
[ -f "$TRAINER_LOG" ] && tail -120 "$TRAINER_LOG"

echo "============================================"
echo "[launch-8b] done. judge will be cancelled by trap."
echo "============================================"
