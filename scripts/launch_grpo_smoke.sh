#!/bin/bash
# Orchestrate the GRPO smoke:
#   1. sbatch judge_serve.sh (8-GPU, Qwen 397B)
#   2. wait for /v1/models on the assigned node
#   3. sbatch grpo_smoke.sh with JUDGE_HOST exported
#   4. wait for trainer to finish, then scancel judge
#
# Run on the login pod. Login pod's http_proxy 403s compute-node polls, so we
# unset proxies first (learned this the hard way in the CoT orchestrator).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "============================================"
echo "[launch] submitting judge server"
echo "============================================"
JUDGE_JOB=$("$SBATCH" --parsable "$REPO/scripts/slurm/judge_serve.sh")
[ -z "$JUDGE_JOB" ] && { echo "judge sbatch returned no job id" >&2; exit 2; }
echo "judge job id: $JUDGE_JOB"
JUDGE_LOG="$LOGS/judge_serve-$JUDGE_JOB.out"

cleanup() {
  echo "[launch] scancel judge $JUDGE_JOB"
  scancel "$JUDGE_JOB" 2>/dev/null || true
}

echo "[launch] waiting for judge job to reach RUNNING..."
JUDGE_NODE=""
# Partition may be congested — allow up to 4h in-queue for the judge.
for i in $(seq 1 2880); do   # 2880 * 5s = 4h
  STATE_AND_NODE=$(squeue -h -j "$JUDGE_JOB" -o '%T %N' 2>/dev/null || true)
  if [ -z "$STATE_AND_NODE" ]; then
    echo "[launch] judge $JUDGE_JOB vanished from queue"
    [ -f "$JUDGE_LOG" ] && tail -50 "$JUDGE_LOG"
    exit 3
  fi
  STATE=${STATE_AND_NODE%% *}
  NODE=${STATE_AND_NODE##* }
  if [ "$STATE" = "RUNNING" ] && [ -n "$NODE" ] && [ "$NODE" != "(None)" ]; then
    JUDGE_NODE=$NODE
    echo "[launch] judge running on $JUDGE_NODE (after ${i}*5s)"
    break
  fi
  # Print a status line every ~5 minutes so the log doesn't look dead.
  if [ $((i % 60)) -eq 0 ]; then
    echo "[launch] still queued: state=$STATE reason=$(squeue -h -j "$JUDGE_JOB" -o '%R' 2>/dev/null) elapsed=$((i*5))s"
  fi
  sleep 5
done
[ -z "$JUDGE_NODE" ] && { echo "[launch] judge did not start within 4h" >&2; cleanup; exit 4; }

trap cleanup EXIT

echo "[launch] polling http://$JUDGE_NODE:8000/v1/models (up to 30 min for 397B warmup)..."
READY=0
for i in $(seq 1 360); do
  if curl -sf --max-time 5 "http://$JUDGE_NODE:8000/v1/models" >/dev/null 2>&1; then
    echo "[launch] /v1/models OK after ${i}*5s"
    READY=1
    break
  fi
  if ! squeue -h -j "$JUDGE_JOB" -o '%T' 2>/dev/null | grep -q .; then
    echo "[launch] judge job died while warming up"
    [ -f "$JUDGE_LOG" ] && tail -120 "$JUDGE_LOG"
    exit 5
  fi
  sleep 5
done
[ "$READY" -ne 1 ] && { echo "[launch] /v1/models never responded" >&2; [ -f "$JUDGE_LOG" ] && tail -200 "$JUDGE_LOG"; exit 6; }

echo "============================================"
echo "[launch] submitting GRPO trainer (JUDGE_HOST=$JUDGE_NODE)"
echo "============================================"
TRAINER_JOB=$("$SBATCH" --parsable \
  --export=ALL,JUDGE_HOST="$JUDGE_NODE",JUDGE_PORT=8000 \
  "$REPO/scripts/slurm/grpo_smoke.sh")
[ -z "$TRAINER_JOB" ] && { echo "trainer sbatch returned no job id" >&2; exit 7; }
echo "trainer job id: $TRAINER_JOB"
TRAINER_LOG="$LOGS/grpo_smoke-$TRAINER_JOB.out"

echo "[launch] waiting for trainer to complete..."
while squeue -h -j "$TRAINER_JOB" -o '%T' 2>/dev/null | grep -q .; do
  sleep 30
done
TRAINER_STATE=$(sacct -j "$TRAINER_JOB" --format=State -n -X 2>/dev/null | head -1 | awk '{print $1}')
echo "[launch] trainer state: $TRAINER_STATE"

echo "============================================"
echo "[launch] trainer log tail:"
echo "============================================"
[ -f "$TRAINER_LOG" ] && tail -120 "$TRAINER_LOG"

echo "============================================"
echo "[launch] done. judge will be cancelled by trap."
echo "============================================"
