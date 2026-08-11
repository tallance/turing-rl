#!/bin/bash
# Orchestrate the CoT pipeline:
#   1. sbatch cot_server.sh (1 GPU, Qwen3-8B vLLM)
#   2. wait until /v1/models responds on the assigned node
#   3. sbatch cot_generate_smoke.sh with COT_HOST exported
#   4. wait for the client job to finish, then scancel the server
#
# Run on the login pod. Logs go to ~/projects/turing-rl/logs/.

set -uo pipefail

# The login pod has a corporate http_proxy that 403s requests to compute-node IPs.
# Unset so curl talks directly to the SLURM-assigned node.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh
LOGS=$REPO/logs
mkdir -p "$LOGS"

echo "============================================"
echo "[launch] submitting CoT server"
echo "============================================"
SERVER_JOB=$("$SBATCH" --parsable "$REPO/scripts/slurm/cot_server.sh")
if [ -z "$SERVER_JOB" ]; then
  echo "sbatch did not return a job id" >&2
  exit 2
fi
echo "server job id: $SERVER_JOB"
SERVER_LOG="$LOGS/cot_server-$SERVER_JOB.out"

cleanup() {
  echo "[launch] scancel $SERVER_JOB"
  scancel "$SERVER_JOB" 2>/dev/null || true
}

echo "[launch] waiting for server job to enter RUNNING state..."
SERVER_NODE=""
for i in $(seq 1 240); do  # up to 20 min
  STATE_AND_NODE=$(squeue -h -j "$SERVER_JOB" -o '%T %N' 2>/dev/null || true)
  if [ -z "$STATE_AND_NODE" ]; then
    echo "[launch] job $SERVER_JOB no longer in queue (failed/cancelled?)"
    [ -f "$SERVER_LOG" ] && tail -50 "$SERVER_LOG"
    exit 3
  fi
  STATE=${STATE_AND_NODE%% *}
  NODE=${STATE_AND_NODE##* }
  if [ "$STATE" = "RUNNING" ] && [ -n "$NODE" ] && [ "$NODE" != "(None)" ]; then
    SERVER_NODE=$NODE
    echo "[launch] server running on $SERVER_NODE after ${i}*5s"
    break
  fi
  sleep 5
done

if [ -z "$SERVER_NODE" ]; then
  echo "[launch] server did not start within 20 min" >&2
  cleanup
  exit 4
fi

trap cleanup EXIT

echo "[launch] polling http://$SERVER_NODE:8000/v1/models ..."
READY=0
for i in $(seq 1 360); do  # up to 30 min for first-time vllm warmup
  if curl -sf --max-time 5 "http://$SERVER_NODE:8000/v1/models" >/dev/null 2>&1; then
    echo "[launch] /v1/models OK after ${i}*5s"
    READY=1
    break
  fi
  # Bail if server job died meanwhile.
  if ! squeue -h -j "$SERVER_JOB" -o '%T' 2>/dev/null | grep -q .; then
    echo "[launch] server job vanished while waiting for /v1/models"
    [ -f "$SERVER_LOG" ] && tail -100 "$SERVER_LOG"
    exit 5
  fi
  sleep 5
done

if [ "$READY" -ne 1 ]; then
  echo "[launch] /v1/models never responded" >&2
  [ -f "$SERVER_LOG" ] && tail -200 "$SERVER_LOG"
  exit 6
fi

echo "============================================"
echo "[launch] submitting CoT client (COT_HOST=$SERVER_NODE)"
echo "============================================"
CLIENT_JOB=$("$SBATCH" --parsable \
  --export=ALL,COT_HOST="$SERVER_NODE",COT_PORT=8000 \
  "$REPO/scripts/slurm/cot_generate_smoke.sh")
if [ -z "$CLIENT_JOB" ]; then
  echo "client sbatch did not return a job id" >&2
  exit 7
fi
echo "client job id: $CLIENT_JOB"
CLIENT_LOG="$LOGS/cot_generate_smoke-$CLIENT_JOB.out"

echo "[launch] waiting for client job to complete..."
while squeue -h -j "$CLIENT_JOB" -o '%T' 2>/dev/null | grep -q .; do
  sleep 30
done

CLIENT_STATE=$(sacct -j "$CLIENT_JOB" --format=State -n -X 2>/dev/null | head -1 | awk '{print $1}')
echo "[launch] client state: $CLIENT_STATE"

echo "============================================"
echo "[launch] client log tail:"
echo "============================================"
[ -f "$CLIENT_LOG" ] && tail -80 "$CLIENT_LOG"

echo "============================================"
echo "[launch] done. server will be cancelled by trap."
echo "============================================"
