#!/bin/bash
#SBATCH --job-name=cot_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/cot_serve-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Self-hosted thinking-OFF CoT generation. Launches 8 single-GPU Qwen3-8B vLLM
# replicas (TP=1) on ports 8000-8007 of this node, waits for all /health probes,
# then runs the async round-robin client scripts/generate_cot_served.py over all
# 8 endpoints. NO --reasoning-parser: thinking is disabled via the client's
# chat_template_kwargs={"enable_thinking": False}, so .content is the clean reply.

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=${TURING_RL_WORK_ROOT:?}
MODEL=Qwen/Qwen3-8B
N_REPLICAS=8
BASE_PORT=8000
HOSTNAME_FQDN=$(hostname)
ENDPOINTS_FILE="$REPO/logs/cot_serve_endpoints-${SLURM_JOB_ID:-local}.txt"
OUT_PARQUET="$TURING_RL_DATA_ROOT/sft/prism_full_s42_sft_cot.parquet"

mkdir -p "$REPO/logs" "$TURING_RL_DATA_ROOT/sft"

echo "============================================"
echo "CoT serve (thinking-off): $N_REPLICAS x $MODEL (TP=1)"
echo "Date: $(date)"
echo "Host: $HOSTNAME_FQDN"
echo "IP:   $(hostname -I)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Launch one replica per GPU, each pinned via CUDA_VISIBLE_DEVICES.
: > "$ENDPOINTS_FILE"
SERVER_PIDS=()
for i in $(seq 0 $((N_REPLICAS - 1))); do
  PORT=$((BASE_PORT + i))
  echo "Launching replica $i on GPU $i port $PORT"
  CUDA_VISIBLE_DEVICES=$i $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --download-dir /home/lancewicki/data/hf_cache \
    --tensor-parallel-size 1 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --host 0.0.0.0 --port "$PORT" \
    > "$REPO/logs/cot_serve-replica${i}-${SLURM_JOB_ID:-local}.out" 2>&1 &
  SERVER_PIDS+=($!)
  echo "http://${HOSTNAME_FQDN}:${PORT}/v1" >> "$ENDPOINTS_FILE"
done

cleanup() {
  echo "Shutting down $N_REPLICAS replicas..."
  for pid in "${SERVER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
}
trap cleanup EXIT

echo "Endpoints file: $ENDPOINTS_FILE"
cat "$ENDPOINTS_FILE"

# Poll each replica's /health until ready (up to ~20 min total).
echo "=== waiting for /health on all replicas ==="
for i in $(seq 0 $((N_REPLICAS - 1))); do
  PORT=$((BASE_PORT + i))
  HEALTH_URL="http://${HOSTNAME_FQDN}:${PORT}/health"
  READY=0
  for attempt in $(seq 1 240); do
    if curl -sf --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; then
      echo "replica $i (port $PORT) healthy after ${attempt} probes"
      READY=1
      break
    fi
    sleep 5
  done
  if [ "$READY" -ne 1 ]; then
    echo "ERROR: replica $i (port $PORT) never became healthy." >&2
    exit 3
  fi
done
echo "All $N_REPLICAS replicas healthy."

cd "$REPO"

echo "=== generate_cot_served (async round-robin, thinking-off) ==="
$PY scripts/generate_cot_served.py \
  --endpoints "$ENDPOINTS_FILE" \
  --out "$OUT_PARQUET" \
  --model "$MODEL" \
  --max_completion_tokens 4096 \
  --max_regen_attempts 10 \
  --concurrency_per_endpoint 16
RC=$?
echo "generate_cot_served exit: $RC"

if [ $RC -eq 0 ]; then
  echo "=== summary ==="
  echo "CoT parquet: $OUT_PARQUET"
  META="${OUT_PARQUET}.cot_metadata.json"
  [ -f "$META" ] && cat "$META"
fi

echo "============================================"
echo "Done at $(date)"
echo "============================================"
exit $RC
