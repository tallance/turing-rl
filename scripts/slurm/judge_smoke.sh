#!/bin/bash
#SBATCH --job-name=judge_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
REPO=${TURING_RL_WORK_ROOT:?}
MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
PORT=8000
VLLM_LOG="$REPO/logs/vllm-${SLURM_JOB_ID}.log"

echo "============================================"
echo "Judge smoke: $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Launch vLLM OpenAI server in background
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size 8 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --host 0.0.0.0 --port $PORT \
  > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vllm pid: $VLLM_PID, logging to logs/vllm-${SLURM_JOB_ID}.log"

cleanup() {
  echo "[cleanup] killing vllm pid $VLLM_PID"
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  sleep 5
  kill -KILL "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Wait for server to come up (poll /v1/models, up to 20 min)
echo "Waiting for vLLM server to be ready..."
ready=0
for i in $(seq 1 240); do
  if curl -sf --max-time 5 "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
    echo "vLLM ready after ${i}*5s"
    ready=1
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM process died before becoming ready"
    echo "--- tail of vllm log ---"
    tail -100 "$VLLM_LOG"
    exit 2
  fi
  sleep 5
done
if [ "$ready" -ne 1 ]; then
  echo "vLLM did not become ready within 20 minutes"
  echo "--- tail of vllm log ---"
  tail -200 "$VLLM_LOG"
  exit 3
fi

echo "============================================"
echo "Running smoke test"
echo "============================================"
$PY $TURING_RL_CODE_ROOT/scripts/smoke_judge.py \
  --base-url "http://localhost:$PORT/v1" \
  --model "$MODEL" \
  --batch-size 64

echo "============================================"
echo "Smoke test complete at $(date)"
echo "============================================"
