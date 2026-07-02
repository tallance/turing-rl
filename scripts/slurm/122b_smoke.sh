#!/bin/bash
#SBATCH --job-name=122b_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/122b_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
MODEL="${MODEL:?must set MODEL env var}"
TP="${TP:-4}"            # tensor-parallel size
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
EXTRA_ARGS=()
[ -n "$MAX_NUM_SEQS" ] && EXTRA_ARGS+=(--max-num-seqs "$MAX_NUM_SEQS")

echo "============================================"
echo "Smoke: $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "TP=$TP  MAX_MODEL_LEN=$MAX_MODEL_LEN  GPU_MEM_UTIL=$GPU_MEM_UTIL  MAX_NUM_SEQS=${MAX_NUM_SEQS:-default}"
echo "GPUs:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --dtype bfloat16 \
  "${EXTRA_ARGS[@]}" \
  --host 0.0.0.0 --port "$PORT" \
  > /home/lancewicki/projects/turing-rl/logs/vllm-${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
echo "vllm pid: $VLLM_PID, log: logs/vllm-${SLURM_JOB_ID}.log"

cleanup() {
  echo "[cleanup] killing vllm pid $VLLM_PID"
  kill -TERM "$VLLM_PID" 2>/dev/null || true
  sleep 5
  kill -KILL "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server..."
ready=0
for i in $(seq 1 240); do
  if curl -sf --max-time 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "vLLM ready after ${i}*5s"; ready=1; break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "vLLM died before ready"
    tail -100 /home/lancewicki/projects/turing-rl/logs/vllm-${SLURM_JOB_ID}.log
    exit 2
  fi
  sleep 5
done
[ "$ready" -ne 1 ] && { echo "vLLM not ready in 20 min"; tail -200 /home/lancewicki/projects/turing-rl/logs/vllm-${SLURM_JOB_ID}.log; exit 3; }

echo "============================================"
echo "Running smoke test"
echo "============================================"
$PY /home/lancewicki/projects/turing-rl/scripts/smoke_judge.py \
  --base-url "http://localhost:$PORT/v1" \
  --model "$MODEL" \
  --batch-size 64

echo "============================================"
echo "Done at $(date)"
echo "============================================"
