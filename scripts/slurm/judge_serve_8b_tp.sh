#!/bin/bash
#SBATCH --job-name=judge_serve_8b_tp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_serve_8b_tp%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Parameterized Qwen3-8B judge server for the TP throughput sweep. Set:
#   TP:   tensor-parallel size (1, 2, 4, 8). Must match --gres=gpu:$TP on sbatch.
#   PORT: HTTP port to serve on. Default 8123. If multiple judges may co-schedule
#         on the same node (small-TP judges often do), give each a unique port.
# The orchestrator (launch_judge_throughput_sweep.sh) sets both.

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -z "${TP:-}" ]; then
  echo "ERROR: TP env var must be set (1|2|4|8). Submit via launch_judge_throughput_sweep.sh." >&2
  exit 2
fi

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
MODEL=Qwen/Qwen3-8B
PORT="${PORT:-8123}"

echo "============================================"
echo "Judge server (8B TP=$TP PORT=$PORT): $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "IP:   $(hostname -I)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size "$TP" \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port $PORT
