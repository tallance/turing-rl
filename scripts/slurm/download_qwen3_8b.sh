#!/bin/bash
#SBATCH --job-name=dl_qwen3_8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/download_qwen3_8b-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache

MODEL="${MODEL:-Qwen/Qwen3-8B}"
HF=/home/lancewicki/miniconda3/envs/judge-vllm/bin/hf

echo "============================================"
echo "Download: $MODEL"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Dest:  $HF_HOME"
echo "============================================"

attempt=1
max_attempts=20
until $HF download "$MODEL" --max-workers 8; do
  rc=$?
  echo "[attempt $attempt] hf download failed with rc=$rc"
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "Max attempts reached, giving up."; exit 1
  fi
  attempt=$((attempt+1))
  sleep 15
done

echo "============================================"
echo "Download complete at $(date)"
slug="${MODEL/\//--}"
du -sh "$HF_HOME/models--$slug" 2>/dev/null
echo "============================================"
