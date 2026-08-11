#!/bin/bash
#SBATCH --job-name=dl_judge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/download_judge-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

# Clear stale V2 proxy vars (V3 TTLS handles routing transparently)
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache

MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
HF_CLI=/home/lancewicki/miniconda3/envs/verl-upstream/bin/huggingface-cli

echo "============================================"
echo "Download judge: $MODEL"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Dest:  $HF_HOME"
echo "============================================"
df -h $HF_HOME | head -2

attempt=1
max_attempts=20
until $HF_CLI download "$MODEL" --max-workers 8; do
  rc=$?
  echo "[attempt $attempt] huggingface-cli download failed with rc=$rc"
  if [ "$attempt" -ge "$max_attempts" ]; then
    echo "Max attempts ($max_attempts) reached, giving up."
    exit 1
  fi
  attempt=$((attempt+1))
  sleep 15
done

echo "============================================"
echo "Download complete at $(date)"
du -sh $HF_HOME/models--Qwen--Qwen3.5-397B-A17B-GPTQ-Int4 2>/dev/null
echo "============================================"
