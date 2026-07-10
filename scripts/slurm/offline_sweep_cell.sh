#!/bin/bash
#SBATCH --job-name=offline_sweep_cell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/offline_sweep_cell%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Offline batched bonus cell: in-process vLLM LLM.generate (TP=8) over the 880-pair
# held-out set (1760 prompts, both orderings), Qwen3-8B thinking-off. Bonus throughput
# comparison against the served cells; raw judge output only (no scoring).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
REPO=/home/lancewicki/projects/turing-rl

echo "============================================"
echo "Offline sweep cell (Qwen3-8B thinking-off, TP=8)"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

cd "$REPO"
$PY scripts/run_offline_sweep_cell.py \
  --pairs results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet \
  --out_dir results/2026-07-08-judge-sweep/raw/sweep/qwen3-8b/off_offline \
  --tensor_parallel_size 8
