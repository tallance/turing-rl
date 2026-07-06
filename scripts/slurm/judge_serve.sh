#!/bin/bash
#SBATCH --job-name=judge_serve
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_serve-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Long-running Qwen3.5-397B-A17B-GPTQ-Int4 judge server. Serves on port 8000 of
# the assigned compute node. Unlike judge_smoke.sh, this does NOT run a test
# client at the end — it stays up until the walltime expires or the caller
# cancels the job. Used by launch_grpo_smoke.sh.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4
PORT=8000

echo "============================================"
echo "Judge server: $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "IP:   $(hostname -I)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Args validated during our earlier judge_smoke run: dtype bfloat16, drop
# --enforce-eager, let vLLM auto-detect quantization. TP=8 across all 8 A100s.
# --reasoning-parser deepseek_r1: splits <think>...</think> out of .content into
# a separate .reasoning field, matching what OpenRouter returns for the paper's
# default judge (`reasoning: {enabled: true}` in shared/api_client.py:66-68).
# Without this flag Qwen3.5-397B thinking is effectively off for JSON-mode calls.
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --reasoning-parser deepseek_r1 \
  --host 0.0.0.0 --port $PORT
