#!/bin/bash
#SBATCH --job-name=judge_serve_8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_serve_8b-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Long-running Qwen3-8B judge server for the First Experiment of the adversarial
# user-simulator proposal: does a *frozen small judge* get reward-hacked by GRPO?
# This is a scaled-down analogue of judge_serve.sh (which serves the 397B judge
# for the paper reproduction). Uses 1 GPU with TP=1; 8B fits comfortably.
#
# --reasoning-parser qwen3: the correct boundary detector for Qwen3's chat
# template (handles <think> in prompt + <tool_call> as implicit reasoning end).
# We used deepseek_r1 initially; source-verified that qwen3 is the right one.

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python
MODEL=Qwen/Qwen3-8B
# Port 8000 clashes with other users' vLLMs on shared A100 nodes (learned the
# hard way: job 9230 died on OSError EADDRINUSE). 8123 keeps us out of their way.
PORT=8123

echo "============================================"
echo "Judge server (8B): $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "IP:   $(hostname -I)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --reasoning-parser qwen3 \
  --host 0.0.0.0 --port $PORT
