#!/bin/bash
#SBATCH --job-name=cot_server
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/cot_server-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Long-running Qwen3-8B vLLM server used as the teacher for data.sft.generate_cot.
# Serves on port 8000 of the assigned compute node. Caller (launch_cot_smoke.sh)
# reads the node name via squeue, polls /v1/models, then runs the client job.

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO

# vllm 0.18 lives in turing-rl-train env (cu130), not judge-vllm.
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
MODEL=Qwen/Qwen3-8B
PORT=8000

echo "============================================"
echo "CoT teacher server: $MODEL"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "IP:   $(hostname -I)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Per the Qwen3-8B model card, vLLM needs --reasoning-parser deepseek_r1
# to split <think>...</think> into a separate `reasoning` field. The .content
# field then matches what OpenRouter returns to generate_cot.py.
# (vLLM 0.18 dropped the older --enable-reasoning flag; the parser flag alone enables it.)
# Sampling: vllm picks up generation_config.json from the model dir (T=0.6, top_p=0.95,
# top_k=20, min_p=0 for thinking mode) — no override flag needed.
$PY -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --download-dir /home/lancewicki/data/hf_cache \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  --reasoning-parser deepseek_r1 \
  --host 0.0.0.0 --port $PORT
