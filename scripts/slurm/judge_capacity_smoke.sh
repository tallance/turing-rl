#!/bin/bash
#SBATCH --job-name=judge_cap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Capacity probe: how much KV/state cache does a judge model get, and how many sequences
# fit, at each candidate context length?
#
# Judge GRPO colocates vLLM with FSDP training on the same GPU, so vLLM only gets
# `gpu_memory_utilization` of the card. Qwen3.5 is hybrid Gated-DeltaNet: 3 of every 4
# layers are recurrent with a FIXED-size state per sequence, and only the 4th caches KV per
# token. That means concurrency does NOT fall linearly with max_model_len the way it would
# on a pure-attention model, and the only reliable way to know the real number is to ask
# vLLM. It reports both lines at engine init:
#     GPU KV cache size: N tokens
#     Maximum concurrency for <len> tokens per request: Yx
#
# Initializes the engine and exits; no generation. Reads nothing from the repo beyond this
# script, so it is safe to run before the training config is finalized.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

CAP_MODEL=${CAP_MODEL:-Qwen/Qwen3.5-9B}
# The training config's split: vLLM gets this fraction, FSDP training gets the rest.
CAP_GPU_UTIL=${CAP_GPU_UTIL:-0.55}
# Current config, the +8192-completion candidates, and the headroom candidate.
CAP_LENS=${CAP_LENS:-"16384 20480 22528"}
CAP_MAX_NUM_SEQS=${CAP_MAX_NUM_SEQS:-64}

echo "=== judge capacity smoke: model=$CAP_MODEL util=$CAP_GPU_UTIL lens='$CAP_LENS' ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for len in $CAP_LENS; do
  echo "########## max_model_len=$len ##########"
  $PY - "$CAP_MODEL" "$CAP_GPU_UTIL" "$len" "$CAP_MAX_NUM_SEQS" <<'PYEOF'
import sys, gc
model, util, length, max_seqs = sys.argv[1], float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
from vllm import LLM
try:
    llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=util,
              max_model_len=length, max_num_seqs=max_seqs, enforce_eager=True,
              trust_remote_code=True, disable_log_stats=True)
    print(f"RESULT len={length} ENGINE_OK", flush=True)
    del llm
except Exception as exc:  # a refusal here is the answer, not a crash
    print(f"RESULT len={length} ENGINE_FAILED {type(exc).__name__}: {str(exc)[:300]}", flush=True)
gc.collect()
PYEOF
done

echo "=== done; grep the log for 'GPU KV cache size', 'Maximum concurrency' and 'RESULT' ==="
