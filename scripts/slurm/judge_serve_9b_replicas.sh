#!/bin/bash
#SBATCH --job-name=judge_serve_9b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=2-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_serve_9b-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Data-parallel judge server for the RL-generator runs. Serves ONE vLLM OpenAI
# api_server with --data-parallel-size DP (TP per replica), so the single-base-URL
# GRPO reward path (reward.py -> OPENAI_API_BASE) is load-balanced across all GPUs.
# Writes the endpoint URL (node IP, cross-node) to JUDGE_ENDPOINT_FILE, then stays
# up until walltime/scancel. thinking-on via --reasoning-parser (family-specific); an EMPTY
# REASONING_PARSER omits the flag, which is what the single-token judge protocol serves with.
#
# Qwen models serve via `python -m vllm.entrypoints.openai.api_server`; Gemma 4 serves via
# the nightly env's `vllm serve` against a pinned offline snapshot. Either way exactly one
# endpoint is published, so the reward path's single OPENAI_API_BASE contract is unchanged.
#
# Env: MODEL (default Qwen/Qwen3.5-9B), TP (default 1), DP (default 8),
#   PORT (default derived from job id), REASONING_PARSER (default qwen3),
#   GEMMA_GPU_MEMORY_UTILIZATION (default 0.90, gemma only),
#   JUDGE_ENDPOINT_FILE (default logs/judge_endpoint-<jobid>.txt).
set -uo pipefail
# Prepare a runtime view only when we are the top-level job script.
#
# turing_rl_prepare_runtime derives its runtime id from SLURM_JOB_ID alone and hard-fails
# if the work directory already exists ("FATAL: runtime work directory already exists").
# This script is normally NOT the top-level script: rl_generator_run_9b.sh prepares the
# runtime and then sruns this file, so sourcing the bootstrap again inside the same job
# collides with the parent's own work root and kills the judge step ~12 s in (jobs 18499,
# 18500). The parent exports TURING_RL_WORK_ROOT, so its presence is the signal that a
# runtime view already exists and should be reused rather than recreated.
#
# Running this script standalone as its own sbatch job still works: nothing has exported
# TURING_RL_WORK_ROOT, so it prepares its own view exactly as before.
if [ -z "${TURING_RL_WORK_ROOT:-}" ]; then
  source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=${TURING_RL_WORK_ROOT:?}
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TP=${TP:-1}
DP=${DP:-8}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
# Empty REASONING_PARSER => omit the flag entirely (thinking-off serving). The single-token
# judge decodes ONE token with enable_thinking=False, so there is no <think> block to parse;
# judge_sweep_cell.sh likewise only adds a parser for thinking-on cells, and that is the
# configuration the seven-cell matrix actually ran against.
RP=()
[ -n "$REASONING_PARSER" ] && RP=(--reasoning-parser "$REASONING_PARSER")
PORT=${PORT:-$((8300 + ${SLURM_JOB_ID:-0} % 400))}
JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:-$REPO/logs/judge_endpoint-${SLURM_JOB_ID}.txt}

# 397B GPTQ anchor serves from judge-vllm; Gemma 4 needs the CUDA-13 nightly env and an
# exact offline snapshot; 9B/dense from turing-rl-train (newer vLLM).
#
# The Gemma constants below are NOT independently chosen -- they mirror the serving path
# already proven on the eval side (scripts/slurm/judge_sweep_cell.sh), which produced
# 440/440 parsed pairs per checkpoint. tests/test_gemma4_judge_runtime.py locks that
# script's copy; the launcher test locks this one, so the two cannot drift apart.
IS_GEMMA4=0
case "$MODEL" in
  *397B*) PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
  google/gemma-4-12B-it)
    IS_GEMMA4=1; GEMMA_SNAPSHOT=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 ;;
  google/gemma-4-31B-it)
    IS_GEMMA4=1; GEMMA_SNAPSHOT=842da3794eaa0b77d5f08bae87a17459d91ff475 ;;
  *)      PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
esac
# custom all-reduce kernel fails on A100 (cap 8.0) at TP>1 -> NCCL fallback.
AR=(); [ "$TP" -gt 1 ] && AR=(--disable-custom-all-reduce)
DPFLAG=(); [ "$DP" -gt 1 ] && DPFLAG=(--data-parallel-size "$DP")

GPU_MEMORY_UTILIZATION=0.85
GEMMA_ARGS=()
if [ "$IS_GEMMA4" = "1" ]; then
  GEMMA_VLLM=/home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly/bin/vllm
  GEMMA_MODEL_PATH=$HF_HOME/hub/models--google--${MODEL#google/}/snapshots/$GEMMA_SNAPSHOT
  [ -x "$GEMMA_VLLM" ] || { echo "ERROR: missing Gemma vLLM: $GEMMA_VLLM" >&2; exit 2; }
  [ -f "$GEMMA_MODEL_PATH/config.json" ] || {
    echo "ERROR: incomplete Gemma snapshot: $GEMMA_MODEL_PATH" >&2; exit 2; }
  # Gemma 4 is multimodal; the judge is text-only, so refuse image/video/audio slots rather
  # than reserve KV for them. 0.90 (vs 0.85) is what the eval cells run at.
  GPU_MEMORY_UTILIZATION=${GEMMA_GPU_MEMORY_UTILIZATION:-0.90}
  GEMMA_ARGS=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  # /tmp is a 1GB tmpfs on this cluster; FlashInfer's JIT workspace does not fit there.
  export TMPDIR=${TMPDIR:-/home/lancewicki/tmp/build}
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/home/lancewicki/tmp/flashinfer}
  mkdir -p "$TMPDIR" "$FLASHINFER_WORKSPACE_BASE"
fi

mkdir -p "$REPO/logs"
echo "============================================"
echo "DP judge server: MODEL=$MODEL TP=$TP DP=$DP port=$PORT parser=${REASONING_PARSER:-<none>} gpu_util=$GPU_MEMORY_UTILIZATION"
[ "$IS_GEMMA4" = "1" ] && echo "gemma_snapshot=$GEMMA_SNAPSHOT path=$GEMMA_MODEL_PATH"
echo "date=$(date) host=$(hostname) endpoint_file=$JUDGE_ENDPOINT_FILE"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

if [ "$IS_GEMMA4" = "1" ]; then
  # Serve the snapshot BY PATH (offline, exact revision) but advertise the canonical model
  # id, so the /v1/models health gate below and the reward path's model name still match.
  "$GEMMA_VLLM" serve "$GEMMA_MODEL_PATH" \
    --served-model-name "$MODEL" \
    --download-dir "$HF_HOME" \
    --tensor-parallel-size "$TP" "${DPFLAG[@]}" \
    --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --dtype bfloat16 \
    "${RP[@]}" "${AR[@]}" "${GEMMA_ARGS[@]}" \
    --host 0.0.0.0 --port "$PORT" &
else
  $PY -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --download-dir "$HF_HOME" \
    --tensor-parallel-size "$TP" "${DPFLAG[@]}" \
    --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --dtype bfloat16 \
    "${RP[@]}" "${AR[@]}" \
    --host 0.0.0.0 --port "$PORT" &
fi
SRV=$!
cleanup() { kill $SRV 2>/dev/null || true; }
trap cleanup EXIT

# Wait for /v1/models serving OUR model (guards against false-positive readiness).
echo "waiting for judge /v1/models (up to 30 min warmup)..."
ok=0
for t in $(seq 1 900); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; then
    ok=1; echo "judge ready after $((t*2))s"; break
  fi
  kill -0 $SRV 2>/dev/null || { echo "server died during warmup" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge" >&2; exit 4; }

# Cross-node endpoint: trainer runs on a different node, so publish node IP (not localhost).
NODE_IP=$(hostname -I | awk '{print $1}')
echo "http://$NODE_IP:$PORT/v1" > "$JUDGE_ENDPOINT_FILE"
echo "endpoint published: $(cat "$JUDGE_ENDPOINT_FILE")"

# Stay up until walltime/scancel.
wait $SRV
