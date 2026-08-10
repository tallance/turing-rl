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
# up until walltime/scancel. thinking-on via --reasoning-parser qwen3.
#
# Env: MODEL (default Qwen/Qwen3.5-9B), TP (default 1), DP (default 8),
#   PORT (default derived from job id), REASONING_PARSER (default qwen3),
#   JUDGE_ENDPOINT_FILE (default logs/judge_endpoint-<jobid>.txt),
#   JUDGE_ENTRYPOINT (module|serve, default module -- see the case block below).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/home/lancewicki/projects/turing-rl
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
TP=${TP:-1}
DP=${DP:-8}
REASONING_PARSER=${REASONING_PARSER:-qwen3}
PORT=${PORT:-$((8300 + ${SLURM_JOB_ID:-0} % 400))}
JUDGE_ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:-$REPO/logs/judge_endpoint-${SLURM_JOB_ID}.txt}

# 397B GPTQ anchor serves from judge-vllm; 9B/dense from turing-rl-train (newer vLLM).
case "$MODEL" in
  *397B*) PY=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
  *)      PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
esac
# custom all-reduce kernel fails on A100 (cap 8.0) at TP>1 -> NCCL fallback.
AR=(); [ "$TP" -gt 1 ] && AR=(--disable-custom-all-reduce)
DPFLAG=(); [ "$DP" -gt 1 ] && DPFLAG=(--data-parallel-size "$DP")

# JUDGE_ENTRYPOINT selects the HTTP frontend. Same vLLM install, same weights, same flags --
# only the launcher differs, so judge outputs are unchanged.
#   module (default) : python -m vllm.entrypoints.openai.api_server. What job 14217 ran.
#   serve            : the `vllm serve` CLI. Replay 14322 measured 0.455 req/s on this
#                      frontend against 0.115-0.148 req/s for the module path under the same
#                      DP-8 topology, because the module path progressively collapses all
#                      traffic onto engine 000 under sustained load.
# Do NOT add --api-server-count: jobs 14359 vs 14361 measured 0.576 vs 0.585 req/s, i.e. the
# flag does nothing. The gain is the entrypoint, not the flag.
JUDGE_ENTRYPOINT=${JUDGE_ENTRYPOINT:-module}
case "$JUDGE_ENTRYPOINT" in
  module) SERVE_CMD=("$PY" -m vllm.entrypoints.openai.api_server --model "$MODEL") ;;
  serve)
    VLLM_BIN="$(dirname "$PY")/vllm"
    [ -x "$VLLM_BIN" ] || { echo "ERROR: JUDGE_ENTRYPOINT=serve but no vLLM CLI at $VLLM_BIN" >&2; exit 5; }
    SERVE_CMD=("$VLLM_BIN" serve "$MODEL") ;;
  *) echo "ERROR: bad JUDGE_ENTRYPOINT=$JUDGE_ENTRYPOINT (want module|serve)" >&2; exit 5 ;;
esac

mkdir -p "$REPO/logs"
echo "============================================"
echo "9B DP judge server: MODEL=$MODEL TP=$TP DP=$DP port=$PORT"
echo "=== judge entrypoint: $JUDGE_ENTRYPOINT ==="
echo "date=$(date) host=$(hostname) endpoint_file=$JUDGE_ENDPOINT_FILE"
"$PY" -c 'import vllm; print("judge vllm version:", vllm.__version__)'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

"${SERVE_CMD[@]}" \
  --download-dir "$HF_HOME" \
  --tensor-parallel-size "$TP" "${DPFLAG[@]}" \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --reasoning-parser "$REASONING_PARSER" "${AR[@]}" \
  --host 0.0.0.0 --port "$PORT" &
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
