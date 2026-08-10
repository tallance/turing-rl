#!/bin/bash
#SBATCH --job-name=gemma31-schema-cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:30:00
#SBATCH --partition=a100
#SBATCH --account=rfai
#SBATCH --gres=gpu:8
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/gemma31-schema-cmp-%j.out

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${REPO:-/home/lancewicki/projects/turing-rl}
MODEL=google/gemma-4-31B-it
MODEL_CACHE=/home/lancewicki/data/hf_cache/hub/models--google--gemma-4-31B-it
VLLM=/home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly/bin/vllm
PYTHON=/home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly/bin/python
INPUT=${INPUT:-/home/lancewicki/tmp/probe31b/sweep/gemma4-31b/on/http/judge-15139-1562847.jsonl}
N_PROMPTS=${N_PROMPTS:-16}
CONCURRENCY=${CONCURRENCY:-4}
PORT=${PORT:-$((8400 + ${SLURM_JOB_ID:-0} % 600))}
OUT_ROOT=${OUT_ROOT:-$REPO/results/2026-08-10-gemma4-31b-full-schema-smoke/raw}
OUT=$OUT_ROOT/job-${SLURM_JOB_ID:-local}

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TMPDIR=/home/lancewicki/tmp/build
export FLASHINFER_WORKSPACE_BASE=/home/lancewicki/tmp/flashinfer
mkdir -p "$OUT" "$TMPDIR" "$FLASHINFER_WORKSPACE_BASE" "$REPO/logs"

[ -x "$VLLM" ] || { echo "ERROR: missing vLLM binary: $VLLM" >&2; exit 2; }
[ -x "$PYTHON" ] || { echo "ERROR: missing Python: $PYTHON" >&2; exit 2; }
[ -f "$INPUT" ] || { echo "ERROR: missing recorded prompt dump: $INPUT" >&2; exit 2; }
[ -f "$MODEL_CACHE/refs/main" ] || { echo "ERROR: missing model ref" >&2; exit 2; }
SNAPSHOT=$(cat "$MODEL_CACHE/refs/main")
MODEL_PATH=$MODEL_CACHE/snapshots/$SNAPSHOT
[ -f "$MODEL_PATH/config.json" ] || { echo "ERROR: incomplete model snapshot: $MODEL_PATH" >&2; exit 2; }

DEPLOYED_SHA=$(cat "$REPO/DEPLOYED_SHA")
export DEPLOYED_SHA
echo "date=$(date --iso-8601=seconds)"
echo "host=$(hostname) job=${SLURM_JOB_ID:-local} deployed_sha=$DEPLOYED_SHA"
echo "model=$MODEL snapshot=$SNAPSHOT input=$INPUT n=$N_PROMPTS concurrency=$CONCURRENCY"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$PYTHON" -c 'import vllm; print("vllm=" + vllm.__version__)'

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$VLLM" serve "$MODEL_PATH" \
  --served-model-name "$MODEL" \
  --download-dir "$HF_HOME" \
  --tensor-parallel-size 8 \
  --disable-custom-all-reduce \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --dtype bfloat16 \
  --reasoning-parser gemma4 \
  --limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' \
  --host 0.0.0.0 --port "$PORT" > "$OUT/vllm-server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

READY=0
for _ in $(seq 1 360); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" | grep -qF "\"$MODEL\""; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "ERROR: vLLM exited before becoming ready" >&2
    tail -n 200 "$OUT/vllm-server.log" >&2
    exit 3
  fi
  sleep 5
done
[ "$READY" -eq 1 ] || { echo "ERROR: timed out waiting for vLLM" >&2; exit 3; }
echo "server ready on port $PORT"

cd "$REPO"
"$PYTHON" scripts/experiments/compare_gemma31_schemas.py \
  --endpoint "http://localhost:$PORT/v1/chat/completions" \
  --input "$INPUT" \
  --out "$OUT" \
  --model "$MODEL" \
  --n "$N_PROMPTS" \
  --concurrency "$CONCURRENCY"

echo "comparison complete: $OUT"
