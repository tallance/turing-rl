#!/bin/bash
#SBATCH --job-name=judge_dp_replay
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_dp_replay-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Replays the first N validation judge prompts from job 14217 through the same
# vLLM 0.18 / Qwen3.5-9B DP=8 stack, changing only the launcher to the official
# multi-frontend CLI. The production control is the corresponding prefix in the
# existing reward dump and judge log.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=INFO
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
VLLM=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/vllm
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
N=${N:-512}
CONCURRENCY=${CONCURRENCY:-64}
TIMEOUT=${TIMEOUT:-1800}
PORT=${PORT:-$((8700 + ${SLURM_JOB_ID:-0} % 300))}
INPUT_DUMP=${INPUT_DUMP:-$REPO/results/grpo/rl-generator/9b_full5ep_kl1e4_lr1e4_temp1/reward_dump/reward-14217-1041480.jsonl}
OUT=${OUT:-$REPO/results/judge_dp_replay/${SLURM_JOB_ID}}

[ -x "$PY" ] || { echo "ERROR: missing Python: $PY" >&2; exit 2; }
[ -x "$VLLM" ] || { echo "ERROR: missing vLLM CLI: $VLLM" >&2; exit 2; }
[ -f "$INPUT_DUMP" ] || { echo "ERROR: missing input dump: $INPUT_DUMP" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT" "$REPO/logs"

SERVER_LOG=$OUT/server.log
GPU_LOG=$OUT/gpu_dmon.log
CLIENT_LOG=$OUT/client.log
SERVER_CMD=(
  "$VLLM" serve "$MODEL"
  --download-dir "$HF_HOME"
  --tensor-parallel-size 1
  --data-parallel-size 8
  --api-server-count 8
  --max-model-len 32768
  --gpu-memory-utilization 0.85
  --dtype bfloat16
  --reasoning-parser qwen3
  --host 0.0.0.0
  --port "$PORT"
)

echo "============================================"
echo "judge DP replay"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "model=$MODEL n=$N concurrency=$CONCURRENCY timeout=$TIMEOUT port=$PORT"
echo "input_dump=$INPUT_DUMP"
echo "out=$OUT"
echo "deployed_sha=$(cat "$REPO/DEPLOYED_SHA" 2>/dev/null || echo missing)"
"$PY" -c 'import aiohttp, sys, torch, vllm; print("python={} aiohttp={} torch={} vllm={}".format(sys.version.split()[0], aiohttp.__version__, torch.__version__, vllm.__version__))'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
printf 'server_command='
printf '%q ' "${SERVER_CMD[@]}"
printf '\n'
echo "============================================"

"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SRV=$!
MON=""
cleanup() {
  if [ -n "$MON" ]; then kill "$MON" 2>/dev/null || true; fi
  kill "$SRV" 2>/dev/null || true
}
trap cleanup EXIT TERM INT

echo "waiting for /v1/models serving $MODEL (up to 30 minutes)..."
ready=0
for attempt in $(seq 1 900); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; then
    ready=1
    echo "server ready after $((attempt * 2)) seconds"
    break
  fi
  if ! kill -0 "$SRV" 2>/dev/null; then
    echo "ERROR: server exited during startup" >&2
    tail -160 "$SERVER_LOG"
    exit 3
  fi
  sleep 2
done
[ "$ready" -eq 1 ] || { echo "ERROR: server readiness timeout" >&2; tail -160 "$SERVER_LOG"; exit 4; }

nvidia-smi dmon -s pucm -d 10 -o DT > "$GPU_LOG" 2>&1 &
MON=$!

cd "$REPO"
"$PY" scripts/benchmark_judge_dp_replay.py \
  --endpoint "http://localhost:$PORT/v1" \
  --dumps "$INPUT_DUMP" \
  --out "$OUT" \
  --n "$N" \
  --concurrency "$CONCURRENCY" \
  --model "$MODEL" \
  --timeout "$TIMEOUT" 2>&1 | tee "$CLIENT_LOG"
RC=${PIPESTATUS[0]}

kill "$MON" 2>/dev/null || true
wait "$MON" 2>/dev/null || true
MON=""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
echo "client_exit=$RC date=$(date --iso-8601=seconds)"
exit "$RC"
