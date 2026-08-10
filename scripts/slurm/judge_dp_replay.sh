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
# Qwen3.5-9B DP=8 stack, changing only the launcher to the official multi-
# frontend CLI. SERVER_ENV can select a separately validated vLLM environment;
# the replay client remains in the original training environment.
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
SERVER_ENV=${SERVER_ENV:-/home/lancewicki/miniconda3/envs/turing-rl-train}
PY_SERVER=$SERVER_ENV/bin/python
VLLM=$SERVER_ENV/bin/vllm
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
N=${N:-512}
CONCURRENCY=${CONCURRENCY:-64}
TIMEOUT=${TIMEOUT:-1800}
DURATION=${DURATION:-0}
API_SERVER_COUNT=${API_SERVER_COUNT:-8}
PORT=${PORT:-$((8700 + ${SLURM_JOB_ID:-0} % 300))}
INPUT_DUMP=${INPUT_DUMP:-$REPO/results/grpo/rl-generator/9b_full5ep_kl1e4_lr1e4_temp1/reward_dump/reward-14217-1041480.jsonl}
OUT=${OUT:-$REPO/results/judge_dp_replay/${SLURM_JOB_ID}}

[ -x "$PY_SERVER" ] || { echo "ERROR: missing server Python: $PY_SERVER" >&2; exit 2; }
[ -x "$VLLM" ] || { echo "ERROR: missing vLLM CLI: $VLLM" >&2; exit 2; }
[ -x "$PY_CLIENT" ] || { echo "ERROR: missing client Python: $PY_CLIENT" >&2; exit 2; }
[ -f "$INPUT_DUMP" ] || { echo "ERROR: missing input dump: $INPUT_DUMP" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT" "$REPO/logs"

SERVER_LOG=$OUT/server.log
GPU_LOG=$OUT/gpu_dmon.log
METRICS_LOG=$OUT/metrics.log
PROCESS_LOG=$OUT/process_cpu.log
CLIENT_LOG=$OUT/client.log
SERVER_CMD=(
  "$VLLM" serve "$MODEL"
  --download-dir "$HF_HOME"
  --tensor-parallel-size 1
  --data-parallel-size 8
  --api-server-count "$API_SERVER_COUNT"
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
echo "model=$MODEL n=$N concurrency=$CONCURRENCY timeout=$TIMEOUT duration=$DURATION api_server_count=$API_SERVER_COUNT port=$PORT"
echo "input_dump=$INPUT_DUMP"
echo "out=$OUT"
echo "server_env=$SERVER_ENV"
echo "deployed_sha=$(cat "$REPO/DEPLOYED_SHA" 2>/dev/null || echo missing)"
"$PY_SERVER" -c 'import sys, torch, vllm; print("server_python={} torch={} cuda={} vllm={}".format(sys.version.split()[0], torch.__version__, torch.version.cuda, vllm.__version__))'
"$PY_CLIENT" -c 'import aiohttp, sys; print("client_python={} aiohttp={}".format(sys.version.split()[0], aiohttp.__version__))'
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
printf 'server_command='
printf '%q ' "${SERVER_CMD[@]}"
printf '\n'
echo "============================================"

"${SERVER_CMD[@]}" > "$SERVER_LOG" 2>&1 &
SRV=$!
MON=""
METRICS_MON=""
PROC_MON=""
cleanup() {
  if [ -n "$MON" ]; then kill "$MON" 2>/dev/null || true; fi
  if [ -n "$METRICS_MON" ]; then kill "$METRICS_MON" 2>/dev/null || true; fi
  if [ -n "$PROC_MON" ]; then kill "$PROC_MON" 2>/dev/null || true; fi
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

( while kill -0 "$SRV" 2>/dev/null; do
    date --iso-8601=ns
    curl -sf -m 5 "http://localhost:$PORT/metrics" || true
    sleep 10
  done ) > "$METRICS_LOG" 2>&1 &
METRICS_MON=$!

( while kill -0 "$SRV" 2>/dev/null; do
    date --iso-8601=ns
    ps -eo pid=,etimes=,time=,pcpu=,rss=,comm=,args= \
      | grep -E 'VLLM::APIServer|EngineCore_DP|VLLM::Worker|vllm serve' \
      | grep -v grep || true
    sleep 10
  done ) > "$PROCESS_LOG" 2>&1 &
PROC_MON=$!

nvidia-smi dmon -s pucm -d 10 -o DT > "$GPU_LOG" 2>&1 &
MON=$!

cd "$REPO"
"$PY_CLIENT" scripts/benchmark_judge_dp_replay.py \
  --endpoint "http://localhost:$PORT/v1" \
  --dumps "$INPUT_DUMP" \
  --out "$OUT" \
  --n "$N" \
  --concurrency "$CONCURRENCY" \
  --model "$MODEL" \
  --timeout "$TIMEOUT" \
  --duration "$DURATION" 2>&1 | tee "$CLIENT_LOG"
RC=${PIPESTATUS[0]}

kill "$MON" 2>/dev/null || true
wait "$MON" 2>/dev/null || true
MON=""
kill "$METRICS_MON" 2>/dev/null || true
wait "$METRICS_MON" 2>/dev/null || true
METRICS_MON=""
kill "$PROC_MON" 2>/dev/null || true
wait "$PROC_MON" 2>/dev/null || true
PROC_MON=""
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits
echo "client_exit=$RC date=$(date --iso-8601=seconds)"
exit "$RC"
