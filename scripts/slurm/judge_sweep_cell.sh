#!/bin/bash
#SBATCH --job-name=sweep_cell
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sweep_cell-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# All-in-one judge sweep cell: boots REPLICAS vLLM servers on one 8-GPU node,
# waits for health, then launches REPLICAS process-sharded clients
# (scripts/run_judge_sweep_cell.py) that score the frozen 880-pair set through
# the real GRPO reward path. Reward + HTTP dumps land under
#   results/2026-07-08-judge-sweep/raw/sweep/$CELL_NAME/$THINKING_MODE/{reward,http}
#
# Required env: MODEL TP REPLICAS THINKING_MODE CELL_NAME
# Optional env: PORT_BASE (default 8130), MAX_PAIRS (cap pairs for calibration/smoke)
# Submit with --gres=gpu:$((TP*REPLICAS)) (the launcher does this).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

for v in MODEL TP REPLICAS THINKING_MODE CELL_NAME; do
  [ -z "${!v:-}" ] && { echo "ERROR: $v unset" >&2; exit 2; }
done
case "$THINKING_MODE" in on|off) ;; *) echo "ERROR: THINKING_MODE must be on|off" >&2; exit 2 ;; esac
# Unique per-job default port base: this cluster does NOT isolate the network
# namespace per Slurm job, so co-scheduled gpu:1 cells on one node would collide
# on a fixed port (a client would then hit a co-tenant's wrong-model server and
# get a 404 "model does not exist"). Derive from SLURM_JOB_ID so each job differs.
PORT_BASE=${PORT_BASE:-$((8130 + ${SLURM_JOB_ID:-0} % 800))}
MAX_PAIRS=${MAX_PAIRS:-}
[ $((REPLICAS*TP)) -gt 8 ] && { echo "ERROR: REPLICAS*TP>8 (asked $((REPLICAS*TP)))" >&2; exit 2; }

export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
# All cell models are pre-cached, so serve OFFLINE. Without this, every replica
# hits the HF hub API at startup (config/tokenizer resolution) even for cached
# weights; N replicas x several cells starting at once -> HF 429 Too Many Requests
# -> "Engine core initialization failed". Offline uses the local snapshot only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
REPO=/home/lancewicki/projects/turing-rl

# 397B GPTQ anchor serves from the pinned judge-vllm env; smaller/newer judges
# (incl. new Qwen3.5 dense archs) serve from turing-rl-train (newer vLLM/transformers).
case "$MODEL" in
  *397B*) PY_SERVER=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
  *)      PY_SERVER=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
esac
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

PAIRS=$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
[ -f "$PAIRS" ] || { echo "ERROR: pair-set not found: $PAIRS" >&2; exit 2; }

# The client appends $CELL_NAME/$THINKING_MODE to --out_dir, so pass the sweep ROOT.
SWEEP_ROOT=$REPO/results/2026-07-08-judge-sweep/raw/sweep
MODE_DIR=$SWEEP_ROOT/$CELL_NAME/$THINKING_MODE
mkdir -p "$MODE_DIR/vllm_server" "$MODE_DIR/reward" "$MODE_DIR/http"

echo "============================================"
echo "sweep cell: MODEL=$MODEL CELL_NAME=$CELL_NAME MODE=$THINKING_MODE TP=$TP REPLICAS=$REPLICAS"
echo "date=$(date) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "out=$MODE_DIR"
echo "============================================"

RP=()
[ "$THINKING_MODE" = "on" ] && RP=(--reasoning-parser qwen3)

PIDS=(); URLS=()
for i in $(seq 0 $((REPLICAS-1))); do
  gpus=$(seq -s, $((i*TP)) $((i*TP+TP-1)))
  port=$((PORT_BASE+i))
  CUDA_VISIBLE_DEVICES=$gpus $PY_SERVER -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
    --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 \
    "${RP[@]}" --host 0.0.0.0 --port $port \
    > "$MODE_DIR/vllm_server/replica_$i.log" 2>&1 &
  PIDS+=($!)
  URLS+=("http://localhost:$port/v1")
done

cleanup() { for p in "${PIDS[@]}"; do kill $p 2>/dev/null || true; done; }
trap cleanup EXIT

# Wait for every replica's /v1/models AND verify it serves OUR model (guards
# against a false-positive ready when a co-scheduled job's server holds the port).
for i in $(seq 0 $((REPLICAS-1))); do
  port=$((PORT_BASE+i)); ok=0
  for t in $(seq 1 900); do
    if curl -sf -m 2 http://localhost:$port/v1/models 2>/dev/null | grep -qF "\"$MODEL\""; then
      ok=1; break
    fi
    sleep 2
  done
  [ $ok -eq 1 ] || { echo "TIMEOUT waiting on replica $i (port $port) serving $MODEL" >&2; exit 3; }
  echo "replica $i ready (port $port, model $MODEL)"
done

ENDPOINTS=$(IFS=,; echo "${URLS[*]}")
EXTRA=(); [ -n "$MAX_PAIRS" ] && EXTRA=(--max_pairs "$MAX_PAIRS")

cd "$REPO"
CLIENT_PIDS=()
for i in $(seq 0 $((REPLICAS-1))); do
  $PY_CLIENT scripts/run_judge_sweep_cell.py \
    --pairs "$PAIRS" --endpoints "$ENDPOINTS" \
    --model "$MODEL" --thinking_mode "$THINKING_MODE" \
    --out_dir "$SWEEP_ROOT" --cell_name "$CELL_NAME" \
    --concurrency_per_endpoint 16 \
    --endpoint_index $i --num_endpoints $REPLICAS "${EXTRA[@]}" &
  CLIENT_PIDS+=($!)
done

RC=0
for p in "${CLIENT_PIDS[@]}"; do wait $p || RC=1; done
echo "=== clients exit: $RC ==="
exit $RC
