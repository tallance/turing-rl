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
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

for v in MODEL TP REPLICAS THINKING_MODE CELL_NAME; do
  [ -z "${!v:-}" ] && { echo "ERROR: $v unset" >&2; exit 2; }
done
case "$THINKING_MODE" in on|off) ;; *) echo "ERROR: THINKING_MODE must be on|off" >&2; exit 2 ;; esac
JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE:-full}
case "$JUDGE_PROMPT_STYLE" in
  full|single_token) ;;
  *) echo "ERROR: JUDGE_PROMPT_STYLE must be full|single_token, got '$JUDGE_PROMPT_STYLE'" >&2; exit 2 ;;
esac
export JUDGE_PROMPT_STYLE
# Unique per-job default port base: this cluster does NOT isolate the network
# namespace per Slurm job, so co-scheduled gpu:1 cells on one node would collide
# on a fixed port (a client would then hit a co-tenant's wrong-model server and
# get a 404 "model does not exist"). Derive from SLURM_JOB_ID so each job differs.
PORT_BASE=${PORT_BASE:-$((8130 + ${SLURM_JOB_ID:-0} % 800))}
MAX_PAIRS=${MAX_PAIRS:-}
# In-flight requests per endpoint. Pure throughput knob (no effect on verdicts;
# vLLM queues if KV is tight). 32 keeps the single-endpoint anchor busy.
CONCURRENCY=${CONCURRENCY:-32}
[ $((REPLICAS*TP)) -gt 8 ] && { echo "ERROR: REPLICAS*TP>8 (asked $((REPLICAS*TP)))" >&2; exit 2; }

# vLLM allocates additional TCPStore/ZMQ ports internally. Its default
# close-before-bind probe can race when several replicas start together and
# select the same ephemeral port. Give each replica a deterministic port band;
# VLLM_PORT is the start of the internal scan, not the OpenAI API port below.
VLLM_INTERNAL_PORT_BASE=${VLLM_INTERNAL_PORT_BASE:-20000}
VLLM_INTERNAL_PORT_STRIDE=${VLLM_INTERNAL_PORT_STRIDE:-100}
case "$VLLM_INTERNAL_PORT_BASE:$VLLM_INTERNAL_PORT_STRIDE" in
  *[!0-9:]*|:*|*:)
    echo "ERROR: VLLM internal port base/stride must be positive integers" >&2
    exit 2
    ;;
esac
[ "$VLLM_INTERNAL_PORT_BASE" -ge 1024 ] && [ "$VLLM_INTERNAL_PORT_STRIDE" -ge 32 ] || {
  echo "ERROR: VLLM internal port base must be >=1024 and stride must be >=32" >&2
  exit 2
}
VLLM_INTERNAL_PORT_LAST=$((VLLM_INTERNAL_PORT_BASE + REPLICAS * VLLM_INTERNAL_PORT_STRIDE - 1))
[ "$VLLM_INTERNAL_PORT_LAST" -le 65535 ] || {
  echo "ERROR: VLLM internal port bands exceed TCP port 65535" >&2
  exit 2
}

export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
# All cell models are pre-cached, so serve OFFLINE. Without this, every replica
# hits the HF hub API at startup (config/tokenizer resolution) even for cached
# weights; N replicas x several cells starting at once -> HF 429 Too Many Requests
# -> "Engine core initialization failed". Offline uses the local snapshot only.
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Reduce CUDA allocator fragmentation (helps the memory-tight dense cells fit).
export PYTORCH_ALLOC_CONF=expandable_segments:True
# Allow an isolated checkout while PAIRS/SWEEP_ROOT point at canonical results.
REPO=${REPO:-/home/lancewicki/projects/turing-rl}

# 397B uses its pinned environment; Gemma 4 Unified uses the tested CUDA-13
# nightly environment and exact offline snapshots; Qwen keeps the prior path.
IS_GEMMA4=0
case "$MODEL" in
  *397B*) PY_SERVER=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
  google/gemma-4-12B-it)
    IS_GEMMA4=1
    GEMMA_SNAPSHOT=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    ;;
  google/gemma-4-31B-it)
    IS_GEMMA4=1
    GEMMA_SNAPSHOT=842da3794eaa0b77d5f08bae87a17459d91ff475
    ;;
  *)      PY_SERVER=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
esac
if [ "$IS_GEMMA4" = "1" ]; then
  GEMMA_VLLM=/home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly/bin/vllm
  GEMMA_CACHE=/home/lancewicki/data/hf_cache/hub/models--google--${MODEL#google/}
  GEMMA_MODEL_PATH=$GEMMA_CACHE/snapshots/$GEMMA_SNAPSHOT
  [ -x "$GEMMA_VLLM" ] || { echo "ERROR: missing Gemma vLLM: $GEMMA_VLLM" >&2; exit 2; }
  [ -f "$GEMMA_MODEL_PATH/config.json" ] || {
    echo "ERROR: incomplete Gemma snapshot: $GEMMA_MODEL_PATH" >&2
    exit 2
  }
fi
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

# Default full 880 pair-set; override with PAIRS=<parquet> (e.g. a missing-pairs
# subset for a targeted re-run of timed-out pairs).
PAIRS=${PAIRS:-$REPO/results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet}
[ -f "$PAIRS" ] || { echo "ERROR: pair-set not found: $PAIRS" >&2; exit 2; }

# The client appends $CELL_NAME/$THINKING_MODE to --out_dir, so pass the sweep ROOT.
SWEEP_ROOT=${SWEEP_ROOT:-$REPO/results/2026-07-08-judge-sweep/raw/sweep}
# --- BEGIN mode-dir ---
# Kept identical to run_judge_sweep_cell.cell_output_dirs() and to the stale-output guard
# in launch_judge_eval_matrix.sh. If the writer and the guard disagree the guard inspects
# a directory nothing writes, and a single_token rerun appends into the full-schema cell.
# tests/test_judge_sweep_cell_paths.py executes this block and the guard against each other.
MODE_DIR=$SWEEP_ROOT/$CELL_NAME/$THINKING_MODE
[ "$JUDGE_PROMPT_STYLE" = "full" ] || MODE_DIR=$MODE_DIR/$JUDGE_PROMPT_STYLE
# --- END mode-dir ---
mkdir -p "$MODE_DIR/vllm_server" "$MODE_DIR/reward" "$MODE_DIR/http"
TIMING_JOB_STARTED_EPOCH=$(date +%s.%N)
TIMING_JOB_STARTED_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)

echo "============================================"
echo "sweep cell: MODEL=$MODEL CELL_NAME=$CELL_NAME MODE=$THINKING_MODE STYLE=$JUDGE_PROMPT_STYLE TP=$TP REPLICAS=$REPLICAS"
echo "date=$(date) host=$(hostname)"
[ "$IS_GEMMA4" = "1" ] && echo "gemma_snapshot=$GEMMA_SNAPSHOT"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "out=$MODE_DIR"
echo "============================================"

RP=()
if [ "$THINKING_MODE" = "on" ]; then
  if [ "$IS_GEMMA4" = "1" ]; then
    RP=(--reasoning-parser gemma4)
  else
    RP=(--reasoning-parser qwen3)
  fi
fi

# On these A100s (capability 8.0) vLLM's custom all-reduce kernel fails at TP>1
# for some models (observed: Qwen3.5-27B dense -> "custom_all_reduce.cuh:455
# invalid argument", every TP config, NOT an OOM). Fall back to NCCL all-reduce
# for any multi-GPU cell -- robust, numerically equivalent, negligible cost here.
AR=()
[ "$TP" -gt 1 ] && AR=(--disable-custom-all-reduce)

# Optional quantization override. Empty = let vLLM auto-detect from the checkpoint
# (bf16 weights, or GPTQ/AWQ from config.json). Set QUANT=fp8 to dynamically
# quantize a bf16 checkpoint to fp8 at load (W8A16 on Ampere via Marlin) so a
# too-big-for-bf16 model fits (e.g. 122B: 234GB bf16 -> ~122GB fp8).
QZ=()
[ -n "${QUANT:-}" ] && QZ=(--quantization "$QUANT")

# Optional speculative decoding (server-side THROUGHPUT experiment; unrelated to the
# parse-failure investigation -- see post-plan 2026-07-14-cot-failure-diagnostic.md).
# SPEC_DECODE = a vLLM --speculative-config JSON, e.g. ngram (no draft model needed,
# helps most when output repeats/copies the prompt, which this judge does):
#   {"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4,"prompt_lookup_min":2}
SD=()
[ -n "${SPEC_DECODE:-}" ] && SD=(--speculative-config "$SPEC_DECODE")

# Optional max_num_seqs cap. Needed with SPEC_DECODE on the hybrid-Mamba 397B: spec
# decoding reserves draft-token slots that shrink the Mamba cache (~158 blocks), and the
# default max_num_seqs=256 > blocks aborts CUDA-graph capture. Concurrency is small (8),
# so capping to <=158 is free. (See post-plan: job 9826 failure.)
MS=()
[ -n "${MAX_NUM_SEQS:-}" ] && MS=(--max-num-seqs "$MAX_NUM_SEQS")

GPU_MEMORY_UTILIZATION=0.85
GEMMA_ARGS=()
if [ "$IS_GEMMA4" = "1" ]; then
  GPU_MEMORY_UTILIZATION=${GEMMA_GPU_MEMORY_UTILIZATION:-0.90}
  GEMMA_ARGS=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}')
  export TMPDIR=${TMPDIR:-/home/lancewicki/tmp/build}
  export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-/home/lancewicki/tmp/flashinfer}
  mkdir -p "$TMPDIR" "$FLASHINFER_WORKSPACE_BASE"
fi

PIDS=(); URLS=()
for i in $(seq 0 $((REPLICAS-1))); do
  gpus=$(seq -s, $((i*TP)) $((i*TP+TP-1)))
  port=$((PORT_BASE+i))
  internal_port=$((VLLM_INTERNAL_PORT_BASE + i * VLLM_INTERNAL_PORT_STRIDE))
  echo "replica $i ports: api=$port internal_start=$internal_port"
  if [ "$IS_GEMMA4" = "1" ]; then
    VLLM_PORT=$internal_port CUDA_VISIBLE_DEVICES=$gpus "$GEMMA_VLLM" serve "$GEMMA_MODEL_PATH" \
      --served-model-name "$MODEL" \
      --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
      --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --dtype bfloat16 \
      "${RP[@]}" "${AR[@]}" "${QZ[@]}" "${SD[@]}" "${MS[@]}" "${GEMMA_ARGS[@]}" \
      --host 0.0.0.0 --port "$port" > "$MODE_DIR/vllm_server/replica_$i.log" 2>&1 &
  else
    VLLM_PORT=$internal_port CUDA_VISIBLE_DEVICES=$gpus $PY_SERVER -m vllm.entrypoints.openai.api_server \
      --model "$MODEL" --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
      --max-model-len 32768 --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --dtype bfloat16 \
      "${RP[@]}" "${AR[@]}" "${QZ[@]}" "${SD[@]}" "${MS[@]}" --host 0.0.0.0 --port "$port" \
      > "$MODE_DIR/vllm_server/replica_$i.log" 2>&1 &
  fi
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
TIMING_SERVERS_READY_EPOCH=$(date +%s.%N)
TIMING_SERVERS_READY_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)

ENDPOINTS=$(IFS=,; echo "${URLS[*]}")
EXTRA=(); [ -n "$MAX_PAIRS" ] && EXTRA=(--max_pairs "$MAX_PAIRS")

cd "$REPO"
CLIENT_PIDS=()
for i in $(seq 0 $((REPLICAS-1))); do
  $PY_CLIENT scripts/run_judge_sweep_cell.py \
    --pairs "$PAIRS" --endpoints "$ENDPOINTS" \
    --model "$MODEL" --thinking_mode "$THINKING_MODE" \
    --out_dir "$SWEEP_ROOT" --cell_name "$CELL_NAME" \
    --prompt_style "$JUDGE_PROMPT_STYLE" \
    --concurrency_per_endpoint "$CONCURRENCY" \
    --endpoint_index $i --num_endpoints $REPLICAS "${EXTRA[@]}" &
  CLIENT_PIDS+=($!)
done

RC=0
for p in "${CLIENT_PIDS[@]}"; do wait $p || RC=1; done
TIMING_CLIENTS_FINISHED_EPOCH=$(date +%s.%N)
TIMING_CLIENTS_FINISHED_UTC=$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)
TIMING_PATH=$MODE_DIR/timing.json
TIMING_TMP=$TIMING_PATH.tmp.$$
"$PY_CLIENT" - "$TIMING_TMP" <<PY
import json
import os
import sys

started = float("$TIMING_JOB_STARTED_EPOCH")
ready = float("$TIMING_SERVERS_READY_EPOCH")
finished = float("$TIMING_CLIENTS_FINISHED_EPOCH")
record = {
    "format_version": 1,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    "model": "$MODEL",
    "cell_name": "$CELL_NAME",
    "thinking_mode": "$THINKING_MODE",
    "prompt_style": "$JUDGE_PROMPT_STYLE",
    "tp": int("$TP"),
    "replicas": int("$REPLICAS"),
    "concurrency_per_endpoint": int("$CONCURRENCY"),
    "pairs": "$PAIRS",
    "max_pairs": "$MAX_PAIRS" or None,
    "job_started_utc": "$TIMING_JOB_STARTED_UTC",
    "servers_ready_utc": "$TIMING_SERVERS_READY_UTC",
    "clients_finished_utc": "$TIMING_CLIENTS_FINISHED_UTC",
    "model_startup_seconds": round(ready - started, 3),
    "scoring_seconds": round(finished - ready, 3),
    "instrumented_total_seconds": round(finished - started, 3),
    "client_exit_code": int("$RC"),
}
with open(sys.argv[1], "w") as handle:
    json.dump(record, handle, indent=2, sort_keys=True)
    handle.write("\\n")
PY
TIMING_RC=$?
[ "$TIMING_RC" -eq 0 ] || { echo "ERROR: failed to write timing record" >&2; exit 4; }
mv "$TIMING_TMP" "$TIMING_PATH" || { echo "ERROR: failed to publish timing record" >&2; exit 4; }
echo "timing=$TIMING_PATH startup_s=$("$PY_CLIENT" -c "print(round(float('$TIMING_SERVERS_READY_EPOCH')-float('$TIMING_JOB_STARTED_EPOCH'),3))") scoring_s=$("$PY_CLIENT" -c "print(round(float('$TIMING_CLIENTS_FINISHED_EPOCH')-float('$TIMING_SERVERS_READY_EPOCH'),3))")"
echo "=== clients exit: $RC ==="
exit $RC
