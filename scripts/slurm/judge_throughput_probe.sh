#!/bin/bash
#SBATCH --job-name=judge_probe_397b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_probe_397b-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# 397B judge THROUGHPUT PROBE (sizing for the RL-generator plan).
#
# Boots the Qwen3.5-397B-A17B-GPTQ-Int4 judge (TP=8, thinking-ON, the paper-
# faithful mode) on this node, then sweeps CLIENT concurrency with real 34k-char
# Turing prompts to find sustained calls/s + the latency knee + the client
# timeout needed so reward calls don't time out during GRPO. thinking-on because
# the OpenRouter probe proved the paper's judge thinks by default (~5k tokens,
# ~217s/call at conc-32 in the sweep, which then hit 207 timeouts at a 400s
# client timeout). This pins wall-clock and the safe operating concurrency.
#
# Env overrides: CONCURRENCIES (default 16,32,64,96), N (default 64),
#   TIMEOUT (default 1800), PARSER (default qwen3).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=INFO
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/home/lancewicki/projects/turing-rl
# Judge under test (default 397B anchor). Override via env for other judges,
# e.g. MODEL=Qwen/Qwen3.5-9B TP=1 LABEL=q9b (submit with --gres=gpu:1).
MODEL=${MODEL:-Qwen/Qwen3.5-397B-A17B-GPTQ-Int4}
TP=${TP:-8}
LABEL=${LABEL:-judge}
# 397B GPTQ serves from the pinned judge-vllm env; newer Qwen3.5 dense/hybrid
# judges (4b/9b/27b) need turing-rl-train's newer vLLM.
case "$MODEL" in
  *397B*) PY_SERVER=/home/lancewicki/miniconda3/envs/judge-vllm/bin/python ;;
  *)      PY_SERVER=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
esac
PY_CLIENT=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
PORT=$((8200 + ${SLURM_JOB_ID:-0} % 400))
CONCURRENCIES=${CONCURRENCIES:-16,32,64,96}
N=${N:-64}
TIMEOUT=${TIMEOUT:-1800}
PARSER=${PARSER:-qwen3}
# custom all-reduce kernel fails on A100 (cap 8.0) at TP>1 -> NCCL fallback.
AR=(); [ "$TP" -gt 1 ] && AR=(--disable-custom-all-reduce)

# Real full-length Turing prompts (payload_messages) from the sweep's 397B run.
DUMPS=${DUMPS:-$REPO/results/2026-07-08-judge-sweep/raw/sweep/qwen35-397b/on/http/judge-9708-359528.jsonl}
[ -f "$DUMPS" ] || { echo "ERROR: prompt dump not found: $DUMPS" >&2; exit 2; }
OUT=$REPO/results/judge_throughput
mkdir -p "$OUT" "$REPO/logs"

echo "============================================"
echo "397B judge throughput probe"
echo "date=$(date) host=$(hostname) port=$PORT"
echo "model=$MODEL parser=$PARSER conc=$CONCURRENCIES n=$N timeout=${TIMEOUT}s"
echo "dumps=$DUMPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

# Serve: TP=8 across the node; NCCL all-reduce (custom AR kernel fails on A100
# 8.0 at TP>1); thinking-on via --reasoning-parser (splits <think> to .reasoning).
$PY_SERVER -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --download-dir "$HF_HOME" --tensor-parallel-size "$TP" \
  --max-model-len 32768 --gpu-memory-utilization 0.85 --dtype bfloat16 \
  --reasoning-parser "$PARSER" "${AR[@]}" \
  --host 0.0.0.0 --port "$PORT" > "$OUT/probe_server-${SLURM_JOB_ID}.log" 2>&1 &
SRV=$!
cleanup() { kill $SRV 2>/dev/null || true; }
trap cleanup EXIT

echo "waiting for /v1/models serving $MODEL (up to 30 min warmup)..."
ok=0
for t in $(seq 1 900); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; then
    ok=1; echo "judge ready after $((t*2))s"; break
  fi
  kill -0 $SRV 2>/dev/null || { echo "server process died during warmup" >&2; tail -80 "$OUT/probe_server-${SLURM_JOB_ID}.log"; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge" >&2; tail -120 "$OUT/probe_server-${SLURM_JOB_ID}.log"; exit 4; }

cd "$REPO"
$PY_CLIENT scripts/benchmark_judge_throughput.py \
  --endpoint "$LABEL=http://localhost:$PORT/v1" \
  --model "$MODEL" --dumps "$DUMPS" \
  --n "$N" --concurrency "$CONCURRENCIES" --timeout "$TIMEOUT" \
  --out "$OUT"
RC=$?
echo "=== probe client exit: $RC ==="
exit $RC
