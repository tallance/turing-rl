#!/bin/bash
# Orchestrate the 8B judge throughput sweep at TP=1, 2, 4, 8.
#
# 1. Sbatch 4 judge servers in parallel. Each gets a unique PORT
#    (8123/8124/8125/8126) so they can safely co-schedule on the same node.
# 2. Wait for each to reach RUNNING and answer /v1/models on its port.
# 3. Run scripts/benchmark_judge_throughput.py against all 4 endpoints
#    (sweeps client concurrency [1,4,16,64] per endpoint).
# 4. Trap-scancel all 4 judges on exit (success OR failure).
#
# GPU footprint: 15 across ≤4 nodes (1+2+4+8). Time budget: ~30 min after warmup
# at N=10 (measurement wall clock ≈ N × p50_latency / concurrency).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh
LOGS=$REPO/logs
OUT=$REPO/results/judge_throughput
mkdir -p "$LOGS" "$OUT"

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
DUMPS=/home/lancewicki/tmp/judge_dumps/judge-9239-1968156.jsonl
# N=10 keeps low-concurrency measurements < 5 min each. Scale up if the report
# looks noisy (throughput variance high).
N=10
CONCURRENCIES=1,4,16,64
TPS=(1 2 4 8)

# Unique port per TP so co-scheduled judges don't collide on host:port. Slurm
# cgroup isolation lets them boot without EADDRINUSE, but our client would hit
# whichever process the kernel routes to (arbitrary). Different ports → clean.
declare -A PORT_OF
PORT_OF[1]=8123
PORT_OF[2]=8124
PORT_OF[4]=8125
PORT_OF[8]=8126

# --- submit all 4 judges in parallel ---
declare -A JOB_ID   # tp -> slurm job id
declare -A QUEUE_T0 # tp -> submit timestamp
for TP in "${TPS[@]}"; do
  PORT=${PORT_OF[$TP]}
  echo "[sweep] submitting judge tp=$TP port=$PORT..."
  QUEUE_T0[$TP]=$(date +%s)
  JOB=$("$SBATCH" --parsable \
    --gres=gpu:$TP \
    --job-name=judge_8b_tp$TP \
    --export=ALL,TP=$TP,PORT=$PORT \
    -- \
    "$REPO/scripts/slurm/judge_serve_8b_tp.sh")
  if [ -z "$JOB" ]; then
    echo "  sbatch returned no id for tp=$TP" >&2
    exit 2
  fi
  JOB_ID[$TP]=$JOB
  echo "  tp=$TP job_id=$JOB log=$LOGS/judge_serve_8b_tp$JOB.out"
done

# --- cleanup on exit ---
cleanup() {
  echo ""
  echo "[sweep] cleanup: scancel judge jobs"
  for TP in "${TPS[@]}"; do
    JOB=${JOB_ID[$TP]:-}
    [ -n "$JOB" ] && scancel "$JOB" 2>/dev/null || true
  done
}
trap cleanup EXIT

# --- wait for each judge to RUNNING ---
declare -A NODE       # tp -> hostname
declare -A WARMUP_T0  # tp -> RUNNING timestamp
declare -A WARMUP_S   # tp -> RUNNING -> /v1/models seconds
for TP in "${TPS[@]}"; do
  JOB=${JOB_ID[$TP]}
  echo "[sweep] tp=$TP job=$JOB waiting for RUNNING..."
  for i in $(seq 1 2880); do  # up to 4h queue
    STATE_NODE=$(squeue -h -j "$JOB" -o '%T %N' 2>/dev/null || true)
    if [ -z "$STATE_NODE" ]; then
      echo "  tp=$TP job=$JOB vanished from queue" >&2
      tail -60 "$LOGS/judge_serve_8b_tp$JOB.out" 2>/dev/null
      exit 3
    fi
    ST=${STATE_NODE%% *}; ND=${STATE_NODE##* }
    if [ "$ST" = "RUNNING" ] && [ -n "$ND" ] && [ "$ND" != "(None)" ]; then
      NODE[$TP]=$ND
      WARMUP_T0[$TP]=$(date +%s)
      echo "  tp=$TP RUNNING on $ND after $((i*5))s in queue"
      break
    fi
    [ $((i % 60)) -eq 0 ] && echo "  tp=$TP still queued: state=$ST elapsed=$((i*5))s"
    sleep 5
  done
  [ -z "${NODE[$TP]:-}" ] && { echo "tp=$TP never reached RUNNING" >&2; exit 4; }
done

# --- poll /v1/models on each judge's port ---
for TP in "${TPS[@]}"; do
  ND=${NODE[$TP]}
  PORT=${PORT_OF[$TP]}
  JOB=${JOB_ID[$TP]}
  echo "[sweep] tp=$TP polling http://$ND:$PORT/v1/models (up to 10 min)..."
  READY=0
  for i in $(seq 1 120); do
    if curl -sf --max-time 5 "http://$ND:$PORT/v1/models" -o /dev/null 2>/dev/null; then
      WARMUP_S[$TP]=$(( $(date +%s) - WARMUP_T0[$TP] ))
      echo "  tp=$TP READY after ${i}*5s (warmup ${WARMUP_S[$TP]}s from RUNNING)"
      READY=1
      break
    fi
    if ! squeue -h -j "$JOB" -o '%T' 2>/dev/null | grep -q .; then
      echo "  tp=$TP job died during warmup" >&2
      tail -120 "$LOGS/judge_serve_8b_tp$JOB.out" 2>/dev/null
      exit 5
    fi
    [ $((i % 12)) -eq 0 ] && echo "  tp=$TP still warming (${i}*5s)"
    sleep 5
  done
  [ "$READY" -ne 1 ] && { echo "tp=$TP /v1/models never responded" >&2; exit 6; }
done

# --- assemble meta json for the report ---
# Build the python source as a bash heredoc first so $TP etc. expand cleanly
# (previous single-quoted-inside-python-string form ate the expansions).
TS=$(date +%Y%m%d-%H%M%S)
META_JSON=$OUT/meta-$TS.json
META_PY=$(cat <<EOF
import json
meta = {}
EOF
)
for TP in "${TPS[@]}"; do
  META_PY+="
meta['tp$TP'] = {'tp': $TP, 'job_id': '${JOB_ID[$TP]}', 'node': '${NODE[$TP]}', 'port': ${PORT_OF[$TP]}, 'warmup_s': ${WARMUP_S[$TP]}}"
done
META_PY+="
open('$META_JSON', 'w').write(json.dumps(meta, indent=2))
print('wrote', '$META_JSON')"
$PY -c "$META_PY"

# --- run the benchmark client ---
echo ""
echo "[sweep] running benchmark client (n=$N, concurrencies=$CONCURRENCIES)..."
ENDPOINT_ARGS=()
for TP in "${TPS[@]}"; do
  ENDPOINT_ARGS+=(--endpoint "tp$TP=http://${NODE[$TP]}:${PORT_OF[$TP]}/v1")
done

cd "$REPO"
$PY scripts/benchmark_judge_throughput.py \
  "${ENDPOINT_ARGS[@]}" \
  --dumps "$DUMPS" \
  --n "$N" \
  --concurrency "$CONCURRENCIES" \
  --meta-json "$META_JSON" \
  --out "$OUT"
RC=$?

echo ""
echo "[sweep] benchmark exit: $RC"
echo "[sweep] artifacts in $OUT"
ls -la "$OUT"/*.jsonl "$OUT"/*.md 2>/dev/null | tail -10
exit $RC
