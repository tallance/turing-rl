#!/bin/bash
#SBATCH --job-name=judge_conc_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_conc_probe-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Measure judge throughput vs client concurrency on the DP-8 topology the GRPO trainer uses.
#
# WHY: job 13634 spent 41.5 of its 44.1 h waiting on the judge (93.9%). Reward dumps show
# 9952 calls at ~101 s mean latency and an EFFECTIVE CONCURRENCY OF ONLY 6.3 -- the trainer
# ran with TURING_JUDGE_MAX_CONCURRENCY=8, i.e. one in-flight request per DP rank, so vLLM
# never got to batch. The eval path (8 separate servers x 32 concurrency) sustained
# 0.49 calls/s vs training's 0.063 -- 7.8x on identical hardware.
#
# The cap was added to stop the job-13628 timeout cascade (concurrency 128 at a 400 s
# timeout: queue wait exceeded the timeout and every request failed). That fixed the crash
# by starving the server. This probe tests the alternative -- high concurrency with a long
# timeout -- and finds the knee.
#
# Self-contained: boots one vLLM DP=8 server on this node, sweeps client concurrency, tears
# the server down on exit. Prompts come from the REAL 9B reward dumps (~22k chars each), so
# latency reflects production prompt length rather than a synthetic short prompt.
#
# Env: CONCURRENCIES (default 8,32,64,128)  N (default 64)  TIMEOUT (default 1800)
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 VLLM_LOGGING_LEVEL=WARNING
export PYTORCH_ALLOC_CONF=expandable_segments:True

REPO=/home/lancewicki/projects/turing-rl
cd "$REPO" || exit 2
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
VLLM=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/vllm

MODEL=${MODEL:-Qwen/Qwen3.5-9B}
DP=${DP:-8}
CONCURRENCIES=${CONCURRENCIES:-8,32,64,128}
N=${N:-64}
TIMEOUT=${TIMEOUT:-1800}
PORT=${PORT:-$((8500 + ${SLURM_JOB_ID:-0} % 300))}
OUT=$REPO/results/judge_throughput
STAMP=$(date +%Y%m%d-%H%M%S)
PROMPTS=/home/lancewicki/tmp/judge_dumps/turing9b_real_prompts.jsonl
mkdir -p "$OUT" "$(dirname "$PROMPTS")"

# --- build probe payloads from the real training reward dumps -------------------------
if [ ! -s "$PROMPTS" ]; then
  echo "[probe] building prompts from the 9B reward dumps..."
  $PY - "$PROMPTS" <<'PROMPTGEN'
import glob, json, sys
out = sys.argv[1]
n = 0
with open(out, "w") as w:
    for f in sorted(glob.glob(
            "results/grpo/rl-generator/9b_half_kl1e4_lr1e4_temp1/reward_dump/*.jsonl")):
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            p = r.get("judge_prompt")
            if not isinstance(p, str) or len(p) < 1000:
                continue
            w.write(json.dumps({"payload_messages": [{"role": "user", "content": p}]}) + "\n")
            n += 1
            if n >= 256:
                break
        if n >= 256:
            break
print(f"[probe] wrote {n} real judge prompts -> {out}")
PROMPTGEN
fi

echo "=== judge concurrency probe: MODEL=$MODEL DP=$DP conc=$CONCURRENCIES N=$N timeout=${TIMEOUT}s ==="
echo "=== node=$(hostname) port=$PORT date=$(date) ==="

# --- boot ONE DP-8 server (the trainer's topology: single endpoint, load-balanced) ----
HF_HUB_OFFLINE=1 "$VLLM" serve "$MODEL" \
  --tensor-parallel-size 1 --data-parallel-size "$DP" --port "$PORT" \
  --reasoning-parser qwen3 --gpu-memory-utilization 0.85 \
  --max-model-len 32768 --download-dir /home/lancewicki/data/hf_cache &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null || true; }
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

echo "[probe] waiting for /v1/models (up to 30 min)..."
ok=0
for t in $(seq 1 900); do
  if curl -sf -m 2 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -qF "\"$MODEL\""; then
    ok=1; echo "[probe] server ready after $((t*2))s"; break
  fi
  kill -0 "$SRV" 2>/dev/null || { echo "ERROR: server died during warmup" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for server" >&2; exit 4; }

# --- sweep client concurrency ---------------------------------------------------------
$PY scripts/benchmark_judge_throughput.py \
    --endpoint "dp${DP}=http://localhost:$PORT/v1" \
    --dumps "$PROMPTS" \
    --model "$MODEL" \
    --n "$N" \
    --concurrency "$CONCURRENCIES" \
    --timeout "$TIMEOUT" \
    --out "$OUT/results-conc-$STAMP.jsonl"
RC=$?
echo "=== probe exit: $RC ; results -> $OUT/results-conc-$STAMP.jsonl ==="
exit $RC
