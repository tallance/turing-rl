#!/bin/bash
#SBATCH --job-name=judge_parity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=01:30:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_parity-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Step 0b for MODE=frac10ep10: prove the reward layer can still read the judge when it is
# served through the `vllm serve` frontend instead of the api_server module.
#
# Deliberately boots the judge with scripts/slurm/judge_serve_9b_replicas.sh --
# the SAME script the real run uses -- so this exercises the JUDGE_ENTRYPOINT=serve branch
# end to end rather than a hand-rolled approximation of it.
#
# Runs as its own job, NOT overlapped onto the durability arm: judge_dp_replay.sh traps EXIT
# and kills its server, so that server does not outlive its client, and injecting extra
# traffic while the client is still running would land inside the window whose throughput
# the arm exists to measure.
#
# Env: JUDGE_ENTRYPOINT (default serve), N (default 50), MAX_PARSE_FAILURE (default 0.25).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=/home/lancewicki/projects/turing-rl
cd "$REPO" || exit 2

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
MODEL=${MODEL:-Qwen/Qwen3.5-9B}
JUDGE_ENTRYPOINT=${JUDGE_ENTRYPOINT:-serve}
N=${N:-50}
MAX_PARSE_FAILURE=${MAX_PARSE_FAILURE:-0.25}
DUMP=${DUMP:-$REPO/results/grpo/rl-generator/9b_full5ep_kl1e4_lr1e4_temp1/reward_dump/reward-14217-1041480.jsonl}
OUT=${OUT:-$REPO/results/judge_parity/${SLURM_JOB_ID}}
ENDPOINT_FILE=$OUT/judge_endpoint.txt

[ -f "$DUMP" ] || { echo "ERROR: missing dump: $DUMP" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "ERROR: output already exists: $OUT" >&2; exit 2; }
mkdir -p "$OUT" "$REPO/logs"

echo "============================================"
echo "judge parity check"
echo "date=$(date --iso-8601=seconds) host=$(hostname) job=${SLURM_JOB_ID:-none}"
echo "entrypoint=$JUDGE_ENTRYPOINT model=$MODEL n=$N max_parse_failure=$MAX_PARSE_FAILURE"
echo "dump=$DUMP"
echo "out=$OUT"
echo "deployed_sha=$(cat "$REPO/DEPLOYED_SHA" 2>/dev/null || echo missing)"
echo "============================================"

MODEL=$MODEL TP=1 DP=8 JUDGE_ENTRYPOINT=$JUDGE_ENTRYPOINT JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  bash scripts/slurm/judge_serve_9b_replicas.sh > "$OUT/server.log" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

echo ">> waiting for judge endpoint (up to 40 min warmup)..."
ok=0
for _ in $(seq 1 1200); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$SRV" 2>/dev/null || { echo "ERROR: judge died during warmup; see $OUT/server.log" >&2; tail -30 "$OUT/server.log" >&2; exit 3; }
  sleep 2
done
[ "$ok" -eq 1 ] || { echo "TIMEOUT waiting for judge endpoint" >&2; exit 4; }

ENDPOINT=$(cat "$ENDPOINT_FILE")
echo ">> judge endpoint: $ENDPOINT"
grep -m1 'judge vllm version' "$OUT/server.log" || true
grep -m1 '=== judge entrypoint' "$OUT/server.log" || true

"$PY" scripts/check_judge_parity.py \
  --endpoint "$ENDPOINT" --dump "$DUMP" --model "$MODEL" \
  --n "$N" --max-parse-failure "$MAX_PARSE_FAILURE" \
  --out "$OUT/parity_summary.json" 2>&1 | tee "$OUT/client.log"
RC=${PIPESTATUS[0]}

echo "=== parity check exit: $RC ==="
exit $RC
