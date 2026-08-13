#!/bin/bash
#SBATCH --job-name=judge_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_probe-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Phase 0 gate: serve one candidate judge and probe it in three decoding regimes.
# The freeform arm is the one that matters -- it is the only regime that matches what
# veRL rollouts will actually produce.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

JUDGE_MODEL=${JUDGE_MODEL:?set JUDGE_MODEL, e.g. Qwen/Qwen3.5-4B}
# NOT $REPO/data: inside a job, $REPO/data symlinks to the IMMUTABLE SOURCE SNAPSHOT,
# which carries only committed python modules -- generated parquets are invisible there.
# TURING_RL_GENERATED_DATA_ROOT is the state-root path where the builder actually writes.
PAIRS=${PAIRS:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1/val.parquet}
OUT_JSON=${OUT_JSON:-$REPO/results/judge-format-probe/$(basename "$JUDGE_MODEL").json}
LIMIT=${LIMIT:-200}
REGIMES=${REGIMES:-"json_schema json_object freeform"}
# The probe is the ONLY producer of the long-format CSV that
# scripts/analyze_judge_training.py consumes, so the Phase 0 run doubles as the zero-shot
# baseline scoring pass. Left unset, the gate still runs but no eval rows are recorded.
# --dump_regime pins the dumped regime to json_schema: the published comparison is under
# forced-schema decoding, and without it the dump silently takes whichever regime ran LAST.
DUMP_CSV=${DUMP_CSV:-}
DUMP_REGIME=${DUMP_REGIME:-json_schema}

export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export PERSONA_JUDGE_ENABLE_THINKING=1
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_OPENAI_TIMEOUT_SECONDS=1800
# The server is DP-8. At the client default of 8 there is one in-flight request per rank, so
# vLLM never batches -- docs/default-params.md measured 6.4x throughput going 8 -> 64, with
# latency flat, and pairs 64 with the 1800s timeout above (the 400s default is what caused
# the job-13628 timeout cascade). 600 calls at 8 takes hours; at 64 it takes minutes.
export JUDGE_PROBE_CONCURRENCY=${JUDGE_PROBE_CONCURRENCY:-64}

# The judge is served by a SEPARATE Slurm job, not backgrounded here.
# cluster_job_bootstrap.sh derives its runtime work directory from job-$SLURM_JOB_ID, so a
# nested bootstrap-sourcing script inside the same job collides on mkdir and dies with
# "runtime work directory already exists" (jobs 15951-15953, all three, 12s in). The serve
# job therefore has its own SLURM_JOB_ID and its own work dir; this job only waits for the
# endpoint it publishes, and tears it down afterwards. That is also why this job needs NO
# GPU: it is an HTTP client.
ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:?set JUDGE_ENDPOINT_FILE (shared with the serve job)}
JUDGE_SERVE_JOB_ID=${RL_JUDGE_JOB_ID:?set RL_JUDGE_JOB_ID so the server is torn down}
cleanup() { scancel "$JUDGE_SERVE_JOB_ID" 2>/dev/null || true; }
trap 'cleanup; exit 143' TERM INT
trap cleanup EXIT

echo "waiting for judge endpoint from job $JUDGE_SERVE_JOB_ID (up to 60 min warmup)..."
ok=0
for _ in $(seq 1 1800); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  state=$(squeue -j "$JUDGE_SERVE_JOB_ID" -h -o '%t' 2>/dev/null | tr -d ' ')
  [ -n "$state" ] || { echo "ERROR: judge serve job $JUDGE_SERVE_JOB_ID left the queue before publishing an endpoint" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge endpoint" >&2; exit 4; }
export OPENAI_API_BASE=$(cat "$ENDPOINT_FILE")
echo "judge endpoint: $OPENAI_API_BASE"

DUMP_ARGS=()
if [ -n "$DUMP_CSV" ]; then
  mkdir -p "$(dirname "$DUMP_CSV")"
  DUMP_ARGS=(--dump_csv "$DUMP_CSV" --dump_regime "$DUMP_REGIME" --model_label "$JUDGE_MODEL")
fi

# shellcheck disable=SC2086
$PY -u scripts/probe_judge_format.py \
  --pairs_parquet "$PAIRS" --model "$JUDGE_MODEL" \
  --out_json "$OUT_JSON" --limit "$LIMIT" --regimes $REGIMES "${DUMP_ARGS[@]}"
