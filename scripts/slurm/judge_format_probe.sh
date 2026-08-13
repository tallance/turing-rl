#!/bin/bash
#SBATCH --job-name=judge_probe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=256G
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
PAIRS=${PAIRS:-$REPO/data/prism/judge/iter1/val.parquet}
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

# Serve the candidate judge, wait for its endpoint file (written only after model-verified
# health -- see judge_serve_9b_replicas.sh), then point the OpenAI-compatible probe client
# at it. Without this, resolve_judge_api_key()/get_openai_api_base() fall through to the
# real OpenAI endpoint instead of our vLLM server. Serving shape follows
# configs/judge_sweep_cells.py: <=30GB footprint -> TP=1 with 8 replicas.
ENDPOINT_FILE=${JUDGE_ENDPOINT_FILE:-$REPO/logs/judge_probe_endpoint-${SLURM_JOB_ID}.txt}
rm -f "$ENDPOINT_FILE"
MODEL=$JUDGE_MODEL JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  bash "$REPO/scripts/slurm/judge_serve_9b_replicas.sh" &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null || true' EXIT

echo "waiting for judge endpoint (up to 60 min warmup)..."
ok=0
for t in $(seq 1 1800); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$SERVE_PID" 2>/dev/null || { echo "ERROR: judge serve step died before publishing endpoint" >&2; exit 3; }
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
