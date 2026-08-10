#!/bin/bash
#SBATCH --job-name=grpo_smoke_8b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/grpo_smoke_8b-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# GRPO smoke variant pointed at an 8B frozen judge instead of the 397B one.
# This is the First Experiment of the adversarial user-simulator proposal:
# does a frozen small judge get reward-hacked (fake beats real >50%)?
#
# Kept as a duplicate of grpo_smoke.sh — same code path, only JUDGE_MODEL and
# a few isolation knobs (wandb project, dump dir, checkpoint dir) differ. All
# concurrency/timeout/completion-token settings are IDENTICAL to the 397B run
# on purpose: fidelity first; tune only after we've seen 8B behavior.
#
# Expected env from scripts/launch_grpo_smoke_8b.sh:
#   JUDGE_HOST   - hostname of the running judge_serve_8b node
#   JUDGE_PORT   - port (default 8000)

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -z "${JUDGE_HOST:-}" ]; then
  echo "ERROR: JUDGE_HOST not set. Run via scripts/launch_grpo_smoke_8b.sh." >&2
  exit 2
fi
JUDGE_PORT="${JUDGE_PORT:-8123}"

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

# === Judge wiring (env vars consumed by shared/api_client.py) ===
export OPENAI_API_BASE="http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
export OPENAI_API_KEY="dummy-self-hosted"
export JUDGE_MODEL="Qwen/Qwen3-8B"
unset OPENROUTER_API_KEY OPENROUTER_PROVIDER_ORDER
export PERSONA_OPENAI_TIMEOUT_SECONDS=1200
export PERSONA_OPENAI_MAX_RETRIES=3
# Same concurrency cap as 397B for a controlled comparison. Small judge should
# handle more, but we're testing hackability, not throughput — revisit later.
export TURING_JUDGE_MAX_CONCURRENCY=4
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192

# Force the judge request to use the full prompt-matched JSON schema instead
# of unconstrained json_object. See reward.py:_resolve_response_format.
export PERSONA_JUDGE_JSON_SCHEMA=1

# Dump every judge call for the smoke. Separate dir from the 397B session so
# the dump viewer for each stays clean.
export PERSONA_JUDGE_DUMP_RATE=1.0
export PERSONA_JUDGE_DUMP_DIR=/home/lancewicki/tmp/judge_dumps_8b

# === Wandb ===
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-smoke-8b-judge
export WANDB_RUN_GROUP=grpo-smoke-8b-judge

# === Ray / runtime ===
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PERSONA_RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-96}"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

export SFT_ADAPTER_PATH="$REPO/results/sft/smoke/final"
export CHECKPOINT_DIR="$REPO/results/grpo/checkpoints_smoke_8bj_${SLURM_JOB_ID}"
export EXPERIMENT_NAME="grpo-smoke-8bj-${SLURM_JOB_ID}"
export PYTHON="$PY"

cd "$REPO"

echo "============================================"
echo "GRPO smoke wrapper (8B judge)"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JUDGE URL:   $OPENAI_API_BASE"
echo "JUDGE_MODEL: $JUDGE_MODEL"
echo "Wandb URL:   ${WANDB_BASE_URL:-<unset>}  project=$WANDB_PROJECT"
echo "SFT adapter: $SFT_ADAPTER_PATH"
echo "Checkpoint:  $CHECKPOINT_DIR"
echo "Dump dir:    $PERSONA_JUDGE_DUMP_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

echo "=== judge reachability probe ==="
if ! curl -sf --max-time 10 "$OPENAI_API_BASE/models" -o /tmp/judge_probe_8b.json; then
  echo "ERROR: judge /v1/models unreachable from $(hostname) -> $JUDGE_HOST:$JUDGE_PORT" >&2
  exit 3
fi
$PY -c "
import json
d = json.load(open('/tmp/judge_probe_8b.json'))
models = [m.get('id') for m in d.get('data', [])]
print(f'judge advertises models: {models}')
"

echo ""
echo "=== invoking bash_scripts/grpo/train_grpo_smoke.sh ==="
bash "$REPO/bash_scripts/grpo/train_grpo_smoke.sh" turing prism history qwen3-8b
RC=$?

echo ""
echo "============================================"
echo "GRPO smoke (8B judge) exit: $RC"
echo "Date:            $(date)"
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== checkpoint dir contents ==="
  ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "(no checkpoint dir at $CHECKPOINT_DIR)"
fi
echo "============================================"
exit $RC
