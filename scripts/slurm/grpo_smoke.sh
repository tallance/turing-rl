#!/bin/bash
#SBATCH --job-name=grpo_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/grpo_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Thin sbatch wrapper around bash_scripts/grpo/train_grpo_smoke.sh (which itself
# is a minimal-diff variant of upstream train_grpo.sh). We only handle:
#   - slurm/env plumbing (proxy unset, .env sourcing, wandb+judge env exports)
#   - GPU reachability probe to the judge before verl init
# All GRPO/verl invocation lives in the bash_scripts/ launcher for parity.
#
# Expected env from scripts/launch_grpo_smoke.sh:
#   JUDGE_HOST   - hostname of the running judge_serve sbatch node
#   JUDGE_PORT   - port (default 8000)

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -z "${JUDGE_HOST:-}" ]; then
  echo "ERROR: JUDGE_HOST not set. Run via scripts/launch_grpo_smoke.sh." >&2
  exit 2
fi
JUDGE_PORT="${JUDGE_PORT:-8000}"

REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

# Sourcing .env: HF_TOKEN, WANDB_API_KEY, WANDB_BASE_URL. Judge URL is overridden below.
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

# === Judge wiring (env vars consumed by shared/api_client.py) ===
export OPENAI_API_BASE="http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
export OPENAI_API_KEY="dummy-self-hosted"
export JUDGE_MODEL="Qwen/Qwen3.5-397B-A17B-GPTQ-Int4"
unset OPENROUTER_API_KEY OPENROUTER_PROVIDER_ORDER
export PERSONA_OPENAI_TIMEOUT_SECONDS=600
export PERSONA_OPENAI_MAX_RETRIES=3
# Judge reply is a small JSON, but --reasoning-parser deepseek_r1 on the judge
# enables extended thinking: model burns most of the budget inside <think> before
# emitting JSON. 8192 matches the paper's default (shared/judge_utils.py:388,
# reward.py:357). Fits within judge --max-model-len=16384 even for our ~9k prompts.
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192

# Dump every judge call for the smoke (~260 calls * ~15KB = ~4MB). See
# shared/api_client.py _dump_judge_response. Real runs should drop this to 0.01
# or unset it entirely.
export PERSONA_JUDGE_DUMP_RATE=1.0
export PERSONA_JUDGE_DUMP_DIR=/home/lancewicki/tmp/judge_dumps

# === Wandb ===
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-smoke
export WANDB_RUN_GROUP=grpo-smoke

# === Ray / runtime ===
export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PERSONA_RAY_NUM_CPUS="${SLURM_CPUS_PER_TASK:-96}"
# Ray worker subprocesses inherit env vars but not cwd/sys.path; explicit PYTHONPATH
# is required so `training.grpo.ray_worker_setup` resolves inside spawned actors.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

# Point their SFT_ADAPTER_PATH default at our smoke adapter.
export SFT_ADAPTER_PATH="$REPO/results/sft/smoke/final"
export CHECKPOINT_DIR="$REPO/results/grpo/checkpoints_smoke_${SLURM_JOB_ID}"
export EXPERIMENT_NAME="grpo-smoke-${SLURM_JOB_ID}"
export PYTHON="$PY"

cd "$REPO"

echo "============================================"
echo "GRPO smoke wrapper"
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JUDGE URL:   $OPENAI_API_BASE"
echo "JUDGE_MODEL: $JUDGE_MODEL"
echo "Wandb URL:   ${WANDB_BASE_URL:-<unset>}  project=$WANDB_PROJECT"
echo "SFT adapter: $SFT_ADAPTER_PATH"
echo "Checkpoint:  $CHECKPOINT_DIR"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "============================================"

echo "=== judge reachability probe ==="
if ! curl -sf --max-time 10 "$OPENAI_API_BASE/models" -o /tmp/judge_probe.json; then
  echo "ERROR: judge /v1/models unreachable from $(hostname) -> $JUDGE_HOST:$JUDGE_PORT" >&2
  exit 3
fi
$PY -c "
import json
d = json.load(open('/tmp/judge_probe.json'))
models = [m.get('id') for m in d.get('data', [])]
print(f'judge advertises models: {models}')
"

echo ""
echo "=== invoking bash_scripts/grpo/train_grpo_smoke.sh ==="
bash "$REPO/bash_scripts/grpo/train_grpo_smoke.sh" turing prism history qwen3-8b
RC=$?

echo ""
echo "============================================"
echo "GRPO smoke exit: $RC"
echo "Date:            $(date)"
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== checkpoint dir contents ==="
  ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "(no checkpoint dir at $CHECKPOINT_DIR)"
fi
echo "============================================"
exit $RC
