#!/bin/bash
#SBATCH --job-name=wandb_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/wandb_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Dummy wandb job: prove we can login + init + log + finish a run against
# https://meta.wandb.io from a compute node. No GPU, no training, no model.
# Decision gate for whether GRPO smoke uses online wandb or offline mode.

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

if [ -f $TURING_RL_STATE_ROOT/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source $TURING_RL_STATE_ROOT/.env
  set +a
fi

# Per the cluster support thread shared by the user, the wandb endpoint for
# this V3 cluster is meta.wandb.io (NOT fairwandb.org).
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

echo "============================================"
echo "wandb smoke"
echo "Date:           $(date)"
echo "Host:           $(hostname)"
echo "WANDB_BASE_URL: $WANDB_BASE_URL"
echo "WANDB_API_KEY:  $([ -n "${WANDB_API_KEY:-}" ] && echo set || echo MISSING)"
echo "============================================"

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "ERROR: WANDB_API_KEY not set in .env or environment" >&2
  exit 2
fi

echo "=== reachability probe (curl -I) ==="
curl -sS -m 10 -o /dev/null -w "HTTP %{http_code} (time %{time_total}s)\n" "$WANDB_BASE_URL" || echo "curl probe failed"

echo ""
echo "=== wandb init + log + finish ==="
# wandb.init() picks up WANDB_API_KEY + WANDB_BASE_URL from env and handles
# `local-` prefixed self-hosted keys correctly via that path. We skip the
# CLI login + wandb.login(key=...) entirely.
$PY <<'PYEOF'
import os, random, time, sys

import wandb

print(f"wandb version: {wandb.__version__}", flush=True)
print(f"wandb base url: {os.environ.get('WANDB_BASE_URL')}", flush=True)

if not os.environ.get("WANDB_API_KEY"):
    print("ERROR: WANDB_API_KEY not in environment", flush=True)
    sys.exit(2)

try:
    run = wandb.init(
        project="turing-rl-smoke",
        name=f"dummy-{int(time.time())}",
        mode="online",
        tags=["smoke", "cluster-connectivity-test"],
        config={"purpose": "verify wandb reachability from compute node"},
    )
except Exception as exc:
    print(f"INIT FAILED: {type(exc).__name__}: {exc}", flush=True)
    sys.exit(3)
print(f"run created: id={run.id} url={run.url}", flush=True)

# Log a few dummy steps
for step in range(5):
    metrics = {
        "loss": 1.0 / (step + 1),
        "lr":   1e-4,
        "rand": random.random(),
        "step": step,
    }
    wandb.log(metrics, step=step)
    print(f"  step {step}: {metrics}", flush=True)
    time.sleep(0.5)

wandb.finish()
print("wandb OK", flush=True)
PYEOF
RC=$?

echo ""
echo "============================================"
echo "wandb smoke exit: $RC"
echo "Date:             $(date)"
echo "============================================"
exit $RC
