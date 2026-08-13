#!/bin/bash
# Submit the judge capacity probe through the snapshot gateway.
#
# scripts/slurm/judge_capacity_smoke.sh is a JOB script; cluster_launch.sh runs its argument
# on the login node, so it needs this launcher to sbatch it. See launch_judge_pairs.sh for
# the same split.
#
# Usage:
#   scripts/cluster_launch.sh --dependency-profile eval \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-capacity \
#     --env CAP_MODEL=Qwen/Qwen3.5-9B \
#     scripts/launch_judge_capacity_smoke.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

CAP_MODEL=${CAP_MODEL:-Qwen/Qwen3.5-9B}
CAP_GPU_UTIL=${CAP_GPU_UTIL:-0.55}
CAP_LENS=${CAP_LENS:-"16384 20480 22528"}
DRY=${DRY:-0}

# Slurm splits --export on commas; CAP_LENS is space-separated, so it is safe here.
EXPORTS="ALL,CAP_MODEL=$CAP_MODEL,CAP_GPU_UTIL=$CAP_GPU_UTIL,CAP_LENS=$CAP_LENS"

echo "=== judge capacity smoke: model=$CAP_MODEL util=$CAP_GPU_UTIL lens='$CAP_LENS' ==="

if [ "$DRY" = "1" ]; then
  echo "[DRY] $SBATCH --parsable --export=$EXPORTS -- scripts/slurm/judge_capacity_smoke.sh"
  exit 0
fi

[ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
jid=$("$SBATCH" --parsable --export="$EXPORTS" -- scripts/slurm/judge_capacity_smoke.sh)
case "$jid" in
  ''|*[!0-9]*) echo "FATAL: sbatch failed (got '$jid')" >&2; exit 1 ;;
esac
echo "submitted capacity smoke job=$jid"
