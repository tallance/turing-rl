#!/bin/bash
# Submit the Phase 0 judge format probe, one job per candidate judge, SERIALIZED.
#
# scripts/slurm/judge_format_probe.sh is a JOB script; cluster_launch.sh runs its argument on
# the login node, so it needs this launcher to sbatch it. See launch_judge_pairs.sh.
#
# Each probe serves one judge across a whole node (TP=1, DP=8 replicas) and scores the same
# pairs in three decoding regimes. Jobs are chained with `afterany` so at most ONE node is
# held at a time: three 8-GPU jobs at once would take the whole 24-GPU QOS allowance and
# starve everything else. `afterany`, not `afterok`, so one judge failing to serve does not
# silently cancel the rest of the sweep.
#
# Usage:
#   scripts/cluster_launch.sh --dependency-profile eval \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-format-probe \
#     --env PROBE_MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B" \
#     --env OUT_ROOT=/home/lancewicki/projects/turing-rl/results/<date>-judge-format-probe \
#     scripts/launch_judge_format_probe.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

PROBE_MODELS=${PROBE_MODELS:-"Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"}
# See judge_format_probe.sh: $REPO/data is the source snapshot, not the state root.
PAIRS=${PAIRS:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1/val.parquet}
OUT_ROOT=${OUT_ROOT:-$REPO/results/judge-format-probe}
PROBE_LIMIT=${PROBE_LIMIT:-200}
DRY=${DRY:-0}

echo "=== phase 0 format probe: models='$PROBE_MODELS' pairs=$PAIRS limit=$PROBE_LIMIT ==="

PREV=""
for model in $PROBE_MODELS; do
  tag=$(echo "$model" | tr '/' '-')
  out_json="$OUT_ROOT/$tag.json"
  dump_csv="$OUT_ROOT/$tag.eval.csv"
  # Comma-free values only: Slurm splits --export on commas.
  EXPORTS="ALL,JUDGE_MODEL=$model,PAIRS=$PAIRS,OUT_JSON=$out_json,DUMP_CSV=$dump_csv,LIMIT=$PROBE_LIMIT"

  dep=""; [ -n "$PREV" ] && dep="--dependency=afterany:$PREV"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] $SBATCH --parsable $dep --export=$EXPORTS -- scripts/slurm/judge_format_probe.sh"
    PREV="dry$RANDOM"; continue
  fi

  [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
  # shellcheck disable=SC2086
  jid=$("$SBATCH" --parsable $dep --export="$EXPORTS" -- scripts/slurm/judge_format_probe.sh)
  case "$jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $model (got '$jid')" >&2; exit 1 ;;
  esac
  echo "submitted $model job=$jid${PREV:+ (after $PREV)}"
  PREV=$jid
done
