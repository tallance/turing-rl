#!/bin/bash
# Submit the Phase 0 judge format probe: one SERVE job + one PROBE job per candidate judge.
#
# Why two jobs per judge rather than one that backgrounds the server:
# cluster_job_bootstrap.sh derives its runtime work directory from job-$SLURM_JOB_ID, so a
# job that sources it and then invokes another bootstrap-sourcing script collides on mkdir
# and dies with "runtime work directory already exists" (jobs 15951-15953). Giving the server
# its own job id gives it its own work dir. It also means the probe job needs no GPU -- it is
# an HTTP client -- so only the server holds the node.
#
# The probe starts with `--dependency=after:<serve>` (after the server has STARTED, not
# finished), polls the shared endpoint file, and scancels the server when it is done. Judges
# are chained with `afterany` on the previous probe so at most one node is held at a time.
#
# Usage:
#   scripts/cluster_launch.sh --dependency-profile eval \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-format-probe \
#     --env OUT_ROOT=/home/lancewicki/projects/turing-rl/results/<date>-judge-format-probe \
#     --env PROBE_MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B" \
#     scripts/launch_judge_format_probe.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

PROBE_MODELS=${PROBE_MODELS:-"Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"}
# $REPO/data is the immutable source snapshot inside a job; generated data lives in the state
# root, which is what TURING_RL_GENERATED_DATA_ROOT points at.
PAIRS=${PAIRS:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1/val.parquet}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT (shared, must be readable by both jobs)}
PROBE_LIMIT=${PROBE_LIMIT:-200}
DRY=${DRY:-0}

mkdir -p "$OUT_ROOT"
echo "=== phase 0 format probe: models='$PROBE_MODELS' pairs=$PAIRS limit=$PROBE_LIMIT ==="

PREV_PROBE=""
for model in $PROBE_MODELS; do
  tag=$(echo "$model" | tr '/' '-')
  endpoint_file="$OUT_ROOT/$tag.endpoint.txt"
  rm -f "$endpoint_file"

  # Serve job. Chained behind the PREVIOUS PROBE so only one node is held at a time; three
  # 8-GPU servers at once would take the whole 24-GPU QOS allowance.
  serve_dep=""; [ -n "$PREV_PROBE" ] && serve_dep="--dependency=afterany:$PREV_PROBE"
  serve_exports="ALL,MODEL=$model,JUDGE_ENDPOINT_FILE=$endpoint_file,TP=1,DP=8"

  if [ "$DRY" = "1" ]; then
    echo "[DRY] serve: $SBATCH --parsable $serve_dep --export=$serve_exports -- scripts/slurm/judge_serve_9b_replicas.sh"
    echo "[DRY] probe: after:<serve> for $model"
    PREV_PROBE="dry$RANDOM"; continue
  fi

  [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
  # shellcheck disable=SC2086
  serve_jid=$("$SBATCH" --parsable $serve_dep --export="$serve_exports" -- scripts/slurm/judge_serve_9b_replicas.sh)
  case "$serve_jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for serve $model (got '$serve_jid')" >&2; exit 1 ;;
  esac

  # Probe job: starts once the server has STARTED (after:, not afterok:), then polls the
  # endpoint file and scancels the server on exit.
  probe_exports="ALL,JUDGE_MODEL=$model,PAIRS=$PAIRS,LIMIT=$PROBE_LIMIT"
  probe_exports="$probe_exports,OUT_JSON=$OUT_ROOT/$tag.json,DUMP_CSV=$OUT_ROOT/$tag.eval.csv"
  probe_exports="$probe_exports,JUDGE_ENDPOINT_FILE=$endpoint_file,RL_JUDGE_JOB_ID=$serve_jid"
  probe_jid=$("$SBATCH" --parsable --dependency=after:"$serve_jid" --export="$probe_exports" -- scripts/slurm/judge_format_probe.sh)
  case "$probe_jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for probe $model (got '$probe_jid'); cancelling serve $serve_jid" >&2
                 scancel "$serve_jid" 2>/dev/null; exit 1 ;;
  esac

  echo "submitted $model: serve=$serve_jid probe=$probe_jid${PREV_PROBE:+ (chained after $PREV_PROBE)}"
  PREV_PROBE=$probe_jid
done
