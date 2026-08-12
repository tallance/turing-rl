#!/bin/bash
# Submit the judge pair-build job(s) through the snapshot gateway.
#
# scripts/slurm/judge_train_gen.sh is a JOB script (#SBATCH headers, sources
# cluster_job_bootstrap.sh, needs SLURM_JOB_ID). cluster_launch.sh runs its positional
# argument on the login node, so it needs a LAUNCHER — this file — that submits the job
# script via snapshot_sbatch.sh. Handing the job script straight to cluster_launch.sh
# fails immediately with "SLURM_JOB_ID: parameter null or not set".
#
# Usage (always through cluster_launch.sh, never sbatch directly):
#   scripts/cluster_launch.sh --dependency-profile data \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-pairs-iter1 \
#     --env OUT_DIR=/home/lancewicki/projects/turing-rl/data/prism/judge/iter1 \
#     --env SPLITS=train \
#     scripts/launch_judge_pairs.sh
#
# SPLITS selects which splits to build (default "train val"). The train split takes the
# [0.0,0.1) hash slice capped at 416 contexts with k=4 generations; the val split takes all
# 352 contexts at k=1. Those per-split parameters live in judge_train_gen.sh, not here.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

SPLITS=${SPLITS:-"train val"}
DRY=${DRY:-0}
OUT_DIR=${OUT_DIR:-$REPO/data/prism/judge/iter1}

echo "=== judge pair build: splits='$SPLITS' out_dir=$OUT_DIR ==="

for split in $SPLITS; do
  case "$split" in
    train|val) ;;
    *) echo "FATAL: SPLITS entries must be train or val, got '$split'" >&2; exit 2 ;;
  esac

  # Slurm splits --export on commas, so every value here must be comma-free. Paths are.
  EXPORTS="ALL,SPLIT=$split,OUT_DIR=$OUT_DIR"

  if [ "$DRY" = "1" ]; then
    echo "[DRY] $SBATCH --parsable --export=$EXPORTS scripts/slurm/judge_train_gen.sh"
    continue
  fi

  [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
  jid=$("$SBATCH" --parsable --export="$EXPORTS" scripts/slurm/judge_train_gen.sh)
  case "$jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for split=$split (got '$jid')" >&2; exit 1 ;;
  esac
  echo "submitted split=$split job=$jid"
done
