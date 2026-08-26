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
#
# --env PROMPT_STYLE=<full|single_token> selects the judge prompt template baked into the
# pair rows: "full" (default, rubric and JSON schema) or one-letter. It reaches
# build_judge_train_pairs.py --prompt-style and is recorded in the sibling .meta.json.
#
# The style is folded into the default OUT_DIR as a nested segment, and a single_token
# OUT_DIR must name the style:
#     full          -> $TURING_RL_GENERATED_DATA_ROOT/prism/judge/iter1
#     single_token  -> $TURING_RL_GENERATED_DATA_ROOT/prism/judge/iter1/single_token
# The filename does not carry the style, so two styles built into one OUT_DIR overwrite each
# other -- and they overwrite the .meta.json alongside, which is the only thing recording
# which style a parquet holds. The overwrite therefore destroys its own evidence. The guard
# fires on the path the caller typed rather than silently redirecting, because the mistake
# being caught is reusing the iter1 path out of habit.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

SPLITS=${SPLITS:-"train val"}
DRY=${DRY:-0}

# Validated before OUT_DIR because the default path depends on the style.
PROMPT_STYLE=${PROMPT_STYLE:-full}
case "$PROMPT_STYLE" in
  full|single_token) ;;
  *) echo "FATAL: PROMPT_STYLE must be full|single_token, got '$PROMPT_STYLE'" >&2; exit 2 ;;
esac

# $REPO/data is the immutable source snapshot inside a job; generated data belongs in the
# state root, which is what TURING_RL_GENERATED_DATA_ROOT points at. Resolved lazily: a
# caller who passes OUT_DIR must not also need TURING_RL_GENERATED_DATA_ROOT set.
if [ -z "${OUT_DIR:-}" ]; then
  OUT_DIR=${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1
  # Nested for non-default styles only, matching launch_judge_eval_matrix.sh's reward_dir.
  [ "$PROMPT_STYLE" = "full" ] || OUT_DIR=$OUT_DIR/$PROMPT_STYLE
fi

if [ "$PROMPT_STYLE" = "single_token" ]; then
  case "$OUT_DIR" in
    *single_token*|*single-token*) ;;
    *) echo "FATAL: a single_token OUT_DIR must name the style: $OUT_DIR" >&2; exit 2 ;;
  esac
fi

echo "=== judge pair build: splits='$SPLITS' style=$PROMPT_STYLE out_dir=$OUT_DIR ==="

for split in $SPLITS; do
  case "$split" in
    train|val) ;;
    *) echo "FATAL: SPLITS entries must be train or val, got '$split'" >&2; exit 2 ;;
  esac

  # Slurm splits --export on commas, so every value here must be comma-free. Paths are.
  EXPORTS="ALL,SPLIT=$split,OUT_DIR=$OUT_DIR,PROMPT_STYLE=$PROMPT_STYLE"

  if [ "$DRY" = "1" ]; then
    echo "[DRY] $SBATCH --parsable --export=$EXPORTS -- scripts/slurm/judge_train_gen.sh"
    continue
  fi

  [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
  # The `--` script boundary is a repo convention enforced by
  # tests/test_cluster_workflow.py::test_direct_snapshot_gateway_calls_include_script_boundary:
  # without it sbatch can parse the script path as an option.
  jid=$("$SBATCH" --parsable --export="$EXPORTS" -- scripts/slurm/judge_train_gen.sh)
  case "$jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for split=$split (got '$jid')" >&2; exit 1 ;;
  esac
  echo "submitted split=$split job=$jid"
done
