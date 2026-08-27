#!/bin/bash
# Submit the judge discriminator's cross-entropy (CE) training run through the snapshot gateway.
#
# scripts/slurm/sft_variant.sh is a JOB script (#SBATCH headers, sources
# cluster_job_bootstrap.sh, needs SLURM_JOB_ID). cluster_launch.sh runs its positional
# argument on the login node, so it needs a LAUNCHER -- this file -- that submits the job
# script via snapshot_sbatch.sh. Handing sft_variant.sh straight to cluster_launch.sh fails
# immediately with "SLURM_JOB_ID: parameter null or not set".
#
# Usage (always through cluster_launch.sh, never sbatch directly):
#   scripts/cluster_launch.sh --dependency-profile sft \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-ce \
#     --env MODEL=qwen35-9b-judge \
#     --env DATA=<abs>/ce_train.jsonl \
#     --env OUT=<abs judge checkpoint dir> \
#     scripts/launch_judge_ce_train.sh
#
# Everything this launcher does is validation the job cannot do cheaply: sft_variant.sh
# resolves its configuration only after a node has been allocated, so a wrong MODEL or a
# missing DATA costs an allocation before it is noticed. The three checks below are the
# three deviations from the documented invocation that run to completion while measuring
# something other than what was asked for:
#
#   MODEL  - the judge aliases only. `${MODEL:-qwen3-8b}` inside sft_variant.sh treats empty
#            as unset, so an unresolved `--env MODEL=$SOMETHING_UNSET` would silently train a
#            GENERATOR under a judge run root. Empty is rejected here for that reason.
#   DATA   - the CE jsonl. Unset, sft_variant.sh falls back to the generator's SFT corpus.
#   OUT    - required. The per-VARIANT default bakes the generator's prism_full_s42 dataset
#            name into a judge checkpoint path.
#
# Optional: EPOCHS, MAX_TRAIN_EXAMPLES (forwarded only when set; sft_variant.sh validates
# them), VARIANT (default bf16_fsdp -- qlora_r64 would pass --force_qlora against a judge
# config that sets use_qlora: false, and the variant is not cross-checked against the model
# anywhere), DRY=1 to print the sbatch line instead of submitting.
#
# NOPACK is forced to 1 by the judge aliases inside sft_variant.sh and is deliberately not
# set here; --max_seq_length 8192 is hardcoded there too. This launcher only validates and
# submits.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

DRY=${DRY:-0}

MODEL=${MODEL:-}
case "$MODEL" in
  qwen35-4b-judge|qwen35-9b-judge) ;;
  "") echo "FATAL: MODEL is unset or empty; set qwen35-4b-judge or qwen35-9b-judge" >&2; exit 2 ;;
  *) echo "FATAL: MODEL must be a judge alias (qwen35-4b-judge|qwen35-9b-judge), got '$MODEL'" >&2
     exit 2 ;;
esac

DATA=${DATA:-}
[ -n "$DATA" ] || { echo "FATAL: DATA is unset or empty; point it at the CE training jsonl" >&2; exit 2; }
[ -f "$DATA" ] || { echo "FATAL: DATA does not exist: $DATA" >&2; exit 2; }

OUT=${OUT:-}
[ -n "$OUT" ] || { echo "FATAL: OUT is unset or empty; name the judge checkpoint dir" >&2; exit 2; }

VARIANT=${VARIANT:-bf16_fsdp}
case "$VARIANT" in
  qlora_r64|bf16_fsdp|bf16_fa2) ;;
  *) echo "FATAL: VARIANT must be qlora_r64|bf16_fsdp|bf16_fa2, got '$VARIANT'" >&2; exit 2 ;;
esac

# Slurm splits --export on commas, so every value here must be comma-free. Paths are.
EXPORTS="ALL,MODEL=$MODEL,VARIANT=$VARIANT,DATA=$DATA,OUT=$OUT"
# Unset = not forwarded at all, so sft_variant.sh keeps its "unset means the yaml decides"
# behaviour rather than receiving an empty string.
[ -n "${EPOCHS:-}" ] && EXPORTS="$EXPORTS,EPOCHS=$EPOCHS"
[ -n "${MAX_TRAIN_EXAMPLES:-}" ] && EXPORTS="$EXPORTS,MAX_TRAIN_EXAMPLES=$MAX_TRAIN_EXAMPLES"

echo "=== judge CE train: model=$MODEL variant=$VARIANT data=$DATA out=$OUT ==="

if [ "$DRY" = "1" ]; then
  echo "[DRY] $SBATCH --parsable --export=$EXPORTS -- scripts/slurm/sft_variant.sh"
  exit 0
fi

[ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
# The `--` script boundary is a repo convention enforced by
# tests/test_cluster_workflow.py::test_direct_snapshot_gateway_calls_include_script_boundary:
# without it sbatch can parse the script path as an option.
jid=$("$SBATCH" --parsable --export="$EXPORTS" -- scripts/slurm/sft_variant.sh)
case "$jid" in
  ''|*[!0-9]*) echo "FATAL: sbatch failed (got '$jid')" >&2; exit 1 ;;
esac
echo "submitted judge CE train job=$jid"
