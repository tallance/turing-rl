#!/bin/bash
# Judge-only rerun of the full five-epoch evaluation with the corrected schema.
# Cells are strictly model-major. Batches are continued by a tiny dependent CPU
# job so no more than ten jobs (current controller + 8 cells + next controller)
# are present at once.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${REPO:-/home/lancewicki/projects/turing-rl}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
DATA_ROOT=${TURING_RL_INPUT_DATA_ROOT:-$REPO/data}
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
EVAL_ROOT=${EVAL_ROOT:-$REPO/results/2026-08-10-test-eval-9b-full5ep-full-schema}
EVAL_PARQUET=${EVAL_PARQUET:-$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
SLURM_SCRIPT=${SLURM_SCRIPT:-$REPO/scripts/slurm/judge_sweep_cell.sh}
CONTINUE_SCRIPT=${CONTINUE_SCRIPT:-$REPO/scripts/slurm/continue_full_schema_eval.sh}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-full5ep-step}
PAIRS_TAG=${PAIRS_TAG:-880}
JOB_PREFIX=${JOB_PREFIX:-te}
OFFSET=${OFFSET:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
CHAIN_AFTER=${CHAIN_AFTER:-}
DRY=${DRY:-0}
SKIP_SPLIT_GUARD=${SKIP_SPLIT_GUARD:-0}

STEPS=${STEPS:-"0 32 64 96 128 160 192 224 256 288 320"}
JUDGES=${JUDGES:-"qwen35-9b gemma4-12b gemma4-31b qwen35-4b qwen35-27b"}
JUDGE_MODES=${JUDGE_MODES:-}
REUSED_STEP0_CELLS=${REUSED_STEP0_CELLS:-}
read -r -a STEP_VALUES <<< "$STEPS"
read -r -a JUDGE_VALUES <<< "$JUDGES"
[ "${#STEP_VALUES[@]}" -gt 0 ] && [ "${#JUDGE_VALUES[@]}" -gt 0 ] || {
  echo "FATAL: STEPS and JUDGES must each contain at least one value" >&2
  exit 2
}
if [ -n "$JUDGE_MODES" ]; then
  read -r -a JUDGE_MODE_VALUES <<< "$JUDGE_MODES"
  [ "${#JUDGE_MODE_VALUES[@]}" -eq "${#JUDGE_VALUES[@]}" ] || {
    echo "FATAL: JUDGE_MODES must contain one mode per JUDGES entry" >&2
    exit 2
  }
else
  JUDGE_MODE_VALUES=()
  for _judge in "${JUDGE_VALUES[@]}"; do JUDGE_MODE_VALUES+=(on); done
fi
for mode in "${JUDGE_MODE_VALUES[@]}"; do
  case "$mode" in on|off) ;; *) echo "FATAL: judge modes must be on|off (got '$mode')" >&2; exit 2 ;; esac
done
TOTAL=$((${#STEP_VALUES[@]} * ${#JUDGE_VALUES[@]}))

case "$OFFSET:$BATCH_SIZE" in
  *[!0-9:]*|:*|*:) echo "FATAL: OFFSET and BATCH_SIZE must be non-negative integers" >&2; exit 2 ;;
esac
[ "$BATCH_SIZE" -ge 1 ] && [ "$BATCH_SIZE" -le 8 ] || {
  echo "FATAL: BATCH_SIZE must be in [1,8] to preserve the queue bound" >&2
  exit 2
}
[ "$OFFSET" -lt "$TOTAL" ] || { echo "evaluation already fully submitted (offset=$OFFSET)"; exit 0; }

cd "$REPO" || exit 2
mkdir -p "$EVAL_ROOT/raw/pairs" "$REPO/logs"

if [ "$SKIP_SPLIT_GUARD" = "1" ]; then
  [ "$DRY" = "1" ] || { echo "FATAL: SKIP_SPLIT_GUARD is allowed only with DRY=1" >&2; exit 2; }
else
  "$PY" scripts/check_eval_split.py \
    --eval_parquet "$EVAL_PARQUET" --expect heldout \
    --out_json "$EVAL_ROOT/split_guard.json" || exit 2
fi

export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_OPENAI_TIMEOUT_SECONDS=1800
export REPO EVAL_ROOT EVAL_PARQUET SLURM_SCRIPT CONTINUE_SCRIPT GEN_KEY_PREFIX PAIRS_TAG
export BATCH_SIZE STEPS JUDGES JUDGE_MODES REUSED_STEP0_CELLS JOB_PREFIX

submit() {
  local dep="$1"; shift
  local deparg=""
  [ -n "$dep" ] && deparg="--dependency=$dep"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $deparg $*" >&2
    echo "dry$RANDOM"
  else
    [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; return 2; }
    "$SBATCH" --parsable $deparg "$@"
  fi
}

need_jid() {
  if [ "$DRY" = "1" ]; then [ -n "$1" ] && return 0; fi
  case "$1" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $2 (got '$1')" >&2; exit 1 ;;
  esac
}

END=$((OFFSET + BATCH_SIZE))
[ "$END" -gt "$TOTAL" ] && END=$TOTAL
PREV="$CHAIN_AFTER"

for ((idx=OFFSET; idx<END; idx++)); do
  judge_index=$((idx / ${#STEP_VALUES[@]}))
  step_index=$((idx % ${#STEP_VALUES[@]}))
  judge=${JUDGE_VALUES[$judge_index]}
  mode=${JUDGE_MODE_VALUES[$judge_index]}
  step=${STEP_VALUES[$step_index]}
  gen_key=${GEN_KEY_PREFIX}${step}
  pairs=$EVAL_ROOT/raw/pairs/gen_${gen_key}_${PAIRS_TAG}.parquet
  [ -f "$pairs" ] || { echo "FATAL: missing pair set: $pairs" >&2; exit 2; }

  sweep_root=$EVAL_ROOT/raw/$gen_key/sweep
  reward_dir=$sweep_root/$judge/$mode/reward
  if [ -d "$reward_dir" ] && [ -n "$(ls -A "$reward_dir" 2>/dev/null)" ]; then
    reuse=0
    if [ "$step" = "0" ]; then
      for reused_cell in $REUSED_STEP0_CELLS; do
        [ "$judge" = "$reused_cell" ] && reuse=1
      done
    fi
    if [ "$reuse" = "1" ]; then
      manifest=$EVAL_ROOT/provenance/step0_reuse.json
      [ -f "$manifest" ] || { echo "FATAL: missing step-0 reuse manifest: $manifest" >&2; exit 2; }
      if [ "$DRY" != "1" ]; then
        "$PY" scripts/verify_judge_completeness.py \
          --reward_dir "$reward_dir" --pairs "$pairs" --expect_pairs "$PAIRS_TAG" || exit 2
      fi
      echo "reusing verified step-0 cell $judge/$mode -> $reward_dir" >&2
      continue
    fi
    echo "FATAL: refusing stale output in $reward_dir" >&2
    exit 2
  fi

  read -r model tp replicas concurrency <<EOF
$($PY -c "from configs.judge_sweep_cells import resolve_cell; c=resolve_cell('$judge'); print(c['model_id'], c['tp'], c['replicas'], c.get('concurrency', 32))")
EOF
  [ -n "${model:-}" ] && [ -n "${tp:-}" ] && [ -n "${replicas:-}" ] || {
    echo "FATAL: could not resolve judge cell $judge" >&2
    exit 2
  }

  dep=""
  [ -n "$PREV" ] && dep="afterok:$PREV"
  gpus=$((tp * replicas))
  jid=$(submit "$dep" --gres=gpu:$gpus --job-name="${JOB_PREFIX}_${judge}_${step}" \
    --export=ALL,MODEL=$model,TP=$tp,REPLICAS=$replicas,CONCURRENCY=$concurrency,THINKING_MODE=$mode,CELL_NAME=$judge,PAIRS=$pairs,SWEEP_ROOT=$sweep_root \
    -- \
    "$SLURM_SCRIPT")
  need_jid "$jid" "$judge/step$step"
  PREV="$jid"
  echo "submitted [$idx/$((TOTAL-1))] $judge step$step -> $jid (gpu:$gpus concurrency:$concurrency)" >&2
done

if [ "$END" -lt "$TOTAL" ]; then
  dep=""
  if [ -n "$PREV" ]; then
    dep="afterok:$PREV"
  elif [ -n "${SLURM_JOB_ID:-}" ]; then
    # A batch can contain only a verified reused cell, so it submits no new
    # judge job and leaves PREV empty. Chain the next controller after this
    # controller rather than emitting the invalid dependency "afterok:".
    dep="afterok:$SLURM_JOB_ID"
  fi
  next=$(submit "$dep" --gres=gpu:0 --job-name="${JOB_PREFIX}_continue" \
    --export=ALL,NEXT_OFFSET=$END -- "$CONTINUE_SCRIPT")
  need_jid "$next" "continuation at offset $END"
  PREV="$next"
  echo "scheduled continuation offset=$END -> $next" >&2
fi

echo "chain tail job: $PREV"
echo "submitted indices [$OFFSET,$END) of $TOTAL"
