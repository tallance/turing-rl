#!/bin/bash
# Judge-only rerun of the full five-epoch evaluation with the corrected schema.
# Cells are strictly model-major. Batches are continued by a tiny dependent CPU
# job so no more than ten jobs (current controller + 8 cells + next controller)
# are present at once.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${REPO:-/home/lancewicki/projects/turing-rl}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
DATA_ROOT=${TURING_RL_DATA_ROOT:-$REPO/data}
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
EVAL_ROOT=${EVAL_ROOT:-$REPO/results/2026-08-10-test-eval-9b-full5ep-full-schema}
EVAL_PARQUET=${EVAL_PARQUET:-$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
SLURM_SCRIPT=${SLURM_SCRIPT:-$REPO/scripts/slurm/judge_sweep_cell.sh}
CONTINUE_SCRIPT=${CONTINUE_SCRIPT:-$REPO/scripts/slurm/continue_full_schema_eval.sh}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-full5ep-step}
PAIRS_TAG=${PAIRS_TAG:-880}
OFFSET=${OFFSET:-0}
BATCH_SIZE=${BATCH_SIZE:-8}
CHAIN_AFTER=${CHAIN_AFTER:-}
DRY=${DRY:-0}
SKIP_SPLIT_GUARD=${SKIP_SPLIT_GUARD:-0}

STEPS=(0 32 64 96 128 160 192 224 256 288 320)
JUDGES=(qwen35-9b gemma4-12b gemma4-31b qwen35-4b qwen35-27b)
TOTAL=$((${#STEPS[@]} * ${#JUDGES[@]}))

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
export REPO EVAL_ROOT EVAL_PARQUET SLURM_SCRIPT CONTINUE_SCRIPT GEN_KEY_PREFIX PAIRS_TAG BATCH_SIZE

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
  judge_index=$((idx / ${#STEPS[@]}))
  step_index=$((idx % ${#STEPS[@]}))
  judge=${JUDGES[$judge_index]}
  step=${STEPS[$step_index]}
  gen_key=${GEN_KEY_PREFIX}${step}
  pairs=$EVAL_ROOT/raw/pairs/gen_${gen_key}_${PAIRS_TAG}.parquet
  [ -f "$pairs" ] || { echo "FATAL: missing pair set: $pairs" >&2; exit 2; }

  read -r model tp replicas concurrency <<EOF
$($PY -c "from configs.judge_sweep_cells import resolve_cell; c=resolve_cell('$judge'); print(c['model_id'], c['tp'], c['replicas'], c.get('concurrency', 32))")
EOF
  [ -n "${model:-}" ] && [ -n "${tp:-}" ] && [ -n "${replicas:-}" ] || {
    echo "FATAL: could not resolve judge cell $judge" >&2
    exit 2
  }

  sweep_root=$EVAL_ROOT/raw/$gen_key/sweep
  reward_dir=$sweep_root/$judge/on/reward
  if [ -d "$reward_dir" ] && [ -n "$(ls -A "$reward_dir" 2>/dev/null)" ]; then
    echo "FATAL: refusing stale output in $reward_dir" >&2
    exit 2
  fi

  dep=""
  [ -n "$PREV" ] && dep="afterok:$PREV"
  gpus=$((tp * replicas))
  jid=$(submit "$dep" --gres=gpu:$gpus --job-name="te_${judge}_${step}" \
    --export=ALL,MODEL=$model,TP=$tp,REPLICAS=$replicas,CONCURRENCY=$concurrency,THINKING_MODE=on,CELL_NAME=$judge,PAIRS=$pairs,SWEEP_ROOT=$sweep_root \
    "$SLURM_SCRIPT")
  need_jid "$jid" "$judge/step$step"
  PREV="$jid"
  echo "submitted [$idx/$((TOTAL-1))] $judge step$step -> $jid (gpu:$gpus concurrency:$concurrency)" >&2
done

if [ "$END" -lt "$TOTAL" ]; then
  dep="afterok:$PREV"
  next=$(submit "$dep" --gres=gpu:0 --job-name=te_eval_continue \
    --export=ALL,NEXT_OFFSET=$END "$CONTINUE_SCRIPT")
  need_jid "$next" "continuation at offset $END"
  PREV="$next"
  echo "scheduled continuation offset=$END -> $next" >&2
fi

echo "chain tail job: $PREV"
echo "submitted indices [$OFFSET,$END) of $TOTAL"
