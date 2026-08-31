#!/bin/bash
# Score an existing generator trajectory with single-token judges.
#
# Re-scores pair sets that already exist: the generated turns were produced once for the
# full-schema evaluation, so nothing is regenerated here and the generator side stays
# byte-identical to the full-schema curves it will be read against.
#
# The pair sets for one trajectory can live in more than one result root (a run that was
# extended writes its new steps to a second root), so SOURCE_ROOTS is a list and each step
# is resolved against all of them.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
CODE_ROOT=${TURING_RL_CODE_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=$CODE_ROOT/scripts/snapshot_sbatch.sh
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
RESULTS=${RESULTS:-/home/lancewicki/projects/turing-rl/results}
SOURCE_ROOTS=${SOURCE_ROOTS:-"$RESULTS/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema $RESULTS/2026-08-19-test-eval-9b-train10pct-20ep-every2ep-test50pct-full-schema"}
# Every 2 epochs of the 20-epoch run (6 GRPO steps per epoch).
STEPS=${STEPS:-"0 12 24 36 48 60 72 84 96 108 120"}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-train10pct-step}
PAIRS_TAG=${PAIRS_TAG:-440}
JOB_PREFIX=${JOB_PREFIX:-st_traj}
TP=${TP:-1}
REPLICAS=${REPLICAS:-8}
CONCURRENCY=${CONCURRENCY:-32}
DRY=${DRY:-0}

# One independent chain per judge, "cell|model" separated by ';'. Independent on purpose:
# chaining the judges together would let one judge's failure strand the other two.
JUDGES=${JUDGES:-"qwen35-9b-st|Qwen/Qwen3.5-9B;gemma4-12b-st|google/gemma-4-12B-it;judge-9b-ce-st|/home/lancewicki/projects/turing-rl/checkpoints/sft/judge_qwen35_9b_ce_dense"}

# Fixed, not configurable: this launcher exists to run the single-token protocol, and
# judge_sweep_cell.sh rejects single_token with thinking on. Passing them explicitly rather
# than relying on the cell default (which is "full") keeps a silent protocol swap impossible.
THINKING_MODE=off
JUDGE_PROMPT_STYLE=single_token

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

read -r -a step_values <<< "$STEPS"
[ "${#step_values[@]}" -gt 0 ] || { echo "FATAL: STEPS is empty" >&2; exit 2; }
for step in "${step_values[@]}"; do
  case "$step" in ''|*[!0-9]*) echo "FATAL: invalid step '$step'" >&2; exit 2 ;; esac
done

# Resolve one step to exactly one source parquet. A step that no root has is fatal; a step
# two roots both have is fatal unless they are byte-identical -- "first root wins" would
# quietly score the wrong checkpoint and still draw a plausible curve.
resolve_pairs() {
  local step=$1 found="" found_digest="" root candidate candidate_digest
  for root in $SOURCE_ROOTS; do
    candidate=$root/raw/pairs/gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet
    [ -f "$candidate" ] || continue
    candidate_digest=$(sha256_of "$candidate")
    if [ -z "$found" ]; then
      found=$candidate
      found_digest=$candidate_digest
    elif [ "$candidate_digest" != "$found_digest" ]; then
      echo "FATAL: step $step exists in two source roots with different content:" >&2
      echo "  $found ($found_digest)" >&2
      echo "  $candidate ($candidate_digest)" >&2
      exit 2
    fi
  done
  [ -n "$found" ] || {
    echo "FATAL: no source root provides step $step" >&2
    echo "  looked for gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet under:" >&2
    for root in $SOURCE_ROOTS; do echo "    $root/raw/pairs" >&2; done
    exit 2
  }
  printf '%s\n' "$found"
}

IFS=';' read -r -a judge_specs <<< "$JUDGES"
[ "${#judge_specs[@]}" -gt 0 ] || { echo "FATAL: JUDGES is empty" >&2; exit 2; }
for spec in "${judge_specs[@]}"; do
  cell=${spec%%|*}
  model=${spec#*|}
  [ -n "$cell" ] && [ -n "$model" ] && [ "$cell" != "$spec" ] || {
    echo "FATAL: JUDGES entry must be 'cell|model', got '$spec'" >&2; exit 2; }
  # A local checkpoint must exist now. An HF identifier is resolved by the serving job.
  case "$model" in
    /*) [ -f "$model/config.json" ] || {
          echo "FATAL: missing local judge model: $model/config.json" >&2; exit 2; } ;;
  esac
done

# Claim the root before staging or submitting, so two concurrent invocations cannot append
# duplicate work into one run root while both chains are still pending.
claim=$EVAL_ROOT/provenance/single_token_trajectory.claim
mkdir -p "$EVAL_ROOT/provenance"
if ! mkdir "$claim" 2>/dev/null; then
  echo "FATAL: EVAL_ROOT is already claimed for submission: $claim" >&2
  exit 2
fi
printf 'prompt_style=%s\nthinking_mode=%s\nsource_sha=%s\nclaimed_at_utc=%s\n' \
  "$JUDGE_PROMPT_STYLE" "$THINKING_MODE" "${TURING_RL_SOURCE_SHA:-unknown}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$claim/metadata.txt"

# Stage every pair set into the run root and record where it came from, so the package is
# self-contained and each step's provenance survives independently of the source roots.
mkdir -p "$EVAL_ROOT/raw/pairs"
manifest=$EVAL_ROOT/provenance/pair_sources.psv
printf 'step|source|sha256\n' > "$manifest"
for step in "${step_values[@]}"; do
  source_pairs=$(resolve_pairs "$step")
  digest=$(sha256_of "$source_pairs")
  staged=$EVAL_ROOT/raw/pairs/gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet
  if [ -f "$staged" ]; then
    [ "$(sha256_of "$staged")" = "$digest" ] || {
      echo "FATAL: staged pair set differs from its source: $staged" >&2; exit 2; }
  else
    cp -p "$source_pairs" "$staged"
  fi
  printf '%s|%s|%s\n' "$step" "$source_pairs" "$digest" >> "$manifest"
done
echo "staged ${#step_values[@]} pair set(s) -> $EVAL_ROOT/raw/pairs"

submit() {
  local dep="$1"; shift
  local deparg=""
  [ -n "$dep" ] && deparg="--dependency=$dep"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $deparg $*" >&2
    echo "dry$RANDOM"
  else
    "$SBATCH" --parsable $deparg "$@"
  fi
}

need_jid() {
  if [ "$DRY" = "1" ]; then [ -n "$1" ] && return 0; fi
  case "$1" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $2 (got '$1')" >&2; exit 1 ;;
  esac
}

for spec in "${judge_specs[@]}"; do
  cell=${spec%%|*}
  model=${spec#*|}
  previous=""
  for step in "${step_values[@]}"; do
    pairs=$EVAL_ROOT/raw/pairs/gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet
    sweep_root=$EVAL_ROOT/raw/${GEN_KEY_PREFIX}${step}/sweep
    reward_dir=$sweep_root/$cell/$THINKING_MODE/$JUDGE_PROMPT_STYLE/reward
    if [ -d "$reward_dir" ] && [ -n "$(find "$reward_dir" -maxdepth 1 -type f -print -quit)" ]; then
      echo "FATAL: refusing stale output in $reward_dir" >&2
      exit 2
    fi

    dep=""
    [ -n "$previous" ] && dep="afterok:$previous"
    exports="ALL,MODEL=$model,TP=$TP,REPLICAS=$REPLICAS,CONCURRENCY=$CONCURRENCY"
    exports="$exports,THINKING_MODE=$THINKING_MODE,JUDGE_PROMPT_STYLE=$JUDGE_PROMPT_STYLE"
    exports="$exports,CELL_NAME=$cell,PAIRS=$pairs,SWEEP_ROOT=$sweep_root"
    jid=$(submit "$dep" --gres=gpu:$((TP * REPLICAS)) --job-name="${JOB_PREFIX}_${cell}_${step}" \
      --export="$exports" -- scripts/slurm/judge_sweep_cell.sh)
    need_jid "$jid" "$cell step$step"
    echo "submitted $cell step$step -> $jid${previous:+ (after $previous)}"
    previous=$jid
  done
  echo "$cell chain tail job: $previous"
done
