#!/bin/bash
# Score the trained judges of one prompt-style arm plus five zero-shot baselines on one
# pinned thinking policy over the frozen 880-pair held-out set.
#   full          -> the four round-1 GRPO judges + five zero-shot models (nine cells)
#   single_token  -> the two cross-entropy judges + the same five models (seven cells)
# Each arm names and checks only its own trained models; see the MATRIX selection below.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
CODE_ROOT=${TURING_RL_CODE_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=$CODE_ROOT/scripts/snapshot_sbatch.sh
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
PAIRS=${PAIRS:-/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet}
THINKING_MODE=${THINKING_MODE:-on}
CONFIRM_THINKING_OFF=${CONFIRM_THINKING_OFF:-0}
JOB_PREFIX=${JOB_PREFIX:-jeval}
DRY=${DRY:-0}

JUDGE_4B_GRADED_MODEL=${JUDGE_4B_GRADED_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-14-judge-4b-eval-v2/models/judge-4b-graded-step52/hf_dense}
JUDGE_4B_DIRECTIONAL_MODEL=${JUDGE_4B_DIRECTIONAL_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-14-judge-4b-eval-v2/models/judge-4b-directional-step52/hf_dense}
JUDGE_9B_GRADED_MODEL=${JUDGE_9B_GRADED_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/judge-9b-graded-step52/hf_dense}
JUDGE_9B_DIRECTIONAL_MODEL=${JUDGE_9B_DIRECTIONAL_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/judge-9b-directional-step52/hf_dense}

# The single_token arm's trained judges. Merged dense checkpoints, same convention as above;
# the defaults are where this experiment's merge step writes them, and Task 17 passes them
# explicitly. They are only consulted -- and only required to exist -- for that arm.
JUDGE_4B_CE_MODEL=${JUDGE_4B_CE_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-26-single-token-judge/models/judge-4b-ce/hf_dense}
JUDGE_9B_CE_MODEL=${JUDGE_9B_CE_MODEL:-/home/lancewicki/projects/turing-rl/results/2026-08-26-single-token-judge/models/judge-9b-ce/hf_dense}

case "$THINKING_MODE" in
  on) ;;
  off)
    [ "$CONFIRM_THINKING_OFF" = "1" ] || {
      echo "FATAL: thinking-off evaluation requires CONFIRM_THINKING_OFF=1" >&2
      echo "       THINKING_MODE defaults to on so off cannot be selected accidentally." >&2
      exit 2
    }
    case "$EVAL_ROOT" in
      /*thinking-off*) ;;
      *) echo "FATAL: a thinking-off EVAL_ROOT must contain 'thinking-off': $EVAL_ROOT" >&2; exit 2 ;;
    esac
    ;;
  *) echo "FATAL: THINKING_MODE must be on|off, got '$THINKING_MODE'" >&2; exit 2 ;;
esac

JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE:-full}
case "$JUDGE_PROMPT_STYLE" in
  full) ;;
  single_token)
    case "$EVAL_ROOT" in
      *single-token*) ;;
      *) echo "FATAL: a single_token EVAL_ROOT must name the style: $EVAL_ROOT" >&2; exit 2 ;;
    esac
    ;;
  *) echo "FATAL: JUDGE_PROMPT_STYLE must be full|single_token, got '$JUDGE_PROMPT_STYLE'" >&2; exit 2 ;;
esac
# The single-token protocol pins enable_thinking=False on every request (eval/single_token_judge.py),
# so THINKING_MODE only decides how the cell is *labelled*: the mode directory, the provenance
# thinking_mode field and the submission record would all read "on" for a run that executed
# thinking-off. Reject rather than silently rewrite THINKING_MODE -- an altered submission is
# exactly as hard to notice as the mislabel it fixes, and the thinking-off arm already demands an
# explicit opt-in. Checked after both variables are validated, so the values quoted here are real.
if [ "$JUDGE_PROMPT_STYLE" = "single_token" ] && [ "$THINKING_MODE" != "off" ]; then
  echo "FATAL: JUDGE_PROMPT_STYLE=$JUDGE_PROMPT_STYLE requires THINKING_MODE=off, got THINKING_MODE=$THINKING_MODE" >&2
  echo "       The single-token judge always serves with thinking disabled, so a thinking-on run" >&2
  echo "       would attribute every artifact to the thinking-on arm. Re-submit with" >&2
  echo "       THINKING_MODE=off CONFIRM_THINKING_OFF=1 and an EVAL_ROOT naming both." >&2
  exit 2
fi
export JUDGE_PROMPT_STYLE

case "$EVAL_ROOT" in /*) ;; *) echo "FATAL: EVAL_ROOT must be absolute: $EVAL_ROOT" >&2; exit 2 ;; esac
[ -f "$PAIRS" ] || { echo "FATAL: pair set not found: $PAIRS" >&2; exit 2; }
# Each arm serves its own trained judges, so each checks only its own. Requiring the other
# arm's checkpoints would block a run on models it never loads.
#
# cell_name, model, tensor parallelism, replicas, and concurrency per endpoint.
# Every row occupies one eight-A100 node.
if [ "$JUDGE_PROMPT_STYLE" = "single_token" ]; then
  CHECKED_MODELS=("$JUDGE_4B_CE_MODEL" "$JUDGE_9B_CE_MODEL")
  # Cell names carry a -st suffix. Both arms land in one merged table where prompt_style is
  # a column rather than part of the cell name, so a name shared with the full arm would
  # leave the row unattributable to an arm by name alone.
  # Serving shapes for the five zero-shot rows are copied verbatim from the full arm below:
  # they encode node-level serving constraints (gemma4-31b only fits at TP=8, and at low
  # concurrency), which the prompt style does not change. The two CE rows mirror the full
  # arm's 4B and 9B trained-judge rows. The trained cells lead the chain because a failure
  # in an afterok chain strands everything after it, and they are what this arm measures.
  MATRIX=$(cat <<EOF
judge-9b-ce-st $JUDGE_9B_CE_MODEL 1 8 32
judge-4b-ce-st $JUDGE_4B_CE_MODEL 1 8 32
qwen35-27b-st Qwen/Qwen3.5-27B 8 1 32
gemma4-31b-st google/gemma-4-31B-it 8 1 4
gemma4-12b-st google/gemma-4-12B-it 1 8 4
qwen35-9b-st Qwen/Qwen3.5-9B 1 8 32
qwen35-4b-st Qwen/Qwen3.5-4B 1 8 32
EOF
)
else
  CHECKED_MODELS=(
    "$JUDGE_4B_GRADED_MODEL" "$JUDGE_4B_DIRECTIONAL_MODEL"
    "$JUDGE_9B_GRADED_MODEL" "$JUDGE_9B_DIRECTIONAL_MODEL"
  )
  # The known high-retry directional 9B cell is deliberately last so a failure there cannot
  # block the other comparison cells.
  MATRIX=$(cat <<EOF
judge-9b-graded-step52 $JUDGE_9B_GRADED_MODEL 1 8 32
judge-4b-graded-step52 $JUDGE_4B_GRADED_MODEL 1 8 32
judge-4b-directional-step52 $JUDGE_4B_DIRECTIONAL_MODEL 1 8 32
qwen35-27b Qwen/Qwen3.5-27B 8 1 32
gemma4-31b google/gemma-4-31B-it 8 1 4
gemma4-12b google/gemma-4-12B-it 1 8 4
qwen35-9b Qwen/Qwen3.5-9B 1 8 32
qwen35-4b Qwen/Qwen3.5-4B 1 8 32
judge-9b-directional-step52 $JUDGE_9B_DIRECTIONAL_MODEL 1 8 32
EOF
)
fi

for model in "${CHECKED_MODELS[@]}"; do
  [ -f "$model/config.json" ] || { echo "FATAL: trained judge model is incomplete: $model" >&2; exit 2; }
done

SWEEP_ROOT=$EVAL_ROOT/raw/sweep
while read -r cell _model _tp _replicas _concurrency; do
  reward_dir=$SWEEP_ROOT/$cell/$THINKING_MODE/reward
  [ "$JUDGE_PROMPT_STYLE" = "full" ] || reward_dir=$SWEEP_ROOT/$cell/$THINKING_MODE/$JUDGE_PROMPT_STYLE/reward
  if [ -d "$reward_dir" ] && [ -n "$(find "$reward_dir" -maxdepth 1 -type f -print -quit)" ]; then
    echo "FATAL: refusing stale output in $reward_dir" >&2
    exit 2
  fi
done <<< "$MATRIX"

# Claim the run root atomically before the first sbatch. Reward-file guards cannot detect a
# duplicate invocation while the first chain is still pending, which would make both chains append
# to the same JSONL files. A retained retry must use a fresh run root after investigating the claim.
claim=$EVAL_ROOT/provenance/judge_eval_matrix_submission.claim
mkdir -p "$EVAL_ROOT/provenance"
if ! mkdir "$claim" 2>/dev/null; then
  echo "FATAL: EVAL_ROOT is already claimed for submission: $claim" >&2
  exit 2
fi
printf 'thinking_mode=%s\nsource_sha=%s\nclaimed_at_utc=%s\n' \
  "$THINKING_MODE" "${TURING_RL_SOURCE_SHA:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$claim/metadata.txt"
mkdir -p "$SWEEP_ROOT"

echo "=== judge held-out matrix: THINKING_MODE=$THINKING_MODE ==="
echo "=== pairs=$PAIRS ==="
echo "=== output=$SWEEP_ROOT ==="

previous=""
while read -r cell model tp replicas concurrency; do
  [ -n "$cell" ] || continue
  gpus=$((tp * replicas))
  dependency=""
  [ -n "$previous" ] && dependency="--dependency=afterok:$previous"
  exports="ALL,MODEL=$model,TP=$tp,REPLICAS=$replicas,CONCURRENCY=$concurrency"
  exports="$exports,THINKING_MODE=$THINKING_MODE,CELL_NAME=$cell,PAIRS=$PAIRS,SWEEP_ROOT=$SWEEP_ROOT"
  # Explicit, not merely inherited through --export=ALL: the style decides both the judge
  # protocol and the cell's output path, so it must be visible in the submission record.
  exports="$exports,JUDGE_PROMPT_STYLE=$JUDGE_PROMPT_STYLE"

  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $dependency --gres=gpu:$gpus --job-name=${JOB_PREFIX}_${cell} --export=$exports -- scripts/slurm/judge_sweep_cell.sh" >&2
    jid="dry-$cell"
  else
    jid=$("$SBATCH" --parsable $dependency --gres=gpu:$gpus \
      --job-name="${JOB_PREFIX}_${cell}" --export="$exports" \
      -- scripts/slurm/judge_sweep_cell.sh)
    case "$jid" in
      ''|*[!0-9]*) echo "FATAL: sbatch failed for $cell (got '$jid')" >&2; exit 1 ;;
    esac
  fi
  echo "submitted $cell ($THINKING_MODE) -> $jid${previous:+ (afterok $previous)}"
  previous=$jid
done <<< "$MATRIX"

echo "matrix tail job: $previous"
