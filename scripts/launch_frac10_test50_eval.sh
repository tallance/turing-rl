#!/bin/bash
# Evaluate the 10%-training, 10-epoch GRPO trajectory on a frozen 50% held-out subset.
set -euo pipefail

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
RUN_TAG=${RUN_TAG:-9b_frac10_10ep_kl1e4_lr1e4_temp1}
SOURCE_EVAL_PARQUET=${SOURCE_EVAL_PARQUET:-$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
EVAL_PARQUET=${EVAL_PARQUET:-$TURING_RL_GENERATED_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet}
EVAL_ROWS=${EVAL_ROWS:-440}
EVAL_SEED=${EVAL_SEED:-42}

STEPS=${STEPS:-"0 6 12 18 24 30 36 42 48 54 60"}
MERGE_STEPS=${MERGE_STEPS:-"6 12 18 24 30 36 42 48 54 60"}
JUDGES=${JUDGES:-"qwen35-9b gemma4-12b gemma4-31b qwen35-4b qwen35-27b"}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-train10pct-step}
PAIRS_TAG=${PAIRS_TAG:-440}
JOB_PREFIX=${JOB_PREFIX:-te_t10t50}
MERGE_BATCH_SIZE=${MERGE_BATCH_SIZE:-4}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-1}
PHASE=${PHASE:-prepare}
OFFSET=${OFFSET:-0}
DRY=${DRY:-0}

MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
CONTINUE_SCRIPT=${CONTINUE_SCRIPT:-$REPO/scripts/slurm/continue_full_schema_eval.sh}

export EVAL_ROOT RUN_TAG SOURCE_EVAL_PARQUET EVAL_PARQUET EVAL_ROWS EVAL_SEED
export STEPS JUDGES GEN_KEY_PREFIX PAIRS_TAG JOB_PREFIX MERGED_EP3 CONTINUE_SCRIPT
export MERGE_STEPS MERGE_BATCH_SIZE GEN_BATCH_SIZE JUDGE_BATCH_SIZE
export SWEEP_BASE="$EVAL_ROOT" EVAL_EXPECT=heldout BACKEND=vllm
export GEN_TEMPERATURE=0.7 GEN_TOP_P=0.8 GEN_TOP_K=20 GEN_MAX_TOKENS=1024
export GEN_TRUNCATE_PROMPT_TOKENS=12500 GEN_MAX_MODEL_LEN=13524

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

submit_continuation() {
  local dep="$1" next_phase="$2" next_offset="$3"
  local jid
  jid=$(submit "$dep" --gres=gpu:0 --job-name="${JOB_PREFIX}_continue" \
    --export=ALL,NEXT_PHASE="$next_phase",NEXT_OFFSET="$next_offset" -- \
    scripts/slurm/continue_frac10_test50_eval.sh)
  need_jid "$jid" "continuation $next_phase/$next_offset"
  echo "$jid"
}

case "$OFFSET" in ''|*[!0-9]*) echo "FATAL: OFFSET must be a non-negative integer" >&2; exit 2 ;; esac
[ "$EVAL_ROWS" = "$PAIRS_TAG" ] || {
  echo "FATAL: EVAL_ROWS=$EVAL_ROWS must equal PAIRS_TAG=$PAIRS_TAG" >&2
  exit 2
}

case "$PHASE" in
  prepare)
    [ -f "$SOURCE_EVAL_PARQUET" ] || { echo "FATAL: missing source eval parquet: $SOURCE_EVAL_PARQUET" >&2; exit 2; }
    prep=$(submit "" --gres=gpu:0 --job-name="${JOB_PREFIX}_prepare" --export=ALL -- \
      scripts/slurm/sample_eval_subset.sh)
    need_jid "$prep" "held-out subset"
    tail=$(submit_continuation "afterok:$prep" merge 0)
    echo "submitted prepare=$prep continuation=$tail"
    ;;

  merge)
    read -r -a values <<< "$MERGE_STEPS"
    [ "$OFFSET" -lt "${#values[@]}" ] || { echo "FATAL: merge offset $OFFSET is past ${#values[@]} steps" >&2; exit 2; }
    end=$((OFFSET + MERGE_BATCH_SIZE)); [ "$end" -le "${#values[@]}" ] || end=${#values[@]}
    ids=""
    for ((i=OFFSET; i<end; i++)); do
      step=${values[$i]}
      actor=$REPO/results/grpo/rl-generator/$RUN_TAG/checkpoints/global_step_${step}/actor
      [ "$DRY" = "1" ] || [ -d "$actor" ] || { echo "FATAL: missing actor: $actor" >&2; exit 2; }
      jid=$(submit "" --gres=gpu:0 --job-name="${JOB_PREFIX}_merge_${step}" \
        --export=ALL,STEP="$step" -- scripts/slurm/merge_grpo_ckpt.sh)
      need_jid "$jid" "merge step$step"
      ids="${ids:+$ids:}$jid"
    done
    if [ "$end" -lt "${#values[@]}" ]; then next_phase=merge; next_offset=$end
    else next_phase=generate; next_offset=0
    fi
    tail=$(submit_continuation "afterok:$ids" "$next_phase" "$next_offset")
    echo "submitted merge indices [$OFFSET,$end) continuation=$tail"
    ;;

  generate)
    read -r -a values <<< "$STEPS"
    [ "$OFFSET" -lt "${#values[@]}" ] || { echo "FATAL: generation offset $OFFSET is past ${#values[@]} steps" >&2; exit 2; }
    end=$((OFFSET + GEN_BATCH_SIZE)); [ "$end" -le "${#values[@]}" ] || end=${#values[@]}
    build_ids=""
    for ((i=OFFSET; i<end; i++)); do
      step=${values[$i]}
      if [ "$step" = "0" ]; then model=$MERGED_EP3; else model=$EVAL_ROOT/models/step${step}/hf_dense; fi
      [ "$DRY" = "1" ] || [ -d "$model" ] || { echo "FATAL: missing validated dense model: $model" >&2; exit 2; }
      gen_key=${GEN_KEY_PREFIX}${step}
      gen=$(submit "" --gres=gpu:1 --job-name="${JOB_PREFIX}_gen_${step}" \
        --export=ALL,GEN_KEY="$gen_key",MODEL_ID="$model",CKPT= -- \
        scripts/slurm/generator_infer.sh)
      need_jid "$gen" "generation step$step"
      build=$(submit "afterok:$gen" --gres=gpu:0 --job-name="${JOB_PREFIX}_build_${step}" \
        --export=ALL,GEN_KEY="$gen_key" -- scripts/slurm/build_pairs.sh)
      need_jid "$build" "pair build step$step"
      build_ids="${build_ids:+$build_ids:}$build"
    done
    if [ "$end" -lt "${#values[@]}" ]; then next_phase=generate; next_offset=$end
    else next_phase=judge; next_offset=0
    fi
    tail=$(submit_continuation "afterok:$build_ids" "$next_phase" "$next_offset")
    echo "submitted generation indices [$OFFSET,$end) continuation=$tail"
    ;;

  judge)
    OFFSET=$OFFSET BATCH_SIZE=$JUDGE_BATCH_SIZE bash scripts/launch_full_schema_eval.sh
    ;;

  *) echo "FATAL: unknown PHASE=$PHASE (expected prepare|merge|generate|judge)" >&2; exit 2 ;;
esac
