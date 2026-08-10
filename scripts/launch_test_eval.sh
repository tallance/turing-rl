#!/bin/bash
# Test-set eval launcher for the 9B GRPO checkpoints (held-out 880 / 128 unseen users).
#
# Deliberately NOT launch_generator_sweep.sh: that one submits generation at --gres=gpu:8 and
# serializes every job through PREV, and its generator matrix is hardcoded (an unknown GEN_ONLY
# silently matches nothing). Generation here needs exactly 1 GPU (vLLM TP=1), so the three
# checkpoints generate CONCURRENTLY; only the 8-GPU judge cells are serialized.
#
# Shape:
#   gen(gpu:1) x3   in parallel
#     -> build(gpu:0) x3   each afterok its own gen
#        -> judge(gpu:8) xN   serialized (one 8-replica node at a time)
#
# Sampling mirrors job 13634's val_kwargs (Qwen3 model-card) so test numbers extend the
# in-training validation curve; judge env mirrors 13634's judge exactly.
#
# PREREQUISITE: every non-zero step in STEPS must already have a GATED dense model built by
# scripts/slurm/merge_grpo_ckpt.sh. veRL checkpoints are unmerged LoRA, so generating straight
# from one evaluates the SFT base and silently reproduces step-0 numbers.
#
# End-to-end runbook: docs/test-set-eval.md
#
# Usage:
#   DRY=1 bash scripts/launch_test_eval.sh                 # print the plan
#   bash scripts/launch_test_eval.sh                       # full gen+judge, 9B thinking-on
#   DO_GEN=0 JUDGES="qwen35-27b qwen35-397b" bash scripts/launch_test_eval.sh
#                                                          # judge-only re-run over existing pairs
#   GEN_ONLY=9b-grpo-step8 bash scripts/launch_test_eval.sh
#
#   # Reuse pairs/models from another result root.
#   EVAL_ROOT=$REPO/results/<new> MODELS_ROOT=$REPO/results/<old>/models \
#   GEN_KEY_PREFIX=9b-full5ep-step STEPS="0 32 ..." DO_GEN=0 \
#   JUDGES="gemma4-31b" bash scripts/launch_test_eval.sh
set -uo pipefail
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO" || exit 2

EVAL_ROOT=${EVAL_ROOT:-$REPO/results/2026-08-03-test-eval-9b-half}
DRY=${DRY:-0}
DO_GEN=${DO_GEN:-1}
DO_JUDGE=${DO_JUDGE:-1}
JUDGES=${JUDGES:-qwen35-9b}
MODES=${MODES:-on}
GEN_ONLY=${GEN_ONLY:-}
CHAIN_AFTER=${CHAIN_AFTER:-}

MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}

# --- evaluated split and reusable artifact locations ---
export EVAL_PARQUET=${EVAL_PARQUET:-$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
export EVAL_EXPECT=${EVAL_EXPECT:-heldout}
export PAIRS_TAG=${PAIRS_TAG:-880}
MODELS_ROOT=${MODELS_ROOT:-$EVAL_ROOT/models}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-grpo-step}

# --- generation config: Qwen3 model-card = job 13634 val_kwargs; lengths = validation caps ---
export SWEEP_BASE="$EVAL_ROOT"
export BACKEND=${BACKEND:-vllm}
export GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.7}
export GEN_TOP_P=${GEN_TOP_P:-0.8}
export GEN_TOP_K=${GEN_TOP_K:-20}
export GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}
export GEN_TRUNCATE_PROMPT_TOKENS=${GEN_TRUNCATE_PROMPT_TOKENS:-12500}
export GEN_MAX_MODEL_LEN=${GEN_MAX_MODEL_LEN:-13524}

# --- judge env: identical to job 13634's judge, so val and test are scored the same way. ---
# The JSON must be exported (NOT placed in --export=ALL,VAR=..): Slurm splits that list on
# commas and the JSON's internal comma would corrupt it.
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
# The 400s default drops pairs: thinking-ON 9B with an 8192-token budget can exceed it under
# 32-way concurrency, and run_judge_sweep_cell.py swallows the TimeoutError (err counter, exit 0).
# Jobs 13946-13948 lost 19/23/10 of 880 pairs that way; the 2026-07 sweep baseline lost 9.
export PERSONA_OPENAI_TIMEOUT_SECONDS=${PERSONA_OPENAI_TIMEOUT_SECONDS:-1800}

# gen_key -> model dir. step 0 is the pre-RL SFT init; every other step is a dense merge built
# by scripts/merge_grpo_adapter.py and cleared by scripts/validate_grpo_merge.py.
# STEPS selects which checkpoints to evaluate, e.g. STEPS="24 32" for the second half.
STEPS=${STEPS:-"0 8 16"}
GENERATORS=""
for s in $STEPS; do
  if [ "$s" = "0" ]; then m=$MERGED_EP3; else m=$MODELS_ROOT/step$s/hf_dense; fi
  GENERATORS="${GENERATORS}${GEN_KEY_PREFIX}${s}|${m}
"
done

CELLS=$(/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python -c "
from configs.judge_sweep_cells import cell_list, extra_cell
cells = cell_list('qwen3.5') + [extra_cell('gemma4-12b'), extra_cell('gemma4-31b')]
for c in cells:
    print(c['cell_name'], c['model_id'], c['tp'], c['replicas'], c.get('concurrency', 32))
")
FILTERED=""
for j in $JUDGES; do
  line=$(echo "$CELLS" | awk -v n="$j" '$1==n')
  [ -z "$line" ] && { echo "FATAL: judge '$j' not in cell_list('qwen3.5')" >&2; exit 1; }
  FILTERED="${FILTERED}${line}
"
done
CELLS="$FILTERED"

submit () {
  local dep="$1"; shift
  # Plain string, not an array: an empty array expansion under `set -u` errors on bash 3.2.
  local deparg=""; [ -n "$dep" ] && deparg="--dependency=$dep"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $deparg $*" >&2
    echo "dry$RANDOM"; return 0
  fi
  sbatch --parsable $deparg "$@"
}
need_jid () {
  if [ "$DRY" = "1" ]; then [ -n "$1" ] && return 0; fi
  case "$1" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $2 (got '$1')" >&2; exit 1 ;;
  esac
}

mkdir -p "$EVAL_ROOT/raw/pairs" "$EVAL_ROOT/raw/sweep" "$REPO/logs"

# Assert and record the split before spending GPU time.
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/check_eval_split.py \
    --eval_parquet "$EVAL_PARQUET" --expect "$EVAL_EXPECT" \
    --out_json "$EVAL_ROOT/split_guard.json" \
  || { echo "FATAL: split guard rejected the eval set; nothing submitted." >&2; exit 1; }

echo "=== test-set eval: EVAL_ROOT=$EVAL_ROOT judges='$JUDGES' modes='$MODES' gen=$DO_GEN judge=$DO_JUDGE ==="
echo "=== eval set: $EVAL_PARQUET (expect=$EVAL_EXPECT, tag=$PAIRS_TAG) ==="
echo "=== sampling: T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K max_tokens=$GEN_MAX_TOKENS ==="

# ---------------- phase 1: generation (parallel, 1 GPU each) + pair build ----------------
BUILD_DEPS=""   # lines: gen_key|build_jid_or_empty
N_MATCHED=0
for entry in $GENERATORS; do
  gk=${entry%%|*}; model=${entry#*|}
  if [ -n "$GEN_ONLY" ] && [ "$gk" != "$GEN_ONLY" ]; then continue; fi
  N_MATCHED=$((N_MATCHED+1))
  pairs=$EVAL_ROOT/raw/pairs/gen_${gk}_${PAIRS_TAG}.parquet

  if [ "$DO_GEN" = "1" ]; then
    if [ ! -d "$model" ]; then
      echo "FATAL: model dir missing for $gk: $model" >&2
      echo "       build it with scripts/merge_grpo_adapter.py and clear scripts/validate_grpo_merge.py first" >&2
      exit 1
    fi
    # gpu:1 overrides generator_infer.sh's gpu:8 header -- vLLM runs TP=1, so 7 would idle.
    gjid=$(submit "$CHAIN_AFTER" --gres=gpu:1 --job-name=tegen_${gk} \
      --export=ALL,GEN_KEY=$gk,MODEL_ID=$model,CKPT=,BACKEND=$BACKEND \
      scripts/slurm/generator_infer.sh)
    need_jid "$gjid" "gen $gk"; echo "submitted gen  $gk -> $gjid ($model)" >&2
    bjid=$(submit "afterok:$gjid" --gres=gpu:0 --job-name=tebuild_${gk} \
      --export=ALL,GEN_KEY=$gk scripts/slurm/build_pairs.sh)
    need_jid "$bjid" "build $gk"; echo "submitted build $gk -> $bjid" >&2
    BUILD_DEPS="${BUILD_DEPS}${gk}|${bjid}
"
  else
    [ -f "$pairs" ] || { echo "FATAL: DO_GEN=0 but no pair-set for $gk ($pairs)" >&2; exit 1; }
    echo "reusing pairs $gk -> $pairs" >&2
    BUILD_DEPS="${BUILD_DEPS}${gk}|
"
  fi
done

# A typo'd GEN_ONLY must not look like a successful no-op submit.
if [ "$N_MATCHED" -eq 0 ]; then
  echo "FATAL: GEN_ONLY='$GEN_ONLY' matched no generator. Known keys:" >&2
  for entry in $GENERATORS; do echo "  ${entry%%|*}" >&2; done
  exit 1
fi

# ---------------- phase 2: judging (serialized, 8 GPUs each) ----------------
[ "$DO_JUDGE" = "1" ] || { echo "DO_JUDGE=0: stopping after generation."; exit 0; }

PREV=""
for dep_entry in $BUILD_DEPS; do
  gk=${dep_entry%%|*}; bjid=${dep_entry#*|}
  pairs=$EVAL_ROOT/raw/pairs/gen_${gk}_${PAIRS_TAG}.parquet
  # The client appends $CELL_NAME/$THINKING_MODE, so pass the per-generator sweep ROOT.
  sweep_root=$EVAL_ROOT/raw/$gk/sweep
  # Here-string, NOT `echo | while`: a pipe puts the loop in a subshell and PREV would not
  # survive, silently breaking judge serialization (and blowing the 8-GPU budget).
  while read -r cell_name model_id tp replicas concurrency; do
    [ -z "$cell_name" ] && continue
    for mode in $MODES; do
      gpus=$((tp*replicas))
      # Pre-submit freshness guard. Reward dumps ACCUMULATE in a reused dir, so a re-run
      # would silently mix stale rows with new. verify_judge_completeness.py catches that
      # afterwards, but only after burning an 8-GPU cell for hours -- refuse up front.
      rdir=$sweep_root/$cell_name/$mode/reward
      if [ -d "$rdir" ] && [ -n "$(ls -A "$rdir" 2>/dev/null)" ]; then
        if [ "${FORCE_REJUDGE:-0}" = "1" ]; then
          echo "FORCE_REJUDGE=1: clearing $rdir" >&2
          [ "$DRY" = "1" ] || rm -rf "$rdir"
        else
          echo "FATAL: reward dir already has output: $rdir" >&2
          echo "       re-judging would mix stale rows with new. Move it aside, or set FORCE_REJUDGE=1." >&2
          exit 1
        fi
      fi
      dep=""
      [ -n "$bjid" ] && dep="afterok:$bjid"
      [ -n "$PREV" ] && dep="${dep:+$dep,}afterany:$PREV"
      sjid=$(submit "$dep" --gres=gpu:$gpus --job-name=tejudge_${gk}_${cell_name}_${mode} \
        --export=ALL,MODEL=$model_id,TP=$tp,REPLICAS=$replicas,CONCURRENCY=$concurrency,THINKING_MODE=$mode,CELL_NAME=$cell_name,PAIRS=$pairs,SWEEP_ROOT=$sweep_root \
        scripts/slurm/judge_sweep_cell.sh)
      need_jid "$sjid" "judge $gk $cell_name $mode"
      echo "submitted judge $gk $cell_name/$mode -> $sjid (gpu:$gpus)" >&2
      PREV="$sjid"
    done
  done <<< "$CELLS"
done

echo "=== submitted. reward dumps: $EVAL_ROOT/raw/<gen_key>/sweep/<cell>/<mode>/reward ==="
echo "=== verify with: python scripts/verify_judge_completeness.py --eval_root $EVAL_ROOT ==="
echo "chain tail job: $PREV" >&2
