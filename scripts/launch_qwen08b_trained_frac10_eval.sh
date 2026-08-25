#!/bin/bash
# Evaluate the frac10 generator trained against Qwen3.5-0.8B with thinking off.
set -euo pipefail

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
STATE_ROOT=${TURING_RL_STATE_ROOT:-/home/lancewicki/projects/turing-rl}
DATA_ROOT=${TURING_RL_INPUT_DATA_ROOT:-$STATE_ROOT/data}
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
STEP0_SOURCE_ROOT=${STEP0_SOURCE_ROOT:-$STATE_ROOT/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema}
SOURCE_STEP0_KEY=9b-train10pct-step0
DESTINATION_STEP0_KEY=9b-qwen08btrain-step0

RUN_TAG=9b_frac10_20ep_qwen08b_nothink_kl1e4_lr1e4_temp1
STEPS="0 12 24 36 48 60 72 84 96 108 120"
MERGE_STEPS="12 24 36 48 60 72 84 96 108 120"
JUDGES="qwen35-0.8b gemma4-12b gemma4-31b qwen35-9b"
JUDGE_MODES="off on on on"
REUSED_STEP0_CELLS="gemma4-12b gemma4-31b qwen35-9b"
GEN_KEY_PREFIX=9b-qwen08btrain-step
JOB_PREFIX=te_q08t50
EVAL_ROWS=${EVAL_ROWS:-440}
PAIRS_TAG=${PAIRS_TAG:-440}
SOURCE_EVAL_PARQUET=${SOURCE_EVAL_PARQUET:-$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
EVAL_PARQUET=${EVAL_PARQUET:-$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet}
MERGE_BATCH_SIZE=${MERGE_BATCH_SIZE:-3}
GEN_BATCH_SIZE=${GEN_BATCH_SIZE:-1}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-1}
PHASE=${PHASE:-merge}
OFFSET=${OFFSET:-0}
REUSE_STEP0=${REUSE_STEP0:-1}

if [ "$REUSE_STEP0" = "1" ]; then
  "$PY" scripts/reuse_test_eval_step.py \
    --source-root "$STEP0_SOURCE_ROOT" \
    --destination-root "$EVAL_ROOT" \
    --source-gen-key "$SOURCE_STEP0_KEY" \
    --destination-gen-key "$DESTINATION_STEP0_KEY" \
    --pairs-tag "$PAIRS_TAG" \
    --expect-pairs "$EVAL_ROWS" \
    --expected-eval-parquet "$EVAL_PARQUET" \
    --mode on \
    --cells gemma4-12b gemma4-31b qwen35-9b
elif [ "$REUSE_STEP0" = "0" ] && [ "${DRY:-0}" = "1" ]; then
  echo "[DRY] reuse verified step 0 from $STEP0_SOURCE_ROOT" >&2
else
  echo "FATAL: REUSE_STEP0 must be 1; setting it to 0 is allowed only with DRY=1" >&2
  exit 2
fi

export EVAL_ROOT RUN_TAG SOURCE_EVAL_PARQUET EVAL_PARQUET EVAL_ROWS
export STEPS MERGE_STEPS JUDGES JUDGE_MODES REUSED_STEP0_CELLS
export GEN_KEY_PREFIX PAIRS_TAG JOB_PREFIX
export MERGE_BATCH_SIZE GEN_BATCH_SIZE JUDGE_BATCH_SIZE PHASE OFFSET

bash scripts/launch_frac10_test50_eval.sh
