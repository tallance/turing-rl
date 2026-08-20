#!/bin/bash
# Evaluate the partial frac10 generator trained against Gemma 4 12B.
set -euo pipefail

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
STATE_ROOT=${TURING_RL_STATE_ROOT:-/home/lancewicki/projects/turing-rl}
DATA_ROOT=${TURING_RL_INPUT_DATA_ROOT:-$STATE_ROOT/data}
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
STEP0_SOURCE_ROOT=${STEP0_SOURCE_ROOT:-$STATE_ROOT/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema}
SOURCE_STEP0_KEY=9b-train10pct-step0
DESTINATION_STEP0_KEY=9b-gemma12btrain-step0

RUN_TAG=9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1
STEPS="12 24 36 48 60 72"
MERGE_STEPS="12 24 36 48 60 72"
JUDGES="gemma4-12b qwen35-9b"
GEN_KEY_PREFIX=9b-gemma12btrain-step
JOB_PREFIX=te_g12t50
EVAL_ROWS=${EVAL_ROWS:-440}
PAIRS_TAG=${PAIRS_TAG:-440}
SOURCE_EVAL_PARQUET=$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/test.parquet
EVAL_PARQUET=$DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
MERGE_BATCH_SIZE=6
GEN_BATCH_SIZE=1
JUDGE_BATCH_SIZE=1
PHASE=merge
OFFSET=0

"$PY" scripts/reuse_test_eval_step.py \
  --source-root "$STEP0_SOURCE_ROOT" \
  --destination-root "$EVAL_ROOT" \
  --source-gen-key "$SOURCE_STEP0_KEY" \
  --destination-gen-key "$DESTINATION_STEP0_KEY" \
  --pairs-tag "$PAIRS_TAG" \
  --expect-pairs "$EVAL_ROWS" \
  --mode on \
  --cells gemma4-12b qwen35-9b

export EVAL_ROOT RUN_TAG SOURCE_EVAL_PARQUET EVAL_PARQUET EVAL_ROWS
export STEPS MERGE_STEPS JUDGES GEN_KEY_PREFIX PAIRS_TAG JOB_PREFIX
export MERGE_BATCH_SIZE GEN_BATCH_SIZE JUDGE_BATCH_SIZE PHASE OFFSET

bash scripts/launch_frac10_test50_eval.sh
