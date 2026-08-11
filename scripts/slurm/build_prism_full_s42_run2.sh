#!/bin/bash
#SBATCH --job-name=prism_full_s42_run2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/build_prism_full_s42_run2-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_DATASETS_CACHE=/home/lancewicki/data/hf_cache/datasets
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

if [ -f $TURING_RL_STATE_ROOT/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source $TURING_RL_STATE_ROOT/.env
  set +a
fi

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
BUILD_DIR=$TURING_RL_GENERATED_DATA_ROOT/prism/full_s42_history_run2
SPLIT_DIR=$TURING_RL_GENERATED_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10_run2
mkdir -p "$BUILD_DIR" "$SPLIT_DIR"

echo "============================================"
echo "Determinism run #2: build + split PRISM at seed=42"
echo "Date: $(date)  Host: $(hostname)"
echo "============================================"

cd "$TURING_RL_WORK_ROOT"

$PY -u -m data.prism.build \
  --output      "$BUILD_DIR/train.parquet" \
  --val_output  "$BUILD_DIR/val.parquet" \
  --test_output "$BUILD_DIR/test.parquet" \
  --conditioning_mode history \
  --shuffle_rows
RC=$?
[ $RC -eq 0 ] || { echo "build failed rc=$RC"; exit $RC; }

$PY -u -m data.prism.split_data \
  --input-dir  "$BUILD_DIR" \
  --output-dir "$SPLIT_DIR"
RC=$?

echo "============================================"
echo "Done at $(date) (rc=$RC)"
echo "============================================"
exit $RC
