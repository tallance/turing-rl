#!/bin/bash
#SBATCH --job-name=prism_split_s42
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/split_prism_full_s42-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
IN_DIR=$TURING_RL_DATA_ROOT/prism/full_s42_history
OUT_DIR=$TURING_RL_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Split PRISM FULL by user (seed=42, heldout=0.1, grpo=0.6)"
echo "In:  $IN_DIR"
echo "Out: $OUT_DIR"
echo "============================================"

cd "$TURING_RL_WORK_ROOT"

$PY -u -m data.prism.split_data \
  --input-dir  "$IN_DIR" \
  --output-dir "$OUT_DIR"
RC=$?

echo "============================================"
echo "Done at $(date) (rc=$RC)"
echo "============================================"
exit $RC
