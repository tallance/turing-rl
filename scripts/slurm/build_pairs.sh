#!/bin/bash
#SBATCH --job-name=build_pairs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/build_pairs-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
# Build a generator's (human, generated) pair-set. Required env: GEN_KEY
# Optional: SWEEP_BASE   relocates the input/output tree (default: the 2026-07-15 sweep)
#           EVAL_PARQUET the prompt set that was generated on (default: held-out test.parquet)
#           PAIRS_TAG    row-count tag in the output filename (default: 880); must match the
#                        parquet's row count, or the artifact name lies about what it holds.
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
SWEEP_BASE=${SWEEP_BASE:-$REPO/results/2026-07-15-generator-sweep}
PKL=$SWEEP_BASE/raw/generator/$GEN_KEY/heldout_inference.pkl
TEST=${EVAL_PARQUET:-$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
PAIRS_TAG=${PAIRS_TAG:-880}
OUT=$SWEEP_BASE/raw/pairs/gen_${GEN_KEY}_${PAIRS_TAG}.parquet
mkdir -p "$(dirname "$OUT")"; cd "$REPO"

# The tag is baked into the artifact name and into every downstream --expect_pairs. If it does not
# equal the eval set's row count the filename misreports what the file holds, which survives into
# the results tree. Cheap to check, so check.
$PY -c "
import sys, pandas as pd
n = len(pd.read_parquet('$TEST', columns=['extra_info']))
if n != $PAIRS_TAG:
    sys.exit(f'PAIRS_TAG=$PAIRS_TAG but $TEST has {n} rows; the pair-set name would be wrong')
" || exit 2

$PY scripts/build_judge_pairs.py --inference_pkl "$PKL" --test_parquet "$TEST" --out "$OUT"
RC=$?; echo "=== exit: $RC ==="; exit $RC
