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
# Build a generator's (human, generated) 880 pair-set. Required env: GEN_KEY
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
PKL=$REPO/results/2026-07-15-generator-sweep/raw/generator/$GEN_KEY/heldout_inference.pkl
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT=$REPO/results/2026-07-15-generator-sweep/raw/pairs/gen_${GEN_KEY}_880.parquet
mkdir -p "$(dirname "$OUT")"; cd "$REPO"
$PY scripts/build_judge_pairs.py --inference_pkl "$PKL" --test_parquet "$TEST" --out "$OUT"
RC=$?; echo "=== exit: $RC ==="; exit $RC
