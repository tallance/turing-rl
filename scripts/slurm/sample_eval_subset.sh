#!/bin/bash
#SBATCH --job-name=te_eval_subset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --partition=a100
#SBATCH --account=rfai
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/te-eval-subset-%j.out

set -euo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
: "${SOURCE_EVAL_PARQUET:?SOURCE_EVAL_PARQUET is required}"
: "${EVAL_PARQUET:?EVAL_PARQUET is required}"
: "${EVAL_ROOT:?EVAL_ROOT is required}"
EVAL_ROWS=${EVAL_ROWS:-440}
EVAL_SEED=${EVAL_SEED:-42}

"$PY" scripts/sample_eval_parquet.py \
  --input "$SOURCE_EVAL_PARQUET" --output "$EVAL_PARQUET" \
  --rows "$EVAL_ROWS" --seed "$EVAL_SEED"
"$PY" scripts/check_eval_split.py \
  --eval_parquet "$EVAL_PARQUET" --split_root "$(dirname "$SOURCE_EVAL_PARQUET")" \
  --expect heldout --out_json "$EVAL_ROOT/split_guard.json"
