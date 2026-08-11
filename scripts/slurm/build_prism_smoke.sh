#!/bin/bash
#SBATCH --job-name=prism_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/build_prism_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_DATASETS_CACHE=/home/lancewicki/data/hf_cache/datasets

if [ -f $TURING_RL_STATE_ROOT/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source $TURING_RL_STATE_ROOT/.env
  set +a
fi

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
OUT_DIR=$TURING_RL_GENERATED_DATA_ROOT/prism/history_smoke
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Build PRISM smoke slice (max_train_users=20)"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Out:   $OUT_DIR"
echo "============================================"

cd "$TURING_RL_WORK_ROOT"

$PY -m data.prism.build \
  --max_train_users 20 \
  --output      "$OUT_DIR/train.parquet" \
  --val_output  "$OUT_DIR/val.parquet" \
  --test_output "$OUT_DIR/test.parquet"
RC=$?

if [ $RC -eq 0 ]; then
  echo ""
  echo "=== row counts ==="
  $PY -c "
import pandas as pd
for name in ('train', 'val', 'test'):
    p = '$OUT_DIR/' + name + '.parquet'
    try:
        df = pd.read_parquet(p)
        print(f'{name:5s} rows={len(df)} cols={list(df.columns)}')
    except Exception as e:
        print(f'{name:5s} read failed: {e}')
"
fi

echo "============================================"
echo "Done at $(date) (rc=$RC)"
echo "============================================"
exit $RC
