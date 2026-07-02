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

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_DATASETS_CACHE=/home/lancewicki/data/hf_cache/datasets

if [ -f /home/lancewicki/projects/turing-rl/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /home/lancewicki/projects/turing-rl/.env
  set +a
fi

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
OUT_DIR=/home/lancewicki/projects/turing-rl/data/prism/history_smoke
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Build PRISM smoke slice (max_train_users=20)"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Out:   $OUT_DIR"
echo "============================================"

cd /home/lancewicki/projects/turing-rl

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
