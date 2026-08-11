#!/bin/bash
#SBATCH --job-name=prism_full_s42
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/build_prism_full_s42-%j.out
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
OUT_DIR=$TURING_RL_DATA_ROOT/prism/full_s42_history
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Build PRISM FULL (seed=42, history mode, no caps)"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Out:   $OUT_DIR"
echo "============================================"

cd "$TURING_RL_WORK_ROOT"

$PY -u -m data.prism.build \
  --output      "$OUT_DIR/train.parquet" \
  --val_output  "$OUT_DIR/val.parquet" \
  --test_output "$OUT_DIR/test.parquet" \
  --conditioning_mode history \
  --shuffle_rows
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
        n_users = df['user_id'].nunique() if 'user_id' in df.columns else -1
        print(f'{name:5s} rows={len(df):>6d}  users={n_users:>5d}')
    except Exception as e:
        print(f'{name:5s} read failed: {e}')
"
fi

echo "============================================"
echo "Done at $(date) (rc=$RC)"
echo "============================================"
exit $RC
