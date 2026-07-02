#!/bin/bash
#SBATCH --job-name=dl_prism
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/download_prism-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_DATASETS_CACHE=/home/lancewicki/data/hf_cache/datasets

# Read HF_TOKEN from .env so the gated dataset can be authenticated
if [ -f /home/lancewicki/projects/turing-rl/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /home/lancewicki/projects/turing-rl/.env
  set +a
fi

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

echo "============================================"
echo "Prime PRISM cache: HannahRoseKirk/prism-alignment (conversations / train)"
echo "Date:  $(date)"
echo "Host:  $(hostname)"
echo "Cache: $HF_DATASETS_CACHE"
echo "Token set: $([ -n "${HF_TOKEN:-}" ] && echo yes || echo NO)"
echo "============================================"

cd /home/lancewicki/projects/turing-rl

$PY -c "
import os
from datasets import load_dataset
ds = load_dataset(
    'HannahRoseKirk/prism-alignment',
    'conversations',
    split='train',
    token=os.environ.get('HF_TOKEN'),
)
print(f'rows: {len(ds)}')
print(f'columns: {ds.column_names}')
print('first row keys:', list(ds[0].keys())[:10])
"
RC=$?

echo "============================================"
echo "Done at $(date) (rc=$RC)"
echo "============================================"
exit $RC
