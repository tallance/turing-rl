#!/bin/bash
# Config-driven judge-sweep launcher. Reads the cell list from the SSOT
# (configs/judge_sweep_cells.py) and submits one sbatch per (cell, thinking-mode).
#
#   FAMILY=qwen3.5 bash scripts/launch_judge_sweep.sh                # full sweep (880 pairs)
#   FAMILY=qwen3.5 CALIBRATION=1 bash scripts/launch_judge_sweep.sh  # 50-pair calibration
#
# FAMILY is set by the Task-17 family decision. CALIBRATION=1 caps each cell at
# 50 pairs (for the per-cell wall-time / >4h gate).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
FAMILY=${FAMILY:?set FAMILY (qwen3 or qwen3.5)}
CALIBRATION=${CALIBRATION:-0}
cd "$REPO"

# One "cell_name model_id tp replicas" line per cell (incl. the 397B anchor).
CELLS=$($PY -c "
from configs.judge_sweep_cells import cell_list
for c in cell_list('$FAMILY'):
    print(c['cell_name'], c['model_id'], c['tp'], c['replicas'])
")

EXTRA=""
[ "$CALIBRATION" = "1" ] && EXTRA=",MAX_PAIRS=50"

while read -r cell_name model_id tp replicas; do
  [ -z "$cell_name" ] && continue
  gpus=$((tp * replicas))
  for mode in off on; do
    jid=$("$SBATCH" --parsable --gres=gpu:$gpus \
      --job-name=sw_${cell_name}_${mode} \
      --export=ALL,MODEL=$model_id,TP=$tp,REPLICAS=$replicas,THINKING_MODE=$mode,CELL_NAME=$cell_name$EXTRA \
      -- \
      scripts/slurm/judge_sweep_cell.sh)
    echo "submitted $cell_name $mode -> job $jid (gpu:$gpus)"
  done
done <<< "$CELLS"
