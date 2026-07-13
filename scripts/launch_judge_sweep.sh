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
REPO=/home/lancewicki/projects/turing-rl
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
FAMILY=${FAMILY:?set FAMILY (qwen3 or qwen3.5)}
CALIBRATION=${CALIBRATION:-0}
cd "$REPO"

CELLS=$($PY -c "import json; from configs.judge_sweep_cells import cell_list; print(json.dumps(cell_list('$FAMILY')))")

echo "$CELLS" | CALIBRATION="$CALIBRATION" $PY -c '
import json, sys, os, subprocess
cells = json.load(sys.stdin)
calib = os.environ.get("CALIBRATION") == "1"
for c in cells:
    for mode in ("off", "on"):
        exp = (f"MODEL={c[\"model_id\"]},TP={c[\"tp\"]},REPLICAS={c[\"replicas\"]},"
               f"THINKING_MODE={mode},CELL_NAME={c[\"cell_name\"]}")
        if calib:
            exp += ",MAX_PAIRS=50"
        subprocess.run(
            ["sbatch", "--parsable",
             f"--gres=gpu:{c[\"tp\"] * c[\"replicas\"]}",
             f"--job-name=sw_{c[\"cell_name\"]}_{mode}",
             "--export=ALL," + exp,
             "scripts/slurm/judge_sweep_cell.sh"],
            check=True,
        )
        print("submitted", c["cell_name"], mode, flush=True)
'
