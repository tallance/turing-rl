#!/bin/bash
#SBATCH --job-name=te_eval_continue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=a100
#SBATCH --account=rfai
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/te-eval-continue-%j.out

set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${REPO:-/home/lancewicki/projects/turing-rl}
: "${NEXT_OFFSET:?NEXT_OFFSET is required}"
cd "$REPO"
OFFSET=$NEXT_OFFSET bash scripts/launch_full_schema_eval.sh
