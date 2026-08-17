#!/bin/bash
#SBATCH --job-name=te_t10t50_continue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:0
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --partition=a100
#SBATCH --account=rfai
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/te-t10t50-continue-%j.out

set -euo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

: "${NEXT_PHASE:?NEXT_PHASE is required}"
: "${NEXT_OFFSET:?NEXT_OFFSET is required}"
PHASE=$NEXT_PHASE OFFSET=$NEXT_OFFSET bash scripts/launch_frac10_test50_eval.sh
