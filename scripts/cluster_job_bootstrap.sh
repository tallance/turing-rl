#!/bin/bash
# Source this once near the top of every maintained Slurm script.

source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_runtime.sh"
turing_rl_prepare_runtime "job-${SLURM_JOB_ID:?}" || exit $?
