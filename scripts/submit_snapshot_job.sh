#!/bin/bash
set -euo pipefail
exec "${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh" "$@"
