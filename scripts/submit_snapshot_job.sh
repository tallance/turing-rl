#!/bin/bash
set -euo pipefail
[ "${1:-}" != "--" ] || shift
exec "${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.sh" "$@"
