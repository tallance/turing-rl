#!/bin/bash
set -euo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_runtime.sh"
turing_rl_prepare_runtime "launcher-$$"
exec "$@"
