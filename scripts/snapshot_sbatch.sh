#!/bin/bash
exec /usr/bin/python3 "${TURING_RL_CODE_ROOT:?}/scripts/snapshot_sbatch.py" "$@"
