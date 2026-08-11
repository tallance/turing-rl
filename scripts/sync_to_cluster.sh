#!/bin/bash
set -u
cat >&2 <<'EOF'
ERROR: mutable cluster checkout deployment has been retired.

Publish and launch immutable committed source with:
  scripts/cluster_launch.sh --run-root /home/lancewicki/projects/turing-rl/results/<run> \
    scripts/<launcher>.sh

Use --debug --label <label> and a results/debug/<label>/ run root for a disposable
debug commit. Dirty and partial deployments are intentionally unsupported.
EOF
exit 2
