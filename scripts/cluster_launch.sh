#!/bin/bash
exec python3 "$(cd "$(dirname "$0")/.." && pwd)/scripts/cluster_launch.py" "$@"
