#!/bin/bash
# Build a writable per-process view over immutable source and mutable state.

turing_rl_prepare_runtime() {
  : "${TURING_RL_CODE_ROOT:?TURING_RL_CODE_ROOT is required}"
  : "${TURING_RL_STATE_ROOT:?TURING_RL_STATE_ROOT is required}"
  : "${TURING_RL_SOURCE_SHA:?TURING_RL_SOURCE_SHA is required}"
  : "${TURING_RL_RUN_CLASS:?TURING_RL_RUN_CLASS is required}"
  : "${TURING_RL_RUN_ROOT:?TURING_RL_RUN_ROOT is required}"
  : "${TURING_RL_DEPENDENCY_PROFILE:?TURING_RL_DEPENDENCY_PROFILE is required}"

  [ "$TURING_RL_CODE_ROOT" != "$TURING_RL_STATE_ROOT" ] || {
    echo "FATAL: immutable code root and mutable state root must differ" >&2
    return 2
  }
  [ -r "$TURING_RL_CODE_ROOT/SOURCE_MANIFEST.json" ] || {
    echo "FATAL: source manifest missing from $TURING_RL_CODE_ROOT" >&2
    return 2
  }
  /usr/bin/python3 - "$TURING_RL_CODE_ROOT/SOURCE_MANIFEST.json" "$TURING_RL_SOURCE_SHA" <<'PY' || return 2
import json, sys
manifest = json.load(open(sys.argv[1]))
if manifest.get("source_sha") != sys.argv[2]:
    raise SystemExit(f"FATAL: source manifest says {manifest.get('source_sha')}, env says {sys.argv[2]}")
PY
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$TURING_RL_CODE_ROOT" /usr/bin/python3 - \
      "$TURING_RL_CODE_ROOT" <<'PY' || return 2
import json, sys
from pathlib import Path
from scripts.cluster_workflow import SOURCE_MANIFEST
from scripts.publish_cluster_source import verify_extracted_tree

root = Path(sys.argv[1])
verify_extracted_tree(root, json.loads((root / SOURCE_MANIFEST).read_text()))
PY

  local runtime_id=${1:-${SLURM_JOB_ID:-launcher}-$$}
  runtime_id="${runtime_id}-${TURING_RL_SOURCE_SHA:0:12}"
  case "$runtime_id" in *[!A-Za-z0-9_.-]*) echo "FATAL: unsafe runtime id: $runtime_id" >&2; return 2 ;; esac
  local work_root="$TURING_RL_RUN_ROOT/work/$runtime_id"
  mkdir -p "$TURING_RL_RUN_ROOT/work" "$TURING_RL_RUN_ROOT/hydra/$runtime_id" \
    "$TURING_RL_RUN_ROOT/provenance/jobs"
  if ! mkdir "$work_root"; then
    echo "FATAL: runtime work directory already exists: $work_root" >&2
    return 2
  fi

  local entry name target
  for entry in "$TURING_RL_CODE_ROOT"/* "$TURING_RL_CODE_ROOT"/.[!.]* \
      "$TURING_RL_CODE_ROOT"/..?*; do
    [ -e "$entry" ] || [ -L "$entry" ] || continue
    name=${entry##*/}
    case "$name" in
      SOURCE_MANIFEST.json|data|results|logs|checkpoints|wandb|outputs|.env) continue ;;
    esac
    target="$work_root/$name"
    [ -e "$target" ] || [ -L "$target" ] || ln -s "$entry" "$target"
  done

  local mutable_root="$TURING_RL_STATE_ROOT"
  [ "$TURING_RL_RUN_CLASS" != "debug" ] || mutable_root="$TURING_RL_RUN_ROOT"
  for name in results logs checkpoints wandb; do
    mkdir -p "$mutable_root/$name"
    target="$work_root/$name"
    [ -e "$target" ] || [ -L "$target" ] || ln -s "$mutable_root/$name" "$target"
  done
  # `data/` contains tracked Python utilities, canonical input datasets, and generated datasets.
  # Keep code in the source view and make payload intent explicit in maintained launchers.
  [ -e "$work_root/data" ] || ln -s "$TURING_RL_CODE_ROOT/data" "$work_root/data"
  export TURING_RL_INPUT_DATA_ROOT="$TURING_RL_STATE_ROOT/data"
  export TURING_RL_GENERATED_DATA_ROOT="$mutable_root/data"
  # Compatibility only. New code must choose INPUT_DATA_ROOT or GENERATED_DATA_ROOT.
  export TURING_RL_DATA_ROOT="$TURING_RL_GENERATED_DATA_ROOT"
  mkdir -p "$TURING_RL_INPUT_DATA_ROOT" "$TURING_RL_GENERATED_DATA_ROOT"
  [ ! -f "$TURING_RL_STATE_ROOT/.env" ] || ln -s "$TURING_RL_STATE_ROOT/.env" "$work_root/.env" 2>/dev/null || true
  [ -e "$work_root/outputs" ] || ln -s "$TURING_RL_RUN_ROOT/hydra/$runtime_id" "$work_root/outputs"

  export TURING_RL_WORK_ROOT="$work_root"
  export TURING_RL_HYDRA_DIR="$TURING_RL_RUN_ROOT/hydra/$runtime_id"
  export REPO="$work_root"
  export PYTHONPATH="$TURING_RL_CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  cd "$work_root" || return 2

  if [ -n "${SLURM_JOB_ID:-}" ]; then
    local job_provenance="$TURING_RL_RUN_ROOT/provenance/jobs/$SLURM_JOB_ID"
    mkdir -p "$job_provenance"
    /usr/bin/python3 "$TURING_RL_CODE_ROOT/scripts/record_runtime_manifest.py" \
      --compare "$TURING_RL_RUN_ROOT/provenance/expected_runtime.json" \
      --out "$job_provenance/runtime.json" || return 2
  fi
}
