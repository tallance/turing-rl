#!/bin/bash
# Score generator checkpoints with an explicitly selected trained Brier judge.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
CODE_ROOT=${TURING_RL_CODE_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=$CODE_ROOT/scripts/snapshot_sbatch.sh
PY=${PY:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}
cd "$REPO"

EVAL_ROOT=${EVAL_ROOT:-${TURING_RL_RUN_ROOT:?}}
SOURCE_EVAL_ROOT=${SOURCE_EVAL_ROOT:-/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema}
BASELINE_CELL_ROOT=${BASELINE_CELL_ROOT:?set BASELINE_CELL_ROOT to the selected judge completed step-0 cell}
MODEL=${MODEL:?set MODEL to the validated dense trained-judge directory}
STEPS=${STEPS:-"64 128 192 256 320"}
GEN_KEY_PREFIX=${GEN_KEY_PREFIX:-9b-full5ep-step}
PAIRS_TAG=${PAIRS_TAG:-880}
CELL_NAME=${CELL_NAME:-judge-9b-brier}
JOB_PREFIX=${JOB_PREFIX:-te_brier}
# Thinking mode defines the comparison. A judge trained thinking-OFF must be scored OFF, and
# every model in one trajectory must share the mode. judge_sweep_cell.sh writes to
# $CELL_NAME/$THINKING_MODE, so the two families coexist rather than overwrite each other.
THINKING_MODE=${THINKING_MODE:-on}
CONFIRM_THINKING_OFF=${CONFIRM_THINKING_OFF:-0}
TP=${TP:-1}
REPLICAS=${REPLICAS:-8}
CONCURRENCY=${CONCURRENCY:-32}
DRY=${DRY:-0}

case "$THINKING_MODE" in
  on) ;;
  off)
    [ "$CONFIRM_THINKING_OFF" = "1" ] || {
      echo "FATAL: thinking-off evaluation requires CONFIRM_THINKING_OFF=1" >&2
      exit 2
    }
    case "$EVAL_ROOT" in
      /*thinking-off*) ;;
      *) echo "FATAL: a thinking-off EVAL_ROOT must contain 'thinking-off': $EVAL_ROOT" >&2; exit 2 ;;
    esac
    ;;
  *) echo "FATAL: THINKING_MODE must be on|off, got '$THINKING_MODE'" >&2; exit 2 ;;
esac

[ -d "$MODEL" ] || { echo "FATAL: missing trained judge model: $MODEL" >&2; exit 2; }
[ -d "$BASELINE_CELL_ROOT/reward" ] || {
  echo "FATAL: missing reused step-0 reward dir: $BASELINE_CELL_ROOT/reward" >&2
  exit 2
}
# The reused step-0 cell must have been scored in THIS mode. judge_sweep_cell.sh names the leaf
# directory after the mode, so a mismatch is mechanically detectable -- and worth detecting: an
# ON step-0 grafted onto an OFF trajectory makes the whole curve cross-mode at its baseline.
baseline_mode=$(basename "$BASELINE_CELL_ROOT")
[ "$baseline_mode" = "$THINKING_MODE" ] || {
  echo "FATAL: cross-mode baseline: BASELINE_CELL_ROOT is '$baseline_mode', THINKING_MODE=$THINKING_MODE" >&2
  exit 2
}

read -r -a step_values <<< "$STEPS"
[ "${#step_values[@]}" -gt 0 ] || { echo "FATAL: STEPS is empty" >&2; exit 2; }
for step in 0 "${step_values[@]}"; do
  case "$step" in ''|*[!0-9]*) echo "FATAL: invalid step '$step'" >&2; exit 2 ;; esac
  pairs=$SOURCE_EVAL_ROOT/raw/pairs/gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet
  [ -f "$pairs" ] || { echo "FATAL: missing pair set: $pairs" >&2; exit 2; }
done

# Claim the root before copying the baseline or submitting the first job. This prevents two
# concurrent invocations from appending duplicate records while both chains are still pending.
claim=$EVAL_ROOT/provenance/brier_trajectory_submission.claim
mkdir -p "$EVAL_ROOT/provenance"
if ! mkdir "$claim" 2>/dev/null; then
  echo "FATAL: EVAL_ROOT is already claimed for submission: $claim" >&2
  exit 2
fi
printf 'thinking_mode=%s\nsource_sha=%s\nclaimed_at_utc=%s\n' \
  "$THINKING_MODE" "${TURING_RL_SOURCE_SHA:-unknown}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "$claim/metadata.txt"

# Materialize the already-completed step-0 cell and all six pair sets in this run root. The
# operation is idempotent only when every existing destination is byte-identical; a mixed or
# partially copied baseline is rejected rather than silently reused.
export EVAL_ROOT SOURCE_EVAL_ROOT BASELINE_CELL_ROOT GEN_KEY_PREFIX PAIRS_TAG CELL_NAME STEPS
export THINKING_MODE
"$PY" - <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_digest(paths: list[Path], root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p.relative_to(root))):
        rel = str(path.relative_to(root)).encode()
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        h.update(bytes.fromhex(digest(path)))
    return h.hexdigest()


def copy_exact(source: Path, destination: Path) -> None:
    if destination.exists():
        if not destination.is_file() or digest(source) != digest(destination):
            raise SystemExit(f"FATAL: existing destination differs from source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


eval_root = Path(os.environ["EVAL_ROOT"])
source_eval = Path(os.environ["SOURCE_EVAL_ROOT"])
baseline = Path(os.environ["BASELINE_CELL_ROOT"])
prefix = os.environ["GEN_KEY_PREFIX"]
tag = os.environ["PAIRS_TAG"]
cell = os.environ["CELL_NAME"]
steps = [0, *map(int, os.environ["STEPS"].split())]

for step in steps:
    name = f"gen_{prefix}{step}_{tag}.parquet"
    copy_exact(source_eval / "raw" / "pairs" / name, eval_root / "raw" / "pairs" / name)

split_guard = source_eval / "split_guard.json"
if split_guard.is_file():
    copy_exact(split_guard, eval_root / "split_guard.json")

destination = eval_root / "raw" / f"{prefix}0" / "sweep" / cell / os.environ["THINKING_MODE"]
source_files = [baseline / "run_metadata.json", *sorted((baseline / "reward").glob("*.jsonl"))]
if not source_files[0].is_file() or len(source_files) == 1:
    raise SystemExit(f"FATAL: incomplete reused step-0 cell: {baseline}")
for source in source_files:
    rel = source.relative_to(baseline)
    copy_exact(source, destination / rel)

metadata = json.loads((baseline / "run_metadata.json").read_text())
record = {
    "format_version": 1,
    "source_cell_root": str(baseline),
    "source_slurm_job_id": str(metadata.get("slurm_job_id", "")),
    "source_tree_sha256": tree_digest(source_files, baseline),
    "pair_source": str(source_eval / "raw" / "pairs" / f"gen_{prefix}0_{tag}.parquet"),
    "pair_sha256": digest(source_eval / "raw" / "pairs" / f"gen_{prefix}0_{tag}.parquet"),
    "destination_cell_root": str(destination),
}
provenance = eval_root / "provenance" / "baseline_reuse.json"
provenance.parent.mkdir(parents=True, exist_ok=True)
encoded = json.dumps(record, indent=2, sort_keys=True) + "\n"
if provenance.exists() and provenance.read_text() != encoded:
    raise SystemExit(f"FATAL: existing baseline provenance differs: {provenance}")
provenance.write_text(encoded)
PY

submit() {
  local dep="$1"; shift
  local deparg=""
  [ -n "$dep" ] && deparg="--dependency=$dep"
  if [ "$DRY" = "1" ]; then
    echo "[DRY] sbatch $deparg $*" >&2
    echo "dry$RANDOM"
  else
    "$SBATCH" --parsable $deparg "$@"
  fi
}

need_jid() {
  if [ "$DRY" = "1" ]; then [ -n "$1" ] && return 0; fi
  case "$1" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for $2 (got '$1')" >&2; exit 1 ;;
  esac
}

previous=""
for step in "${step_values[@]}"; do
  pairs=$EVAL_ROOT/raw/pairs/gen_${GEN_KEY_PREFIX}${step}_${PAIRS_TAG}.parquet
  sweep_root=$EVAL_ROOT/raw/${GEN_KEY_PREFIX}${step}/sweep
  reward_dir=$sweep_root/$CELL_NAME/$THINKING_MODE/reward
  if [ -d "$reward_dir" ] && [ -n "$(find "$reward_dir" -maxdepth 1 -type f -print -quit)" ]; then
    echo "FATAL: refusing stale output in $reward_dir" >&2
    exit 2
  fi

  dep=""
  [ -n "$previous" ] && dep="afterok:$previous"
  exports="ALL,MODEL=$MODEL,TP=$TP,REPLICAS=$REPLICAS,CONCURRENCY=$CONCURRENCY"
  exports="$exports,THINKING_MODE=$THINKING_MODE,CELL_NAME=$CELL_NAME,PAIRS=$pairs,SWEEP_ROOT=$sweep_root"
  jid=$(submit "$dep" --gres=gpu:$((TP * REPLICAS)) --job-name="${JOB_PREFIX}_${step}" \
    --export="$exports" -- scripts/slurm/judge_sweep_cell.sh)
  need_jid "$jid" "Brier judge step$step"
  echo "submitted step$step -> $jid${previous:+ (after $previous)}"
  previous=$jid
done

echo "chain tail job: $previous"
