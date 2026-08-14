#!/bin/bash
# Score trained judge checkpoints on the frozen 880-pair held-out set.
#
# Per arm: one CPU merge job (veRL shards -> validated dense model) then one 8-GPU sweep cell
# that serves it and judges all 880 pairs. The sweep cell is the SAME script that produced the
# zero-shot baselines, so the trained rows and the baseline rows are the same measurement:
# full 37-field ordered schema (PERSONA_JUDGE_JSON_SCHEMA=1), 8192 completion tokens, thinking
# on, and sampling from the model's own generation_config.json -- which merge_grpo_adapter.py
# copies verbatim from the container, so a merged 4B samples exactly like stock Qwen3.5-4B.
#
# The pair set is the step0 cell of the Aug-10 full-schema eval: fakes written by
# qwen35-9b-sft merged_ep3 -- the same generator our judge trained against -- over test10
# contexts, which are disjoint from the grpo60/train slice the judge saw.
#
# Baselines to compare against (zero-shot, thinking on, same pairs):
#   qwen35-4b 0.501 | qwen35-9b 0.518 | gemma4-12b 0.542 | gemma4-31b 0.593 | qwen35-27b 0.631*
#   (*n=313 of 880 -- that cell is incomplete, so gate any 27B comparison on a common subset.)
#
# Usage:
#   scripts/cluster_launch.sh --dependency-profile eval \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-4b-eval \
#     --env EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/<date>-judge-4b-eval \
#     scripts/launch_judge_eval.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2

EVAL_ROOT=${EVAL_ROOT:?set EVAL_ROOT (shared, readable by both jobs)}
ARMS=${ARMS:-"directional graded"}
STEP=${STEP:-52}
# Checkpoints land in the SHARED state-root results tree keyed only on the run tag, because a
# job's $REPO/results is a symlink to it -- not under the per-run root.
CKPT_ROOT=${CKPT_ROOT:-/home/lancewicki/projects/turing-rl/results/grpo/judge}
# Stock Qwen3.5-4B: the judge was trained from these weights, and this snapshot's
# config/tokenizer are the 4.x form the serving env can load (the checkpoint's own are 5.4).
CONTAINER=${CONTAINER:-/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a}
PAIRS=${PAIRS:-/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet}
# 4B is ~8GB bf16: fits one 40GB A100 with KV headroom, so TP=1 x 8 replicas (configs/
# judge_sweep_cells.py::tp_for_size). Same shape the zero-shot 4B baseline cell used.
TP=${TP:-1}
REPLICAS=${REPLICAS:-8}
# Hold 8 GPUs at a time, not 16: the user's own eval chain is competing for the same QOS
# allowance and already hit QOSMaxGRESPerUser.
SERIALIZE=${SERIALIZE:-1}
# One bf16 ULP near magnitude 1. An FSDP2 save re-rounds the 24 frozen
# `linear_attn.norm.weight` tensors of the recurrent layers by exactly this much, which the
# bit-exact form of check D reports as a corrupt backbone (merge jobs 17889/17891). They are
# provably untrained: the directional and graded checkpoints are BIT-IDENTICAL to each other
# on all 24 after 52 steps under different rewards. Anything larger still fails the gate.
SHARED_ATOL=${SHARED_ATOL:-0.00390625}
DRY=${DRY:-0}
# Hoisted out of the sbatch call on purpose: an inline $((...)) puts a ')' inside the
# invocation, which truncates the static "-- boundary present" guard in
# tests/test_cluster_workflow.py and reports a false offender.
GPUS=$((TP * REPLICAS))

[ -f "$PAIRS" ] || { echo "FATAL: pair set not found: $PAIRS" >&2; exit 2; }
mkdir -p "$EVAL_ROOT/models" "$EVAL_ROOT/raw/sweep"

echo "=== judge eval: arms='$ARMS' step=$STEP pairs=$PAIRS ==="
echo "=== container=$CONTAINER ==="

PREV_SWEEP=""
for arm in $ARMS; do
  case "$arm" in
    directional|graded) ;;
    *) echo "FATAL: arm must be directional or graded, got '$arm'" >&2; exit 2 ;;
  esac

  actor=$CKPT_ROOT/Qwen3.5-4B_${arm}_full/checkpoints/global_step_${STEP}/actor
  tag=judge-4b-${arm}-step${STEP}
  dense=$EVAL_ROOT/models/$tag/hf_dense

  merge_exports="ALL,STEP=$STEP,ACTOR_DIR=$actor,CONTAINER=$CONTAINER"
  merge_exports="$merge_exports,EVAL_ROOT=$EVAL_ROOT,MODEL_TAG=$tag,SHARED_ATOL=$SHARED_ATOL"
  sweep_exports="ALL,MODEL=$dense,TP=$TP,REPLICAS=$REPLICAS,THINKING_MODE=on"
  sweep_exports="$sweep_exports,CELL_NAME=$tag,PAIRS=$PAIRS,SWEEP_ROOT=$EVAL_ROOT/raw/sweep"

  if [ "$DRY" = "1" ]; then
    echo "[DRY] merge: $SBATCH --parsable --gres=gpu:0 --export=$merge_exports -- scripts/slurm/merge_grpo_ckpt.sh"
    echo "[DRY] sweep: afterok:<merge> --gres=gpu:$GPUS --export=$sweep_exports -- scripts/slurm/judge_sweep_cell.sh"
    PREV_SWEEP="dry"; continue
  fi

  [ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
  [ -d "$actor" ] || { echo "FATAL: no actor dir at $actor" >&2; exit 2; }

  merge_jid=$("$SBATCH" --parsable --gres=gpu:0 --job-name="jmerge_${arm}" \
      --export="$merge_exports" -- scripts/slurm/merge_grpo_ckpt.sh)
  case "$merge_jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for merge $arm (got '$merge_jid')" >&2; exit 1 ;;
  esac

  # afterok on the merge: the gate exits 5 and STILL leaves hf_dense on disk, so a dependency
  # on the directory existing would happily score an unvalidated model.
  dep="--dependency=afterok:$merge_jid"
  [ "$SERIALIZE" = "1" ] && [ -n "$PREV_SWEEP" ] && dep="$dep,afterany:$PREV_SWEEP"

  sweep_jid=$("$SBATCH" --parsable --gres=gpu:$GPUS --job-name="jeval_${arm}" \
      "$dep" --export="$sweep_exports" -- scripts/slurm/judge_sweep_cell.sh)
  case "$sweep_jid" in
    ''|*[!0-9]*) echo "FATAL: sbatch failed for sweep $arm (got '$sweep_jid'); cancelling merge $merge_jid" >&2
                 scancel "$merge_jid" 2>/dev/null; exit 1 ;;
  esac

  echo "submitted $arm: merge=$merge_jid sweep=$sweep_jid${PREV_SWEEP:+ (after $PREV_SWEEP)}"
  PREV_SWEEP=$sweep_jid
done
