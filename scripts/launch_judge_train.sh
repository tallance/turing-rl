#!/bin/bash
# Submit a judge GRPO run (R0 overfit gate, or a full R1 arm) through the snapshot gateway.
#
# scripts/slurm/judge_grpo_train.sh is a JOB script; cluster_launch.sh runs its argument on
# the login node, so it needs this launcher. See launch_judge_pairs.sh for the same split.
#
# MODE=overfit builds a tiny side-balanced subset first and pins the batch to fit it. That is
# R0: it proves the loop learns before six real runs are committed. Train accuracy should
# climb toward 1.0 on a handful of examples; if it does not, nothing downstream is worth
# launching. Note it also points data.val_files at that training subset, so its "validation"
# numbers are training numbers -- use MODE=valsmoke when you need a real held-out reading.
#
# MODE=valsmoke keeps a real held-out val slice (default 100 pairs = 200 rows) and runs a single
# step with validation before and after. It is the ~20-minute way to compare two configurations
# -- e.g. thinking ON vs OFF -- on format and accuracy before spending ~9h per full arm.
#
# Usage:
#   scripts/cluster_launch.sh --dependency-profile training \
#     --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-r0 \
#     --env JUDGE_MODEL_PATH=Qwen/Qwen3.5-4B --env JUDGE_REWARD_ARM=directional \
#     --env MODE=overfit --env RL_CKPT_DIR=<run-root>/checkpoints \
#     scripts/launch_judge_train.sh
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=${TURING_RL_WORK_ROOT:?run via scripts/cluster_launch.sh}
SBATCH=${TURING_RL_CODE_ROOT:+$TURING_RL_CODE_ROOT/scripts/snapshot_sbatch.sh}
cd "$REPO" || exit 2
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:?set JUDGE_MODEL_PATH, e.g. Qwen/Qwen3.5-4B}
JUDGE_REWARD_ARM=${JUDGE_REWARD_ARM:?set JUDGE_REWARD_ARM to directional or graded}
MODE=${MODE:-full}
OVERFIT_PAIRS=${OVERFIT_PAIRS:-8}
# $REPO/data is the source snapshot inside a job; generated data lives in the state root.
DATA_DIR=${DATA_DIR:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1}
DRY=${DRY:-0}

case "$JUDGE_REWARD_ARM" in
  directional|graded) ;;
  *) echo "FATAL: JUDGE_REWARD_ARM must be directional or graded, got '$JUDGE_REWARD_ARM'" >&2; exit 2 ;;
esac

TRAIN_FILE=$DATA_DIR/train.parquet
VAL_FILE=$DATA_DIR/val.parquet
EXTRA=${EXTRA_OVERRIDES:-}

case "$MODE" in
  full|overfit|valsmoke) ;;
  # Without this, a typo'd MODE falls through to `full` and quietly starts a ~9h run in place of
  # a 20-minute check.
  *) echo "FATAL: MODE must be full, overfit or valsmoke, got '$MODE'" >&2; exit 2 ;;
esac

if [ "$MODE" = overfit ]; then
  TRAIN_FILE=$DATA_DIR/train_overfit${OVERFIT_PAIRS}.parquet
  $PY scripts/build_judge_overfit.py --src "$DATA_DIR/train.parquet" \
      --out "$TRAIN_FILE" --n_pairs "$OVERFIT_PAIRS" || exit 2
  rows=$((OVERFIT_PAIRS * 2))
  # The batch must fit the subset, and train_batch_size * rollout.n must stay divisible by
  # the agent-loop worker count (preflight 17/26): rows*4 / 8 workers.
  EXTRA="$EXTRA data.train_batch_size=$rows actor_rollout_ref.actor.ppo_mini_batch_size=$rows"
  EXTRA="$EXTRA trainer.total_epochs=${OVERFIT_EPOCHS:-30} trainer.save_freq=-1"
  # Validating a 16-row overfit against the full 1410-row val set every epoch would dwarf the
  # run itself; R0 only asks whether TRAIN accuracy saturates.
  EXTRA="$EXTRA data.val_files=$TRAIN_FILE"
  echo "=== R0 overfit gate: $rows rows ($OVERFIT_PAIRS pairs x 2 orders) ==="
fi

if [ "$MODE" = valsmoke ]; then
  # A cheap check of the VALIDATION path -- used to A/B thinking ON vs OFF before committing to
  # six ~9h runs. Distinct from overfit precisely because overfit points data.val_files at its
  # own 16-row training subset: reusing it here would "validate" on the training rows and report
  # nothing about held-out format or accuracy.
  VALSMOKE_PAIRS=${VALSMOKE_PAIRS:-100}
  VALSMOKE_TRAIN_PAIRS=${VALSMOKE_TRAIN_PAIRS:-8}
  # --select longest, NOT the default 'first'. A valsmoke exists to predict a full run, and
  # peak training memory is set by the LONGEST sequence, not a typical one. The leading pairs
  # top out ~3000 tokens short of the corpus maximum, which is exactly the margin that decides
  # whether log_softmax fits: three full arms were launched on a budget a 'first' smoke passed
  # and all three OOMed at that site.
  TRAIN_FILE=$DATA_DIR/train_longest${VALSMOKE_TRAIN_PAIRS}.parquet
  $PY scripts/build_judge_overfit.py --src "$DATA_DIR/train.parquet" \
      --out "$TRAIN_FILE" --n_pairs "$VALSMOKE_TRAIN_PAIRS" --select longest || exit 2
  # Same pair-wise, side-balanced slicer, applied to val: taking raw head rows could land an
  # odd number and unbalance which slot holds the human.
  VAL_FILE=$DATA_DIR/val_smoke${VALSMOKE_PAIRS}.parquet
  $PY scripts/build_judge_overfit.py --src "$DATA_DIR/val.parquet" \
      --out "$VAL_FILE" --n_pairs "$VALSMOKE_PAIRS" || exit 2
  rows=$((VALSMOKE_TRAIN_PAIRS * 2))
  EXTRA="$EXTRA data.train_batch_size=$rows actor_rollout_ref.actor.ppo_mini_batch_size=$rows"
  EXTRA="$EXTRA trainer.total_epochs=1 trainer.save_freq=-1"
  # val_before_train gives the step-0 reading, which is the whole point: it measures the base
  # model under the real rollout path before any gradient step can confound it.
  EXTRA="$EXTRA trainer.val_before_train=True trainer.test_freq=1"
  echo "=== validation smoke: $((VALSMOKE_PAIRS * 2)) val rows, $rows train rows ==="
fi

RUN_TAG=${JUDGE_RUN_TAG:-$(basename "$JUDGE_MODEL_PATH")_${JUDGE_REWARD_ARM}_${MODE}}
EXPORTS="ALL,JUDGE_MODEL_PATH=$JUDGE_MODEL_PATH,JUDGE_REWARD_ARM=$JUDGE_REWARD_ARM"
EXPORTS="$EXPORTS,TRAIN_FILE=$TRAIN_FILE,VAL_FILE=$VAL_FILE,JUDGE_RUN_TAG=$RUN_TAG"

echo "=== judge train: model=$JUDGE_MODEL_PATH arm=$JUDGE_REWARD_ARM mode=$MODE tag=$RUN_TAG ==="
echo "=== train=$TRAIN_FILE ==="
echo "=== extra overrides: ${EXTRA:-<none>} ==="

# Chain an arm behind another run, e.g. JUDGE_DEPENDENCY=afterok:12345 to start the 0/1 arm
# only once the graded arm of the SAME model size has succeeded. afterok, not afterany: if
# graded dies on an OOM the directional run would die the same way, and burning a second
# 8-GPU allocation to rediscover that helps nobody. Same unquoted-empty-string idiom
# launch_judge_format_probe.sh uses for its serve/probe chain.
DEP=""; [ -n "${JUDGE_DEPENDENCY:-}" ] && DEP="--dependency=${JUDGE_DEPENDENCY}"

if [ "$DRY" = "1" ]; then
  echo "[DRY] $SBATCH --parsable $DEP --export=$EXPORTS -- scripts/slurm/judge_grpo_train.sh"
  exit 0
fi

[ -n "$SBATCH" ] || { echo "FATAL: run through scripts/cluster_launch.sh" >&2; exit 2; }
# EXTRA_OVERRIDES contains SPACES. Slurm's --export list is comma-delimited and does not
# survive embedded whitespace cleanly, so it is exported into the environment and carried by
# the leading `ALL` instead — the same rule launch_generator_sweep.sh documents for
# PERSONA_JUDGE_SAMPLING's embedded commas.
export EXTRA_OVERRIDES="$EXTRA"
# shellcheck disable=SC2086  # $DEP is empty or one --dependency=... token, by construction
jid=$("$SBATCH" --parsable $DEP --export="$EXPORTS" -- scripts/slurm/judge_grpo_train.sh)
case "$jid" in
  ''|*[!0-9]*) echo "FATAL: sbatch failed (got '$jid')" >&2; exit 1 ;;
esac
echo "submitted judge train job=$jid"
