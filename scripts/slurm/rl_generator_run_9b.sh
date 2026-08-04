#!/bin/bash
#SBATCH --job-name=rl_gen
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=7-00:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/rl_gen-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Arm-B: single ATOMIC 2-node 9B GRPO run for the RL-generator-vs-fixed-judge probe.
# Slurm allocates BOTH nodes together (no idle-hold). node0 = DP judge
# (judge_serve_9b_replicas.sh, frozen 9B judge — UNCHANGED from the 8B driver),
# node1 = veRL 9B trainer (rl_generator_train_9b.sh). Endpoint handed off via a
# shared file on FSx home; the judge srun step is killed when the trainer srun
# step finishes.
#
# Submit: B0_ROLLOUT_SYNC=1 JUDGE=9b MODE=overfit OVERFIT_EPOCHS=8 RUN_TAG=9b_b0_spike \
#           sbatch --export=ALL scripts/slurm/rl_generator_run_9b.sh
#   JUDGE = 9b | 397b     MODE = overfit | full | epoch1
#   B0_ROLLOUT_SYNC=1 turns on the Step-3b rollout-sync hook (writes rollout_sync.json).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"

# wandb: source .env for WANDB_API_KEY + WANDB_BASE_URL, then force online + the
# self-hosted endpoint (same recipe as the working SFT runs — sft_full.sh/wandb_smoke.sh).
# Exported here so the trainer srun step inherits them (fixes 404 createRunFiles / no-sync).
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://meta.wandb.io}"
export WANDB_MODE=online
# Keep the wandb run dir OFF FSx. Job 13634 completed all 32 steps, yet wandb kept only 30 train
# points and 4 of 5 validations: the FSx wobble that killed the job also stalled wandb's writer,
# so the tail never reached even the LOCAL transaction log (two `wandb sync` runs recovered
# nothing). Node-local tmpfs is immune to that; cleanup() below syncs it and copies it back.
export WANDB_DIR="${WANDB_DIR:-/tmp/wandb-${SLURM_JOB_ID:-$$}}"
mkdir -p "$WANDB_DIR"
# Arm-B trainer env (same one rl_generator_train_9b.sh runs in), used for the exit-time sync.
WANDB_BIN=${WANDB_BIN:-/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/wandb}

JUDGE=${JUDGE:?set JUDGE=9b|397b}
MODE=${MODE:?set MODE=overfit|full|epoch1}
case "$JUDGE" in 9b|397b) ;; *) echo "bad JUDGE=$JUDGE" >&2; exit 2 ;; esac
case "$MODE" in overfit|full|epoch1) ;; *) echo "bad MODE=$MODE" >&2; exit 2 ;; esac
case "$JUDGE" in
  9b)   JUDGE_MODEL=Qwen/Qwen3.5-9B;                  TP=1; DP=8 ;;
  397b) JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4; TP=8; DP=1 ;;
esac

# Two allocated nodes: node0 -> judge, node1 -> trainer.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[ "${#NODES[@]}" -ge 2 ] || { echo "ERROR: need 2 nodes, got '${NODES[*]:-none}'" >&2; exit 2; }
NODE_JUDGE=${NODES[0]}; NODE_TRAIN=${NODES[1]}

RUN_TAG=${RUN_TAG:-${JUDGE}_${MODE}_merged_sft_ref}
RUN_DIR=$REPO/results/grpo/rl-generator/$RUN_TAG
ENDPOINT_FILE=$RUN_DIR/judge_endpoint.txt
REWARD_DUMP_DIR=$RUN_DIR/reward_dump
CKPT_DIR=$RUN_DIR/checkpoints
mkdir -p "$RUN_DIR" "$REWARD_DUMP_DIR" "$CKPT_DIR" "$REPO/logs"
rm -f "$ENDPOINT_FILE"

echo ">> RL-gen atomic 9B run: JUDGE=$JUDGE MODEL=$JUDGE_MODEL MODE=$MODE job=$SLURM_JOB_ID"
echo ">> nodes: judge=$NODE_JUDGE trainer=$NODE_TRAIN  run_dir=$RUN_DIR"

# --- judge step on node0 (concurrent, backgrounded; frozen 9B judge, unchanged) ---
MODEL=$JUDGE_MODEL TP=$TP DP=$DP JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
  srun --nodes=1 --ntasks=1 --nodelist="$NODE_JUDGE" --gres=gpu:8 --overlap \
  bash scripts/slurm/judge_serve_9b_replicas.sh &
JUDGE_PID=$!
# Preserve wandb before the node-local dir vanishes with the job, then push whatever the run
# never managed to upload. Slurm sends SIGTERM before SIGKILL, so this usually gets to run;
# the copy happens first so the transaction log survives even if the sync itself is killed.
save_wandb() {
  [ -d "$WANDB_DIR" ] || return 0
  cp -r "$WANDB_DIR" "$REPO/wandb/joblocal-${SLURM_JOB_ID:-$$}" 2>/dev/null || true
  for d in "$WANDB_DIR"/run-*; do
    [ -d "$d" ] || continue
    timeout 600 "$WANDB_BIN" sync "$d" 2>&1 | tail -2 || true
  done
}
cleanup() { save_wandb; kill "$JUDGE_PID" 2>/dev/null || true; }
trap cleanup EXIT TERM INT

# --- wait for the judge to publish its endpoint (written only after model-verified health) ---
echo ">> waiting for judge endpoint (up to 60 min warmup)..."
ok=0
for t in $(seq 1 1800); do
  [ -s "$ENDPOINT_FILE" ] && { ok=1; break; }
  kill -0 "$JUDGE_PID" 2>/dev/null || { echo "ERROR: judge step died before publishing endpoint" >&2; exit 3; }
  sleep 2
done
[ $ok -eq 1 ] || { echo "TIMEOUT waiting for judge endpoint" >&2; exit 4; }
ENDPOINT=$(cat "$ENDPOINT_FILE")
echo ">> judge endpoint: $ENDPOINT"

# --- reward env for the trainer step (inherited via srun --export=ALL / default) ---
export REWARD_METRIC=turing
export JUDGE_MODEL
export OPENAI_API_BASE="$ENDPOINT"
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
export PERSONA_JUDGE_ENABLE_THINKING=1
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_JUDGE_DUMP_RATE=1.0
export PERSONA_REWARD_DUMP_DIR="$REWARD_DUMP_DIR"
export PERSONA_EVAL_JUDGE_MODEL="$JUDGE_MODEL"
export PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY="${PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY:-128}"
export PERSONA_OPENAI_MAX_RETRIES="${PERSONA_OPENAI_MAX_RETRIES:-3}"
export WANDB_PROJECT="${WANDB_PROJECT:-2026-07-15-rl-generator-vs-fixed-judge}"
export MERGED_SFT_MODEL_PATH="${MERGED_SFT_MODEL_PATH:-checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}"
export EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"   # extra Hydra overrides (e.g. kl_loss_coef) -> trainer step
export RL_MODE="$MODE" RL_JUDGE="$JUDGE" RL_RUN_TAG="$RUN_TAG" RL_RUN_DIR="$RUN_DIR" RL_CKPT_DIR="$CKPT_DIR"
export RL_JUDGE_JOB_ID=""   # no separate judge job; teardown handled here by killing the judge srun step
# B0_ROLLOUT_SYNC (if set on submission) is inherited by the trainer step and enables the
# Step-3b rollout-sync instrumentation hook (writes $RL_RUN_DIR/rollout_sync.json).
export B0_ROLLOUT_SYNC="${B0_ROLLOUT_SYNC:-}"

# --- trainer step on node1 (foreground) — 9B variant ---
# Run the trainer script from node-local disk, NOT from FSx. bash reads a script LAZILY, holding
# the file open and re-reading as it executes, so a multi-day job keeps an FSx handle alive for
# its whole life. Job 13634 died that way after completing all 32 steps:
#   scripts/slurm/rl_generator_train_9b.sh: error reading input file: Stale file handle
# Copying it once removes that entire failure mode.
# The copy must happen ON the trainer node: /tmp is node-local, and this launcher runs on a
# different node than $NODE_TRAIN. bash -c receives its program as a string (already in memory),
# so nothing is re-read from FSx once the copy is done.
srun --nodes=1 --ntasks=1 --nodelist="$NODE_TRAIN" --gres=gpu:8 --overlap \
  bash -c 'L=/tmp/rl_gen_train-${SLURM_JOB_ID:-$$}.sh; cp "$0" "$L" || exit 2; exec bash "$L"' \
  "$REPO/scripts/slurm/rl_generator_train_9b.sh"
RC=$?
echo "=== trainer step exit: $RC ; tearing down judge step ==="
kill "$JUDGE_PID" 2>/dev/null || true
exit $RC
