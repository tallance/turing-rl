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
# Submit through scripts/cluster_launch.sh + scripts/submit_snapshot_job.sh with
# B0_ROLLOUT_SYNC=1 JUDGE=9b MODE=overfit OVERFIT_EPOCHS=8 RUN_TAG=9b_b0_spike.
#   JUDGE = 9b | 397b | gemma4-12b
#   MODE  = overfit | full | epoch1 | full5 | frac10ep10 | frac10ep20
#   full5 = full-dataset 5-epoch production run (325 steps; ckpt + validate every 32).
#   frac10ep10 / frac10ep20 = 10% of train (384 rows), 10 or 20 epochs (6 steps/epoch;
#                ckpt + validate every 6), validating on 50% of the val split.
#                Re-submitting frac10ep20 with a finished frac10ep10 RUN_TAG RESUMES it
#                (trainer.resume_mode=auto keys off default_local_dir); a fresh RUN_TAG
#                starts from the SFT init.
#   gemma4-12b judge serves from the CUDA-13 nightly env at a pinned snapshot; see
#                scripts/slurm/gemma4_judge_training_smoke.sh for the acceptance gates.
#   B0_ROLLOUT_SYNC=1 turns on the Step-3b rollout-sync hook (writes rollout_sync.json).
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO"

# wandb: source .env for WANDB_API_KEY + WANDB_BASE_URL, then force online + the
# self-hosted endpoint (same recipe as the working SFT runs — sft_full.sh/wandb_smoke.sh).
# Exported here so the trainer srun step inherits them (fixes 404 createRunFiles / no-sync).
if [ -f "$REPO/.env" ]; then set -a; source "$REPO/.env"; set +a; fi
# Point the Python secret loader at that same file explicitly.
#
# Sourcing it above only populates this shell. shared/load_env.py:get_openai_api_key calls
# load_local_env() BEFORE inspecting the process environment and raises if no .env FILE is
# found, so exporting OPENAI_API_KEY is not sufficient. Its default search is
# `Path(__file__).resolve().parents[1]/.env` -- and .resolve() follows the runtime view's
# symlinks back into the immutable source snapshot, which by design never contains secrets.
# `~/.env` does not exist on this cluster either, so both candidates miss.
#
# Job 18570 resumed correctly from global_step_60 and then died 9 min later when the first
# reward call reached resolve_judge_api_key(): "Missing OPENAI_API_KEY/OPENROUTER_API_KEY in
# .env file. Expected one of: .../turing-rl-sources/<sha>/.env". Ray workers inherit env
# vars, so exporting ENV_FILE here reaches the reward workers that actually raise.
if [ -z "${ENV_FILE:-}" ] && [ -f "$REPO/.env" ]; then
  export ENV_FILE="$REPO/.env"
  echo "=== judge secret file pinned: ENV_FILE=$ENV_FILE ==="
fi
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

JUDGE=${JUDGE:?set JUDGE=0.8b|9b|9b-ce|397b|gemma4-12b}
MODE=${MODE:?set MODE=overfit|full|epoch1|full5|frac10ep3|frac10ep10|frac10ep20}
case "$JUDGE" in 0.8b|9b|9b-ce|397b|gemma4-12b) ;; *) echo "bad JUDGE=$JUDGE" >&2; exit 2 ;; esac
case "$MODE" in overfit|full|epoch1|full5|frac10ep3|frac10ep10|frac10ep20) ;; *) echo "bad MODE=$MODE" >&2; exit 2 ;; esac
# Serving shape per judge. TP x DP is always 8 (one node): a model whose bf16 footprint fits
# one 40GB A100 with KV/CUDA-graph headroom runs TP=1 across 8 replicas for throughput,
# otherwise it spans the node at TP=8. Same rule configs/judge_sweep_cells.py:tp_for_size
# applies on the eval side -- gemma-4-12B is ~24GB bf16, so it gets the 8-replica shape.
# REASONING_PARSER is PINNED here, never inherited: the boundary detector is model-family
# specific (qwen3 vs gemma4) and a wrong one silently mis-splits thinking text out of
# .content, which the reward path would then fail to parse with nothing in the log saying why.
#
# 9b-ce is the CE-TRAINED 9B judge, served from a local merged checkpoint rather than an HF
# id -- the same weights eval cell judge-9b-ce-st scored (0.802 single-token accuracy on the
# frozen 880 pairs, against 0.541 for the zero-shot 9B; AUC 0.898, a_rate excess 0.009).
# Absolute path on purpose: the judge step runs under the runtime view, where results/ and
# checkpoints/ are symlinks, so a repo-relative path would depend on the child's cwd.
case "$JUDGE" in
  0.8b)      JUDGE_MODEL=Qwen/Qwen3.5-0.8B;                TP=1; DP=8; REASONING_PARSER=qwen3  ;;
  9b)        JUDGE_MODEL=Qwen/Qwen3.5-9B;                  TP=1; DP=8; REASONING_PARSER=qwen3  ;;
  9b-ce)     JUDGE_MODEL=/home/lancewicki/projects/turing-rl/checkpoints/sft/judge_qwen35_9b_ce_dense
             TP=1; DP=8; REASONING_PARSER=qwen3  ;;
  397b)      JUDGE_MODEL=Qwen/Qwen3.5-397B-A17B-GPTQ-Int4; TP=8; DP=1; REASONING_PARSER=qwen3  ;;
  gemma4-12b) JUDGE_MODEL=google/gemma-4-12B-it;           TP=1; DP=8; REASONING_PARSER=gemma4 ;;
esac

# Judge protocol: "full" (37-field JSON verdict) or "single_token" (one A/B token, verdict
# read from logprobs). Single-token decodes ONE token with thinking off, so there is no
# <think> block: clear the parser (judge_serve_9b_replicas.sh omits the flag when empty,
# matching judge_sweep_cell.sh, which only adds one for thinking-on cells) and turn thinking
# off for the reward path too. The scorer pins enable_thinking=False in code regardless, but
# leaving the env at 1 here would misreport what the run did.
JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE:-full}
case "$JUDGE_PROMPT_STYLE" in
  full) ;;
  single_token) REASONING_PARSER=""; PERSONA_JUDGE_ENABLE_THINKING=0 ;;
  *) echo "ERROR: JUDGE_PROMPT_STYLE must be full|single_token, got '$JUDGE_PROMPT_STYLE'" >&2
     exit 2 ;;
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
# parser quoted, not defaulted: '' is the legitimate single-token value, and a ${VAR:-...}
# here would be indistinguishable from reading REASONING_PARSER back from the environment,
# which this script must never do (it is pinned per JUDGE family above).
echo "=== judge serving pinned: model=$JUDGE_MODEL tp=$TP dp=$DP parser='$REASONING_PARSER' style=$JUDGE_PROMPT_STYLE ==="

# --- judge step on node0 (concurrent, backgrounded; frozen judge) ---
MODEL=$JUDGE_MODEL TP=$TP DP=$DP REASONING_PARSER=$REASONING_PARSER JUDGE_ENDPOINT_FILE=$ENDPOINT_FILE \
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
export JUDGE_PROMPT_STYLE
export JUDGE_MODEL
export OPENAI_API_BASE="$ENDPOINT"
export TURING_JUDGE_SCORE_CLIP_MAX=7
export PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'
# Default 1, which is what both completed arms (9B and gemma-4-12B) ran. Overridable because
# a judge small enough to never stop thinking is unusable with it on -- Qwen3.5-0.8B scored
# 0.175 usable with thinking, 0.840 without (Slurm 18901/18913, docs/judge-response-schema.md).
# A run that sets 0 is NOT protocol-comparable with the two arms above.
export PERSONA_JUDGE_ENABLE_THINKING="${PERSONA_JUDGE_ENABLE_THINKING:-1}"
export PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192
export PERSONA_JUDGE_DUMP_RATE=1.0
export PERSONA_REWARD_DUMP_DIR="$REWARD_DUMP_DIR"
export PERSONA_EVAL_JUDGE_MODEL="$JUDGE_MODEL"
# --- judge concurrency: PINNED, never inherited ---------------------------------------
# reward.py:_reward_judge_request_limit() checks TURING_JUDGE_MAX_CONCURRENCY *before*
# PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY, and sbatch --export=ALL propagates the submitting
# shell. So a stray ambient TURING_JUDGE_MAX_CONCURRENCY silently beats the value set here
# and leaves no trace in the repo. That is exactly what happened to job 13634: it inherited
# 8 (confirmed in its log: "[reward_judge] max concurrent requests per process: 8") and spent
# 41.5 of 44.1 h judge-bound at an effective concurrency of 6.3.
# Probe 13999 on this same DP-8 judge topology: throughput scales 6.4x from 8 -> 64 with FLAT
# latency (p50 116->124 s, p95 ~140 s), saturating at 64. See docs/default-params.md.
# Set JUDGE_CONC to change this deliberately; do NOT rely on ambient env.
JUDGE_CONC="${JUDGE_CONC:-64}"
export TURING_JUDGE_MAX_CONCURRENCY="$JUDGE_CONC"
export PERSONA_OPENAI_JUDGE_MAX_CONCURRENCY="$JUDGE_CONC"
# reward.py's own fallback is 400 s. At concurrency 64 the measured p95 is 140 s, but the
# 400 s default is what turned the job-13628 flood into a total-failure cascade, so keep the
# headroom: the cost of a long timeout is bounded, the cost of a false timeout is a lost pair.
export PERSONA_OPENAI_TIMEOUT_SECONDS="${PERSONA_OPENAI_TIMEOUT_SECONDS:-1800}"
echo "=== judge concurrency pinned: $JUDGE_CONC (timeout ${PERSONA_OPENAI_TIMEOUT_SECONDS}s) ==="
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
