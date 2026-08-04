#!/bin/bash
#SBATCH --job-name=merge_grpo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:0
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/merge_grpo-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Stage 0 of the test-set eval: turn a veRL GRPO checkpoint into a dense servable model.
#
#   1. verl.model_merger  -> hf_base/ (reconstructed SFT backbone) + hf_base/lora_adapter/
#   2. merge_grpo_adapter -> hf_dense/ (merged_ep3 container + W + 0.5*B@A on 128 targets)
#   3. validate_grpo_merge -> HARD GATE; nonzero exit means do NOT generate with this model
#
# CPU-only (gpu:0): pure tensor math, but ~19 GB of shards are read and ~18 GB written,
# which is far too heavy for the login node.
#
# Required env: STEP (e.g. 8)   Optional: EVAL_ROOT, RUN_TAG, DISTINCT_FROM
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PYTHONUNBUFFERED=1
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache

REPO=/home/lancewicki/projects/turing-rl
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

STEP=${STEP:?set STEP (GRPO global_step to merge, e.g. 8)}
RUN_TAG=${RUN_TAG:-9b_half_kl1e4_lr1e4_temp1}
EVAL_ROOT=${EVAL_ROOT:-$REPO/results/2026-08-03-test-eval-9b-half}
MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
DISTINCT_FROM=${DISTINCT_FROM:-}

# verl.model_merger must run in the Arm-B env: the checkpoint config is transformers 5.4.
PY_MERGE=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python
# The dense build + gate are pure safetensors math, so the generation env is fine (and
# proves the artifacts are readable by the env that will actually serve them).
PY_EVAL=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

ACTOR=$REPO/results/grpo/rl-generator/$RUN_TAG/checkpoints/global_step_${STEP}/actor
OUT=$EVAL_ROOT/models/step${STEP}
HF_BASE=$OUT/hf_base
HF_DENSE=$OUT/hf_dense

[ -d "$ACTOR" ] || { echo "ERROR: no actor dir at $ACTOR" >&2; exit 2; }
[ -d "$MERGED_EP3" ] || { echo "ERROR: no merged_ep3 at $MERGED_EP3" >&2; exit 2; }
mkdir -p "$OUT"

echo "=== merge_grpo_ckpt: STEP=$STEP RUN_TAG=$RUN_TAG ==="
echo "    actor=$ACTOR"
echo "    out=$OUT   host=$(hostname)   date=$(date)"

if [ -d "$HF_BASE" ] && [ -n "$(ls -A "$HF_BASE" 2>/dev/null)" ]; then
  echo "--- step 1: hf_base already exists, skipping model_merger ---"
else
  echo "--- step 1: verl.model_merger (FSDP2 shards -> hf_base + lora_adapter) ---"
  $PY_MERGE -m verl.model_merger merge --backend fsdp \
      --local_dir "$ACTOR" --target_dir "$HF_BASE" || { echo "FAIL: model_merger" >&2; exit 3; }
fi
[ -f "$HF_BASE/lora_adapter/adapter_model.safetensors" ] || {
  echo "ERROR: no lora_adapter under $HF_BASE -- checkpoint was not an unmerged LoRA?" >&2; exit 3; }

echo "--- step 2: fold the GRPO delta into the merged_ep3 container -> hf_dense ---"
rm -rf "$HF_DENSE"
$PY_EVAL scripts/merge_grpo_adapter.py \
    --base "$MERGED_EP3" --adapter "$HF_BASE/lora_adapter" --out "$HF_DENSE" \
    || { echo "FAIL: merge_grpo_adapter" >&2; exit 4; }

echo "--- step 3: HARD GATE (scripts/validate_grpo_merge.py) ---"
GATE=(--base "$MERGED_EP3" --dense "$HF_DENSE" --adapter "$HF_BASE/lora_adapter" --hf_base "$HF_BASE")
[ -n "$DISTINCT_FROM" ] && GATE+=(--distinct_from "$DISTINCT_FROM")
$PY_EVAL scripts/validate_grpo_merge.py "${GATE[@]}"
RC=$?
if [ $RC -ne 0 ]; then
  echo "=== GATE FAILED (rc=$RC): do NOT run generation with $HF_DENSE ===" >&2
  exit 5
fi

du -sh "$HF_BASE" "$HF_DENSE" 2>/dev/null
echo "=== OK: $HF_DENSE is a validated dense GRPO policy for step $STEP ==="
