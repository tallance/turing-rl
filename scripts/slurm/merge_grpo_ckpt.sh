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
#
#   Pass --export=ALL,STEP=8 and this script through scripts/submit_snapshot_job.sh.
#
# A gate failure exits 5 and hf_dense STILL EXISTS on disk, so callers must gate on the job's
# exit status, never on the directory being present.
#
# End-to-end runbook: docs/test-set-eval.md
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export PYTHONUNBUFFERED=1
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

STEP=${STEP:?set STEP (GRPO global_step to merge, e.g. 8)}
RUN_TAG=${RUN_TAG:-9b_half_kl1e4_lr1e4_temp1}
EVAL_ROOT=${EVAL_ROOT:-$REPO/results/2026-08-03-test-eval-9b-half}
# The container supplies config/tokenizer/chat-template and every NON-target tensor; only the
# LoRA targets are updated. It must be a directory the SERVING env can load: the checkpoint's
# own config/tokenizer are written by transformers 5.4 and do not load under the 4.57.6
# generation env. For a 9B generator that container is merged_ep3; for a judge trained from
# stock weights it is the stock HF snapshot. MERGED_EP3 stays honoured for existing callers.
CONTAINER=${CONTAINER:-${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}}
# Both Qwen3.5-9B and -4B are 32 layers at full_attention_interval=4, so both have
# 32*3 MLP + 8*4 attn = 128 LoRA targets. Overridable because that coincidence is not a law.
EXPECT_TARGETS=${EXPECT_TARGETS:-128}
# Check D bit-exactness tolerance on SHARED tensors. Stays 0 for the 9B generator path.
SHARED_ATOL=${SHARED_ATOL:-0}
DISTINCT_FROM=${DISTINCT_FROM:-}

# verl.model_merger must run in the Arm-B env: the checkpoint config is transformers 5.4.
PY_MERGE=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python
# The dense build + gate are pure safetensors math, so the generation env is fine (and
# proves the artifacts are readable by the env that will actually serve them).
PY_EVAL=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

ACTOR=${ACTOR_DIR:-$REPO/results/grpo/rl-generator/$RUN_TAG/checkpoints/global_step_${STEP}/actor}
# MODEL_TAG names the output dir. It is NOT step${STEP} by default for judges: two arms merged
# at the same step would otherwise write to one directory and silently overwrite each other.
OUT=$EVAL_ROOT/models/${MODEL_TAG:-step${STEP}}
HF_BASE=$OUT/hf_base
HF_DENSE=$OUT/hf_dense

[ -d "$ACTOR" ] || { echo "ERROR: no actor dir at $ACTOR" >&2; exit 2; }
[ -d "$CONTAINER" ] || { echo "ERROR: no container model at $CONTAINER" >&2; exit 2; }
mkdir -p "$OUT"

echo "=== merge_grpo_ckpt: STEP=$STEP RUN_TAG=$RUN_TAG ==="
echo "    actor=$ACTOR"
echo "    out=$OUT   host=$(hostname)   date=$(date)"

# ALWAYS rebuild hf_base. Reusing a nonempty hf_base would be unsafe in a way the gate
# CANNOT detect: the gate proves hf_dense == merged_ep3 + 0.5*B@A, but the adapter comes
# from hf_base, so a stale hf_base (wrong step / wrong run tag) is internally self-consistent
# and passes every check while the model is mislabeled. Rebuilding costs ~35s.
echo "--- step 1: verl.model_merger (FSDP2 shards -> hf_base + lora_adapter) ---"
rm -rf "$HF_BASE"
$PY_MERGE -m verl.model_merger merge --backend fsdp \
    --local_dir "$ACTOR" --target_dir "$HF_BASE" || { echo "FAIL: model_merger" >&2; exit 3; }
[ -f "$HF_BASE/lora_adapter/adapter_model.safetensors" ] || {
  echo "ERROR: no lora_adapter under $HF_BASE -- checkpoint was not an unmerged LoRA?" >&2; exit 3; }

# Record which actor these artifacts came from, so the provenance README and any later
# audit can prove step N was built from step N's shards.
$PY_EVAL - "$ACTOR" "$HF_BASE" "$STEP" "$RUN_TAG" <<'PROV'
import hashlib, json, os, sys
actor, hf_base, step, run_tag = sys.argv[1:5]
shards = sorted(f for f in os.listdir(actor) if f.startswith("model_world_size_"))
meta = [{"name": n, "size": os.stat(os.path.join(actor, n)).st_size} for n in shards]

# Fingerprint the EXTRACTED ADAPTER, not the shard prefix. The LoRA base is frozen, so shard
# names/sizes and their leading bytes are identical across steps -- an earlier version hashed
# those and produced the same value for step 8 and step 16, i.e. it could not detect the very
# mixup it existed to catch. The adapter is the only step-specific content.
adapter = os.path.join(hf_base, "lora_adapter", "adapter_model.safetensors")
h = hashlib.sha256()
with open(adapter, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 24), b""):
        h.update(chunk)
adapter_fp = h.hexdigest()

json.dump({"actor_dir": actor, "step": int(step), "run_tag": run_tag,
           "n_shards": len(shards), "shards": meta,
           "adapter_sha256": adapter_fp,
           "adapter_bytes": os.path.getsize(adapter)},
          open(os.path.join(hf_base, "merge_provenance.json"), "w"), indent=2)
print(f"provenance: step={step} run_tag={run_tag} shards={len(shards)} adapter_sha256={adapter_fp[:16]}")
PROV

echo "--- step 2: fold the GRPO delta into the container -> hf_dense ---"
rm -rf "$HF_DENSE"
$PY_EVAL scripts/merge_grpo_adapter.py \
    --base "$CONTAINER" --adapter "$HF_BASE/lora_adapter" --out "$HF_DENSE" \
    --expect_targets "$EXPECT_TARGETS" \
    || { echo "FAIL: merge_grpo_adapter" >&2; exit 4; }

echo "--- step 3: HARD GATE (scripts/validate_grpo_merge.py) ---"
GATE=(--base "$CONTAINER" --dense "$HF_DENSE" --adapter "$HF_BASE/lora_adapter" --hf_base "$HF_BASE")
GATE+=(--expect_targets "$EXPECT_TARGETS" --shared_atol "$SHARED_ATOL")
[ -n "$DISTINCT_FROM" ] && GATE+=(--distinct_from "$DISTINCT_FROM")
$PY_EVAL scripts/validate_grpo_merge.py "${GATE[@]}"
RC=$?
if [ $RC -ne 0 ]; then
  echo "=== GATE FAILED (rc=$RC): do NOT run generation with $HF_DENSE ===" >&2
  exit 5
fi

du -sh "$HF_BASE" "$HF_DENSE" 2>/dev/null
echo "=== OK: $HF_DENSE is a validated dense GRPO policy for step $STEP ==="
