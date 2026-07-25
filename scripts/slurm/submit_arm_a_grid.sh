#!/bin/bash
# Submit the 6-cell (KL x LR) Arm-A overfit grid on the PROPER (stop-token) checkpoint-78.
# Run from the CLUSTER repo root after sync_to_cluster.sh + Task 4 (merge). preflight-job-check first.
set -euo pipefail
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
MERGED=checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
# Explicit LoRA target (H2 parity) — set Arm A/B LoRA modules to match literally.
# On full-attn Qwen3-8B this is the same 7 modules that full-attention LoRA expands to.
TARGET='actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

# Emit "tag<TAB>overrides" per cell from the SSOT grid module.
$PY - <<'PYEOF' | while IFS=$'\t' read -r TAG OVR; do
from scripts.rl_grid import ARM_A_CELLS, cell_overrides
for c in ARM_A_CELLS:
    print(f"{c['tag']}\t{cell_overrides(c)}")
PYEOF
  echo ">> submitting $TAG :: $OVR $TARGET"
  JUDGE=9b MODE=overfit OVERFIT_EPOCHS=50 \
    RUN_TAG="$TAG" \
    MERGED_SFT_MODEL_PATH="$MERGED" \
    EXTRA_OVERRIDES="$OVR $TARGET" \
    sbatch --export=ALL scripts/slurm/rl_generator_run.sh
done
