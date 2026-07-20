#!/bin/bash
#SBATCH --job-name=sft_variant
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sft_variant-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# Parameterized multi-GPU (torchrun, 8-GPU) SFT launcher. Select the recipe via
# VARIANT env: qlora_r64 | bf16_fsdp | bf16_fa2. Each variant differs only by CLI
# flags — the committed qwen3_8b_lora.yaml stays read-only (no concurrent-sed race).
#   VARIANT=bf16_fsdp sbatch scripts/slurm/sft_variant.sh
# SMOKE=1 does a fast config check (--exit_after_trainer_build --max_train_examples 64).

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

REPO=/home/lancewicki/projects/turing-rl

# Source any .env (WANDB creds, HF_TOKEN, etc.)
if [ -f "$REPO/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO/.env"
  set +a
fi

export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-sft
export WANDB_RUN_GROUP=sft-variants
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
DATA=$REPO/data/sft/prism_full_s42_sft_cot.jsonl

VARIANT=${VARIANT:?set VARIANT=qlora_r64|bf16_fsdp|bf16_fa2}
SMOKE=${SMOKE:-0}
# NOPACK=1 disables trl sequence packing. Under sdpa, packing=True leaks attention
# across packed conversations (trl delegates isolation to FlashAttention varlen, which
# sdpa lacks). --no_packing gives clean per-conversation attention. Writes to a distinct
# "_nopack" output dir so it never resumes a packed run. DELIBERATE deviation from
# upstream (which packs) — documented in our_patches.md.
NOPACK=${NOPACK:-0}

MODEL=${MODEL:-qwen3-8b}   # qwen3-8b | qwen35-9b
# Per-model: output stem, python env, and the FSDP auto-wrap decoder class.
# qwen3.5 needs its own transformers-5.x env (model_type=qwen3_5 unsupported by the 4.57.6
# in turing-rl-train) and a different decoder class (both verified via probe_qwen35.py).
case "$MODEL" in
  qwen3-8b)
    STEM=qwen3_8b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
    FSDP_LAYER_CLS=Qwen3DecoderLayer ;;
  qwen35-9b)
    STEM=qwen35_9b
    PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python
    FSDP_LAYER_CLS=Qwen3_5DecoderLayer ;;
  *) echo "bad MODEL=$MODEL"; exit 2 ;;
esac

case "$VARIANT" in
  qlora_r64)
    OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_qlora_r64
    export WANDB_NAME=sft-qlora-r64
    ;;
  bf16_fsdp)
    OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_bf16_fsdp
    export WANDB_NAME=sft-bf16-fsdp
    ;;
  bf16_fa2)
    OUT=$REPO/checkpoints/sft/${STEM}_prism_full_s42_bf16_fa2
    export WANDB_NAME=sft-bf16-fa2
    ;;
  *)
    echo "bad VARIANT=$VARIANT (expected qlora_r64|bf16_fsdp|bf16_fa2)"; exit 2 ;;
esac

if [ "$NOPACK" = "1" ]; then
  OUT="${OUT}_nopack"
  export WANDB_NAME="${WANDB_NAME}-nopack"
fi

# Build the arg list as an array so the quoted multi-word FSDP value survives intact.
ARGS=(--model "$MODEL" --data_path "$DATA" --output_dir "$OUT" --max_seq_length 8192
      --resume_from_checkpoint auto --report_to wandb --no_torch_compile)

case "$VARIANT" in
  qlora_r64) ARGS+=(--force_qlora --attn_implementation sdpa) ;;
  # bf16 variants pass --no_qlora explicitly so the recipe is self-describing and
  # robust to yaml drift (4-bit bnb + FSDP full_shard is a broken combo).
  bf16_fsdp) ARGS+=(--no_qlora --attn_implementation sdpa --fsdp "full_shard auto_wrap" --fsdp_transformer_layer_cls "$FSDP_LAYER_CLS") ;;
  bf16_fa2)  ARGS+=(--no_qlora --attn_implementation flash_attention_2) ;;
esac

[ "$SMOKE" = "1" ] && ARGS+=(--exit_after_trainer_build --max_train_examples 64)
[ "$NOPACK" = "1" ] && ARGS+=(--no_packing)

mkdir -p "$OUT"
[ -f "$DATA" ] || { echo "ERROR: missing $DATA"; exit 2; }
cd "$REPO"

echo "============================================"
echo "SFT variant (torchrun, 8-GPU)"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
echo "VARIANT: $VARIANT"
echo "Output:  $OUT"
echo "Smoke:   $SMOKE"
echo "NoPack:  $NOPACK"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
echo "============================================"

$PY -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 \
    -m training.sft.lora_sft "${ARGS[@]}"
RC=$?

echo ""
echo "============================================"
echo "SFT variant $VARIANT exit: $RC"
echo "Date:                     $(date)"
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== output dir contents ==="
  ls -la "$OUT" 2>/dev/null
  echo ""
  echo "=== adapter files ==="
  find "$OUT" -maxdepth 3 -name 'adapter_model.safetensors' -printf '%p (%s bytes)\n'
fi
echo "============================================"
exit $RC
