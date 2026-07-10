#!/bin/bash
#SBATCH --job-name=sft_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sft_full-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# DEPRECATED: single-process launch (single GPU, OOMs on 40GB for bf16 8192-seq). Use sft_variant.sh (torchrun, 8-GPU).

# Full 8-GPU SFT on the PRISM full CoT-annotated slice.
# Paper Table 5 LoRA config (r=64, alpha=128, bfloat16 / no QLoRA) from
# qwen3_8b_lora.yaml, with step-checkpointing + auto-resume.
# Output: checkpoints/sft/qwen3_8b_prism_full_s42/

set -uo pipefail

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

# Source any .env (HF_TOKEN if needed, etc.)
if [ -f /home/lancewicki/projects/turing-rl/.env ]; then
  set -a
  # shellcheck disable=SC1091
  source /home/lancewicki/projects/turing-rl/.env
  set +a
fi

export HF_HUB_DISABLE_XET=1
export HF_HOME=/home/lancewicki/data/hf_cache
export HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export PYTHONUNBUFFERED=1
# Enable wandb logging. lora_sft.py reads `report_to` from the YAML,
# so we temporarily patch the YAML to `wandb`, run, then restore. (No CLI flag exists.)
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-sft
export WANDB_RUN_GROUP=sft-full
# WANDB_API_KEY + WANDB_BASE_URL come from .env above.
# Tame bnb / accelerate warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
DATA=$REPO/data/sft/prism_full_s42_sft_cot.jsonl
OUT=$REPO/checkpoints/sft/qwen3_8b_prism_full_s42
YAML=$REPO/training/sft/configs/qwen3_8b_lora.yaml
YAML_BAK=${YAML}.full-bak

mkdir -p "$OUT"

restore_yaml() {
  if [ -f "$YAML_BAK" ]; then
    mv -f "$YAML_BAK" "$YAML"
    echo "[cleanup] restored $YAML"
  fi
}
trap restore_yaml EXIT

# Snapshot YAML, swap report_to: none -> report_to: wandb.
cp -p "$YAML" "$YAML_BAK"
sed -i 's/^report_to:.*/report_to: wandb/' "$YAML"
echo "[patched] $(grep -E '^report_to:' "$YAML")"

echo "============================================"
echo "SFT full (QLoRA, 8 GPU, step-checkpointing + auto-resume)"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
echo "Data:    $DATA"
echo "Output:  $OUT"
echo "Wandb:   $WANDB_BASE_URL  project=$WANDB_PROJECT"
echo "============================================"

[ -f "$DATA" ] || { echo "ERROR: SFT JSONL not found at $DATA"; exit 2; }

cd "$REPO"

$PY -u -m training.sft.lora_sft --model qwen3-8b \
    --data_path $REPO/data/sft/prism_full_s42_sft_cot.jsonl \
    --output_dir $REPO/checkpoints/sft/qwen3_8b_prism_full_s42 \
    --max_seq_length 8192 --resume_from_checkpoint auto
RC=$?

echo ""
echo "============================================"
echo "SFT full exit: $RC"
echo "Date:          $(date)"
if [ $RC -eq 0 ]; then
  echo ""
  echo "=== output dir contents ==="
  ls -la "$OUT" 2>/dev/null
  echo ""
  echo "=== adapter files (look for adapter_model.safetensors) ==="
  find "$OUT" -maxdepth 3 -type f \( -name 'adapter_model.safetensors' -o -name 'adapter_config.json' -o -name 'trainer_state.json' \) -printf '%p  (%s bytes)\n'
fi
echo "============================================"
exit $RC
