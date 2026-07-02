#!/bin/bash
#SBATCH --job-name=sft_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/sft_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

# 1-GPU SFT smoke on the PRISM smoke slice (138 rows, CoT-annotated).
# Uses QLoRA (default in qwen3_8b_lora.yaml). Output: results/sft/smoke/
# Goal: prove the SFT trainer constructs, loss decreases, an adapter is saved.

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
# Enable wandb logging for the smoke. lora_sft.py reads `report_to` from the YAML,
# so we temporarily patch the YAML to `wandb`, run, then restore. (No CLI flag exists.)
export WANDB_MODE=online
export WANDB_PROJECT=turing-rl-smoke
export WANDB_RUN_GROUP=sft-smoke
# WANDB_API_KEY + WANDB_BASE_URL come from .env above.
# Tame bnb / accelerate warnings
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
DATA=$REPO/data/sft/qwen3-8b_prism_smoke_sft_cot.jsonl
OUT=$REPO/results/sft/smoke
YAML=$REPO/training/sft/configs/qwen3_8b_lora.yaml
YAML_BAK=${YAML}.smoke-bak

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
echo "SFT smoke (QLoRA, 1 epoch, 138 rows)"
echo "Date:    $(date)"
echo "Host:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader | head -1
echo "Data:    $DATA"
echo "Output:  $OUT"
echo "Wandb:   $WANDB_BASE_URL  project=$WANDB_PROJECT"
echo "============================================"

[ -f "$DATA" ] || { echo "ERROR: SFT JSONL not found at $DATA"; exit 2; }

cd "$REPO"

$PY -m training.sft.lora_sft \
  --model qwen3-8b \
  --data_path "$DATA" \
  --output_dir "$OUT" \
  --num_epochs 1 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 8192 \
  --no_torch_compile
RC=$?

echo ""
echo "============================================"
echo "SFT smoke exit: $RC"
echo "Date:           $(date)"
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
