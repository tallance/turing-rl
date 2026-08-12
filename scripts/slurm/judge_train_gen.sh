#!/bin/bash
#SBATCH --job-name=judge_gen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/judge_gen-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
#
# Generate the fake turns for judge training, then build the both-orders pair parquet.
# Sampling matches how the frozen 880 eval pairs were produced, so train and eval pairs
# are distribution-matched.
set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache
export HF_HUB_DISABLE_XET=1 PYTHONUNBUFFERED=1
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

REPO=${TURING_RL_WORK_ROOT:?}
cd "$REPO" || exit 2
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
SPLIT=${SPLIT:-train}
SLICE_LO=${SLICE_LO:-0.0}
SLICE_HI=${SLICE_HI:-0.1}
LIMIT=${LIMIT:-416}
GEN_NUM=${GEN_NUM:-4}
OUT_DIR=${OUT_DIR:-$REPO/data/prism/judge/iter1}

DATA_BASE=$TURING_RL_INPUT_DATA_ROOT/prism/full_s42_history_sft40_grpo60_test10/grpo
case "$SPLIT" in
  train) SOURCE_PARQUET=${SOURCE_PARQUET:-$DATA_BASE/train.parquet} ;;
  val)   SOURCE_PARQUET=${SOURCE_PARQUET:-$DATA_BASE/val.parquet}; SLICE_LO=0.0; SLICE_HI=1.0; LIMIT=0; GEN_NUM=1 ;;
  *) echo "ERROR: SPLIT must be train or val, got $SPLIT" >&2; exit 2 ;;
esac

# Generation sampling: Qwen3.5 model card = job 13634 val_kwargs = how the 880 eval pairs were made.
export GEN_TEMPERATURE=${GEN_TEMPERATURE:-0.7}
export GEN_TOP_P=${GEN_TOP_P:-0.8}
export GEN_TOP_K=${GEN_TOP_K:-20}
export GEN_MAX_TOKENS=${GEN_MAX_TOKENS:-1024}

PKL=$OUT_DIR/raw/${SPLIT}_generations.pkl
mkdir -p "$OUT_DIR/raw"

echo "=== judge gen: split=$SPLIT slice=[$SLICE_LO,$SLICE_HI) limit=$LIMIT k=$GEN_NUM ==="
echo "=== model=$MERGED_EP3 sampling T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K ==="

$PY -u -m eval.generate_trained --base_model --model_id "$MERGED_EP3" \
  --test_parquet "$SOURCE_PARQUET" --output "$PKL" --gen_num "$GEN_NUM" \
  --temperature "$GEN_TEMPERATURE" --top_p "$GEN_TOP_P" --top_k "$GEN_TOP_K" \
  --max_tokens "$GEN_MAX_TOKENS" --backend vllm \
  --vllm_max_model_len "${GEN_MAX_MODEL_LEN:-13524}" \
  --vllm_truncate_prompt_tokens "${GEN_TRUNCATE_PROMPT_TOKENS:-12500}" || exit 3

LIMIT_ARG=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARG=(--limit "$LIMIT")
$PY -u scripts/build_judge_train_pairs.py \
  --inference_pkl "$PKL" --source_parquet "$SOURCE_PARQUET" \
  --out "$OUT_DIR/$SPLIT.parquet" \
  --slice_lo "$SLICE_LO" --slice_hi "$SLICE_HI" --split "$SPLIT" "${LIMIT_ARG[@]}"
