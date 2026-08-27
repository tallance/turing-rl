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
#
# REUSE_GENERATIONS=1 (with REUSE_RAW_DIR=<dir>) skips the slice and generation steps and
# builds from generations that already exist. Use it when only the pair RENDERING changes
# (a different PROMPT_STYLE over the same turns): resampling would give the new pair set
# different fake turns from the ones earlier judges trained on, and cost a GPU node for it.
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
# Overridable only so tests/test_judge_reuse_generations.py can run this script for real
# against a recording stub and observe which steps it invokes. Unset on the cluster.
PY=${TURING_RL_JOB_PYTHON:-/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python}

MERGED_EP3=${MERGED_EP3:-$REPO/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
SPLIT=${SPLIT:-train}
SLICE_LO=${SLICE_LO:-0.0}
SLICE_HI=${SLICE_HI:-0.1}
LIMIT=${LIMIT:-416}
GEN_NUM=${GEN_NUM:-4}
OUT_DIR=${OUT_DIR:-${TURING_RL_GENERATED_DATA_ROOT:?}/prism/judge/iter1}
PROMPT_STYLE=${PROMPT_STYLE:-full}

# Opt-in: re-render an EXISTING set of generations under a different --prompt-style instead
# of sampling new ones. Default 0 keeps the three-step slice/generate/build path byte-identical.
REUSE_GENERATIONS=${REUSE_GENERATIONS:-0}
case "$REUSE_GENERATIONS" in
  0|1) ;;
  *) echo "ERROR: REUSE_GENERATIONS must be 0 or 1, got '$REUSE_GENERATIONS'" >&2; exit 2 ;;
esac

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

# Two independent directories, derived separately on purpose. Generation inputs/outputs live
# in a raw/ dir; the builder's parquet lands in OUT_DIR. In generate mode the raw dir sits
# under this run's own OUT_DIR. In reuse mode it does NOT: a single_token build writes to
# .../prism/judge/iter1/single_token while the generations it reuses were written by the
# full-schema build to .../prism/judge/iter1/raw. Neither path is derivable from the other,
# so reuse makes the caller name the raw dir rather than guessing a parent prefix.
if [ "$REUSE_GENERATIONS" = 1 ]; then
  if [ -z "${REUSE_RAW_DIR:-}" ]; then
    echo "ERROR: REUSE_GENERATIONS=1 requires REUSE_RAW_DIR (the dir holding" >&2
    echo "       ${SPLIT}_generations.pkl and ${SPLIT}_source_slice.parquet). It is not" >&2
    echo "       derived from OUT_DIR=$OUT_DIR: a style-nested OUT_DIR has no raw/ of its own." >&2
    exit 2
  fi
  RAW_DIR=$REUSE_RAW_DIR
else
  RAW_DIR=$OUT_DIR/raw
fi
PKL=$RAW_DIR/${SPLIT}_generations.pkl
SLICED_PARQUET=$RAW_DIR/${SPLIT}_source_slice.parquet
mkdir -p "$OUT_DIR"
[ "$REUSE_GENERATIONS" = 1 ] || mkdir -p "$RAW_DIR"

echo "=== judge gen: split=$SPLIT slice=[$SLICE_LO,$SLICE_HI) limit=$LIMIT k=$GEN_NUM style=$PROMPT_STYLE ==="
echo "=== model=$MERGED_EP3 sampling T=$GEN_TEMPERATURE top_p=$GEN_TOP_P top_k=$GEN_TOP_K ==="
if [ "$REUSE_GENERATIONS" = 1 ]; then
  echo "=== generations: REUSE existing (slice and GPU generation skipped) ==="
else
  echo "=== generations: GENERATE fresh (slice, then vLLM sampling) ==="
fi
echo "=== generations dir: $RAW_DIR ==="
echo "===   pickle:        $PKL ==="
echo "===   sliced source: $SLICED_PARQUET ==="
echo "=== builder out dir: $OUT_DIR ==="

LIMIT_ARG=()
[ "$LIMIT" -gt 0 ] && LIMIT_ARG=(--limit "$LIMIT")

if [ "$REUSE_GENERATIONS" = 1 ]; then
  # Missing inputs are fatal rather than a fallback to generation. Generation samples at
  # temperature 0.7, so a silent fallback would hand this pair set DIFFERENT fake turns from
  # the ones the already-trained judges were built on -- the confound reuse exists to avoid --
  # and would burn a GPU node doing it.
  rc=0
  [ -f "$PKL" ] || { echo "ERROR: REUSE_GENERATIONS=1 but no generations pickle at $PKL" >&2; rc=3; }
  [ -f "$SLICED_PARQUET" ] || { echo "ERROR: REUSE_GENERATIONS=1 but no sliced source at $SLICED_PARQUET" >&2; rc=3; }
  [ "$rc" -eq 0 ] || exit "$rc"
else
  # Slice BEFORE generating. Handing the full split to generate_trained would sample k
  # generations for every context and then throw ~90% away in the builder -- ~16.7k
  # generations to keep ~1.7k, in a 12h single-GPU job whose pickle is written only at the
  # end. select_slice is a pure function of extra_info and idempotent, so the same bounds go
  # to both steps and the selected rows are identical to the post-hoc path.
  $PY -u scripts/slice_judge_source.py \
    --source_parquet "$SOURCE_PARQUET" --out "$SLICED_PARQUET" \
    --slice_lo "$SLICE_LO" --slice_hi "$SLICE_HI" "${LIMIT_ARG[@]}" || exit 3

  $PY -u -m eval.generate_trained --base_model --model_id "$MERGED_EP3" \
    --test_parquet "$SLICED_PARQUET" --output "$PKL" --gen_num "$GEN_NUM" \
    --temperature "$GEN_TEMPERATURE" --top_p "$GEN_TOP_P" --top_k "$GEN_TOP_K" \
    --max_tokens "$GEN_MAX_TOKENS" --backend vllm \
    --vllm_max_model_len "${GEN_MAX_MODEL_LEN:-13524}" \
    --vllm_truncate_prompt_tokens "${GEN_TRUNCATE_PROMPT_TOKENS:-12500}" || exit 3
fi

# Same file, same bounds: select_slice on an already-sliced frame is a no-op, so the
# builder's `assert not missing` still checks that every kept context has generations.
# PROMPT_BUDGET_TOKENS must track data.max_prompt_length in qwen35_judge_grpo.yaml -- the
# emitted .meta.json is what that value has to be chosen from.
$PY -u scripts/build_judge_train_pairs.py \
  --inference_pkl "$PKL" --source_parquet "$SLICED_PARQUET" \
  --out "$OUT_DIR/$SPLIT.parquet" \
  --prompt_budget_tokens "${PROMPT_BUDGET_TOKENS:-10240}" \
  --slice_lo "$SLICE_LO" --slice_hi "$SLICE_HI" --split "$SPLIT" \
  --prompt-style "$PROMPT_STYLE" "${LIMIT_ARG[@]}"
