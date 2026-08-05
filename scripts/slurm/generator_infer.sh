#!/bin/bash
#SBATCH --job-name=gen_infer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/gen_infer-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
# Per-generator heldout candidate generation for the generator sweep.
# Required env: GEN_KEY MODEL_ID   Optional: CKPT (empty => --base_model)
# Uses gpu:8 (one whole node) so the single-node chain never overlaps a scoring job;
# vLLM uses TP=1 (7 GPUs idle) — chain serialization matters more than packing here.
# Callers that only need the one GPU can override at submit time: `sbatch --gres=gpu:1`.
#
# Optional overrides (ALL unset by default => byte-identical legacy behaviour):
#   SWEEP_BASE     output root (default: the 2026-07-15 generator-sweep tree)
#   EVAL_PARQUET   prompts to generate on (default: the held-out test.parquet)
#   EVAL_EXPECT    heldout|train|val|any -- asserted by scripts/check_eval_split.py (default heldout)
#   GEN_TEMPERATURE / GEN_TOP_P / GEN_TOP_K        sampling; unset => domain defaults (prism 0.6)
#   GEN_MAX_TOKENS / GEN_TRUNCATE_PROMPT_TOKENS / GEN_MAX_MODEL_LEN
#     length caps; set these to mirror GRPO validation (1024 / 12500 / 13524).
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1  # all models cached/local; avoids the concurrent-rank hub-check race
# BACKEND selects the inference path + env. vLLM (default) can't LoRA-serve Qwen3.5-9B's
# Gated-DeltaNet adapter, so that generator uses BACKEND=hf (transformers+PEFT in the
# transformers-5.x SFT env). See eval/generate_trained.py::generate_for_user_results_hf.
BACKEND=${BACKEND:-vllm}
case "$BACKEND" in
  vllm) PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python ;;
  hf)   PY=/home/lancewicki/miniconda3/envs/turing-rl-sft-qwen35/bin/python ;;
  *) echo "bad BACKEND=$BACKEND (expected vllm|hf)"; exit 2 ;;
esac
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
MODEL_ID=${MODEL_ID:?set MODEL_ID}
CKPT=${CKPT:-}
TEST=${EVAL_PARQUET:-$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet}
EVAL_EXPECT=${EVAL_EXPECT:-heldout}
SWEEP_BASE=${SWEEP_BASE:-$REPO/results/2026-07-15-generator-sweep}
OUT_DIR=$SWEEP_BASE/raw/generator/$GEN_KEY
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"; cd "$REPO"
[ -f "$TEST" ] || { echo "ERROR: missing $TEST"; exit 2; }

# Split guard. EVAL_PARQUET turns "held-out eval" into an unverified claim, so assert the split
# BEFORE loading a model. launch_test_eval.sh checks pre-submit too; this repeats it because a
# direct `sbatch scripts/slurm/generator_infer.sh` bypasses the launcher. Costs a few seconds.
# Always the train env's python: the guard needs pandas, and BACKEND=hf swaps interpreters.
/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/check_eval_split.py \
    --eval_parquet "$TEST" --expect "$EVAL_EXPECT" --out_json "$OUT_DIR/split_guard.json" \
  || { echo "ERROR: split guard rejected $TEST (expect=$EVAL_EXPECT)"; exit 3; }

BASE=(); [ -z "$CKPT" ] && BASE=(--base_model)
CK=(); [ -n "$CKPT" ] && CK=(--checkpoint_dir "$CKPT")

# Only pass a flag when its env var is set, so the default path is unchanged.
SAMPLING=()
[ -n "${GEN_TEMPERATURE:-}" ] && SAMPLING+=(--temperature "$GEN_TEMPERATURE")
[ -n "${GEN_TOP_P:-}" ]       && SAMPLING+=(--top_p "$GEN_TOP_P")
[ -n "${GEN_TOP_K:-}" ]       && SAMPLING+=(--top_k "$GEN_TOP_K")
[ -n "${GEN_MAX_TOKENS:-}" ]  && SAMPLING+=(--max_tokens "$GEN_MAX_TOKENS")
[ -n "${GEN_TRUNCATE_PROMPT_TOKENS:-}" ] && SAMPLING+=(--vllm_truncate_prompt_tokens "$GEN_TRUNCATE_PROMPT_TOKENS")
[ -n "${GEN_MAX_MODEL_LEN:-}" ]          && SAMPLING+=(--vllm_max_model_len "$GEN_MAX_MODEL_LEN")

echo "=== generator_infer: GEN_KEY=$GEN_KEY MODEL_ID=$MODEL_ID CKPT=${CKPT:-<base>} BACKEND=$BACKEND ==="
echo "=== out_dir=$OUT_DIR sampling_overrides=[${SAMPLING[*]-}] ==="
$PY -u -m eval.generate_trained "${BASE[@]}" "${CK[@]}" --test_parquet "$TEST" \
    --model_id "$MODEL_ID" --gen_num 1 --output "$OUT" --conditioning_mode history \
    --backend "$BACKEND" "${SAMPLING[@]}" \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32
RC=$?
$PY -c "import json,os; json.dump({'gen_key':'$GEN_KEY','model_id':'$MODEL_ID',\
'checkpoint_dir':'${CKPT:-}','base_model':$([ -z "$CKPT" ] && echo True || echo False),\
'test_parquet':'$TEST','eval_expect':'$EVAL_EXPECT','gen_num':1,'output':'$OUT','backend':'$BACKEND',\
'sampling_overrides':'${SAMPLING[*]-}',\
'slurm_job_id':os.environ.get('SLURM_JOB_ID')}, open('$OUT_DIR/gen_metadata.json','w'), indent=2)"
echo "=== exit: $RC ==="; exit $RC
