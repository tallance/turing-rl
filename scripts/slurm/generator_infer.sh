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
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache PYTHONUNBUFFERED=1
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
GEN_KEY=${GEN_KEY:?set GEN_KEY}
MODEL_ID=${MODEL_ID:?set MODEL_ID}
CKPT=${CKPT:-}
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT_DIR=$REPO/results/2026-07-15-generator-sweep/raw/generator/$GEN_KEY
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"; cd "$REPO"
[ -f "$TEST" ] || { echo "ERROR: missing $TEST"; exit 2; }

BASE=(); [ -z "$CKPT" ] && BASE=(--base_model)
CK=(); [ -n "$CKPT" ] && CK=(--checkpoint_dir "$CKPT")

echo "=== generator_infer: GEN_KEY=$GEN_KEY MODEL_ID=$MODEL_ID CKPT=${CKPT:-<base>} ==="
$PY -u -m eval.generate_trained "${BASE[@]}" "${CK[@]}" --test_parquet "$TEST" \
    --model_id "$MODEL_ID" --gen_num 1 --output "$OUT" --conditioning_mode history \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32
RC=$?
$PY -c "import json,os; json.dump({'gen_key':'$GEN_KEY','model_id':'$MODEL_ID',\
'checkpoint_dir':'${CKPT:-}','base_model':$([ -z "$CKPT" ] && echo true || echo false),\
'test_parquet':'$TEST','gen_num':1,'output':'$OUT',\
'slurm_job_id':os.environ.get('SLURM_JOB_ID')}, open('$OUT_DIR/gen_metadata.json','w'), indent=2)"
echo "=== exit: $RC ==="; exit $RC
