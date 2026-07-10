#!/bin/bash
#SBATCH --job-name=heldout_inf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/heldout_inf-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai
set -uo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export HF_HOME=/home/lancewicki/data/hf_cache HF_HUB_CACHE=/home/lancewicki/data/hf_cache PYTHONUNBUFFERED=1
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
REPO=/home/lancewicki/projects/turing-rl
# Chosen generator: bf16_fsdp + --no_packing (clean per-conversation attention; full Table-5
# fidelity). --checkpoint_dir resolves to its final/ adapter via resolve_adapter_path.
CKPT=$REPO/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack
TEST=$REPO/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
OUT_DIR=$REPO/results/2026-07-08-judge-sweep/raw/generator
OUT=$OUT_DIR/heldout_inference.pkl
mkdir -p "$OUT_DIR"; cd "$REPO"
$PY -u -m eval.generate_trained --checkpoint_dir "$CKPT" --test_parquet "$TEST" \
    --model_id Qwen/Qwen3-8B --gen_num 1 --output "$OUT" --conditioning_mode history \
    --vllm_tensor_parallel_size 1 --vllm_gpu_memory_utilization 0.6 --vllm_max_num_seqs 32
RC=$?
$PY -c "import json,os; json.dump({'checkpoint_dir':'$CKPT','test_parquet':'$TEST',\
'base_model':'Qwen/Qwen3-8B','gen_num':1,'sampling':'paper Table 4 PRISM defaults (domain-inferred)',\
'output':'$OUT','slurm_job_id':os.environ.get('SLURM_JOB_ID')}, open('$OUT_DIR/heldout_inference_metadata.json','w'), indent=2)"
echo "=== exit: $RC ==="; exit $RC
