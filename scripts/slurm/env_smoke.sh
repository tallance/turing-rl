#!/bin/bash
#SBATCH --job-name=env_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/env_smoke-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
cd "$TURING_RL_WORK_ROOT"

echo "============================================"
echo "turing-rl-train env smoke import"
echo "Date: $(date)"
echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1
echo "============================================"

echo ""
echo "=== version check ==="
$PY -c "
import torch, vllm, verl, transformers, trl, flashinfer, peft, bitsandbytes
print(f'torch        {torch.__version__}')
print(f'  cuda       {torch.version.cuda}  available={torch.cuda.is_available()}  devices={torch.cuda.device_count()}')
print(f'vllm         {vllm.__version__}')
print(f'verl         {verl.__version__}')
print(f'transformers {transformers.__version__}')
print(f'trl          {trl.__version__}')
print(f'flashinfer   {flashinfer.__version__}')
print(f'peft         {peft.__version__}')
print(f'bitsandbytes {bitsandbytes.__version__}')
"
RC1=$?
echo "version-check exit: $RC1"

echo ""
echo "=== training entry-point imports (these trigger verl_runtime_patch) ==="
$PY -c "
print('importing training.grpo.run_verl_main_ppo ...')
from training.grpo import run_verl_main_ppo
print('  OK')
print('importing training.sft.lora_sft ...')
from training.sft import lora_sft
print('  OK')
"
RC2=$?
echo "entrypoint-import exit: $RC2"

echo ""
echo "=== tiny CUDA op ==="
$PY -c "
import torch
x = torch.randn(128, 128, device='cuda')
y = x @ x.T
print('matmul result mean:', y.mean().item())
"
RC3=$?
echo "cuda-op exit: $RC3"

echo "============================================"
echo "Done at $(date)"
if [ "$RC1" -eq 0 ] && [ "$RC2" -eq 0 ] && [ "$RC3" -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  exit 0
else
  echo "FAILED (version=$RC1 entrypoint=$RC2 cuda=$RC3)"
  exit 1
fi
