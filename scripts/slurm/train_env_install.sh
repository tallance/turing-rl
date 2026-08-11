#!/bin/bash
#SBATCH --job-name=train_env_install
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/home/lancewicki/projects/turing-rl/logs/train_env_install-%j.out
#SBATCH --partition=a100
#SBATCH --account=rfai

set -uo pipefail
source "${TURING_RL_CODE_ROOT:?}/scripts/cluster_job_bootstrap.sh"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY

export TMPDIR=/home/lancewicki/tmp/build
export PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

# Parallel build for flash_attn / flashinfer source builds
export MAX_JOBS=8
export PIP_NO_BUILD_ISOLATION=0

PIP=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/pip
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

echo "============================================"
echo "Install turing-rl-train requirements"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "============================================"

$PY -c "import torch; print('torch already installed:', torch.__version__, 'cuda:', torch.version.cuda)"
echo ""

echo "=== installing full requirements.txt + bitsandbytes ==="
# The repo's requirements.txt has irreconcilable internal version conflicts:
#   verl 0.7.1   -> requires numpy<2
#   vllm 0.18    -> requires opencv-python-headless>=4.13 -> requires numpy>=2
# The author must have installed with --no-deps (the pinned set is essentially a pip freeze
# with all transitive deps already listed at exact versions). We do the same.
#
# Other quirks we still handle:
#   - flashinfer-jit-cache lives on flashinfer's own index, not PyPI
#   - bitsandbytes is missing from requirements.txt but needed by training/sft/lora_sft.py for QLoRA
#   - flash_attn 2.8.3 has no cu130 wheel and source build fails on this cluster (system
#     CUDA toolkit is 12.8 vs torch's cu13.0). vLLM will fall back to FlashInfer attention.
#     We filter flash_attn out; it's a perf knob, not load-bearing.
REQS_FILTERED=$TMPDIR/requirements_filtered.txt
grep -v '^flash_attn==' "$TURING_RL_CODE_ROOT/requirements.txt" > "$REQS_FILTERED"
echo "filtered requirements at $REQS_FILTERED"

# Pass 1: install all pinned versions verbatim, bypassing pip's resolver.
$PIP install --no-deps \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --extra-index-url https://flashinfer.ai/whl/cu130/ \
  -r "$REQS_FILTERED"
RC1=$?
echo "pass 1 (no-deps full requirements minus flash_attn) exit code: $RC1"

# Pass 2: install bitsandbytes (not in requirements.txt) with normal deps.
$PIP install 'bitsandbytes>=0.43.0'
RC2=$?
echo "pass 2 (bitsandbytes) exit code: $RC2"

# Final: sanity-check known deps haven't been clobbered.
$PIP check
echo "pip check exit code: $?"

if [ "$RC1" -ne 0 ] || [ "$RC2" -ne 0 ]; then
  RC=1
else
  RC=0
fi
echo "============================================"
echo "pip install exit code: $RC"
echo "Done at $(date)"
echo "============================================"
exit $RC
