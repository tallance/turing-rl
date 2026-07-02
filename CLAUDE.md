# Turing-RL — repro of MIT's user-simulator paper

Reproducing "Learning User Simulators with Turing Rewards" (arXiv:2606.19336). Repo: `~/projects/turing-rl`.

## Environment
- **Conda env for vLLM judge**: `judge-vllm` (vllm 0.23.0, transformers 5.12.1, torch 2.11.0+cu130).
- **For Bash tool calls, prefer direct binary paths over `conda activate`** (e.g., `/home/lancewicki/miniconda3/envs/judge-vllm/bin/python`).
- Slurm: partition `a100` (8× A100-SXM4-**40GB** per node, driver 580.126.09 / CUDA 13.0). Login pod has no GPU — run `sbatch`/`srun`.
- Storage: FSx-NFS at `~`, **43TB free**, no enforced quota. `/tmp` is a 1GB tmpfs — for pip/heavy builds set `TMPDIR=~/tmp/build` and `PIP_CACHE_DIR=~/tmp/pip-cache`.
- HF cache: `~/data/hf_cache` (228GB used; judge model `Qwen3.5-397B-A17B-GPTQ-Int4` already cached, 220GB).

## Critical
- **V3 cluster requires unsetting stale V2 proxy env vars in every job**: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY`. V3 uses transparent TTLS for egress (HuggingFace/PyPI/etc are allowlisted).
- A100s are **40GB**, not 80GB. Turing-RL repo's default `max_model_len=13524` + LoRA r=64 + vLLM rollout will likely OOM. Plan to reduce `max_model_len` or raise `gpu_memory_utilization`.
- Slurm buffers stdout — logs may lag; don't assume failure from an empty log.

## Workflow
- Submit: `sbatch scripts/slurm/<job>.sh`; monitor: `squeue --me` and `tail -f` the log.
- Always use the best model (latest Opus), both for yourself as well as for sub-agents.
