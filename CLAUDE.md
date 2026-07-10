Building on top of "Learning User Simulators with Turing Rewards" (arXiv:2606.19336). Repo: `~/projects/turing-rl`.

Long-term goal: co-train a GRPO user-simulator generator with a trainable discriminator to produce turns indistinguishable from real humans (adversarial extension of Turing-RL's frozen judge).
Additinal context is in `turing-rl/Adversarial-User-Simulation.md`

## Where you are
- **Mac working copy.** Edit code, write specs/plans, implement changes. No GPUs, no Slurm locally.
- Cluster checkout at `/storage/home/lancewicki/projects/turing-rl` on V3 AWS (8× A100-40GB per node) runs everything.
- Sync CODE via git: commit + push to `mine/lancewicki/main` (fork: `tallance/turing-rl`); the cluster checkout pulls.
- **Run cluster commands directly** over the SSH tunnel (see below) — no relay agent needed.

## Workflow
- Edit → commit (→ `git push` for backup/history) → `scripts/sync_to_cluster.sh` deploys the committed tree to the cluster and stamps `DEPLOYED_SHA` → run on the cluster via the tunnel.
- Mac is the SOLE author; the cluster is a compute mirror (never edited/committed there). The sync ships only the committed HEAD, so every run maps to a SHA (`cat DEPLOYED_SHA` on the cluster).
- Always use the best model (latest Opus), both for yourself and sub-agents.

## Cluster access (direct via SSH tunnel — primary)
The Mac agent reaches the cluster directly; the old Mac↔cluster relay-agent roundtrip is no longer required.
- The user keeps a tunnel open in a separate terminal: `ssh -L 2223:localhost:22 rfai-research-aws-use2-1 -N` (prompts 2FA; login pod has a 7-day TTL — refresh with `cloud_corp hpc login rfai-research-aws-use2-1` if it refuses).
- Run any command through it: `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<command>"` — squeue/sinfo/sbatch/scancel, `cat` remote files, tail logs, inspect checkpoints. Read remote files via SSH `cat`, NOT the Read tool.
- Cluster paths/env: repo `/home/lancewicki/projects/turing-rl`; HF cache `/home/lancewicki/data/hf_cache`; conda envs `turing-rl-train` (vLLM/torch/trl) and `judge-vllm` (397B anchor). Slurm is on PATH; partition `a100`.
- **Always run the `preflight-job-check` skill before any `sbatch`.** Don't exceed ~10 concurrent jobs; don't spam/loop Slurm commands. See the `rfai-cluster` skill for full details.
- Deploy code with `scripts/sync_to_cluster.sh` (ships committed HEAD via `git archive|tar`, stamps `DEPLOYED_SHA`, verifies `.py`/`.sh` syntax on the cluster; never touches `checkpoints/ results/ logs/ wandb/`). Pass file paths for a quick dirty debug push. Cluster `git pull` is no longer needed.

## Cluster gotchas (V3)
- **Unset stale V2 proxy env vars in every job**: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY` (V3 uses transparent TLS egress; HF/PyPI allowlisted). Our sbatch scripts already do this.
- **A100s are 40GB, not 80GB.** bf16 8B + long-seq/LoRA can OOM — shard (FSDP) or quantize (QLoRA), and launch multi-GPU via `torch.distributed.run --nproc_per_node=8`, not plain `python` (plain python = single-GPU → wastes 7 GPUs and OOMs).
- **`/tmp` is a 1GB tmpfs** — for pip/heavy builds set `TMPDIR=~/tmp/build` and `PIP_CACHE_DIR=~/tmp/pip-cache`.
- **Slurm buffers stdout** — logs may lag; don't infer failure from an empty/short log.
- Prefer direct binary paths over `conda activate` in one-off commands (e.g. `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python`).

## Fallback: agent comms (only if the tunnel is down)
If direct access isn't available, use the git relay: write instructions to `docs/agent-comms/<plan-name>/mac-to-cluster.md` (one subfolder per plan; `<plan-name>` = plan filename without `.md`), commit + push; the cluster agent runs it and replies in `cluster-to-mac.md` (don't edit that file). Pure local work (editing, planning, non-GPU code, local tests) skips any roundtrip.
