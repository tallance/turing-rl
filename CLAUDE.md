Building on top of "Learning User Simulators with Turing Rewards" (arXiv:2606.19336). Repo: `~/projects/turing-rl`.

Long-term goal: co-train a GRPO user-simulator generator with a trainable discriminator to produce turns indistinguishable from real humans (adversarial extension of Turing-RL's frozen judge).
Additinal context is in `turing-rl/Adversarial-User-Simulation.md`

## Where you are
- **Mac working copy.** Edit code, write specs/plans, implement changes. No GPUs, no Slurm locally.
- Cluster state lives at `/home/lancewicki/projects/turing-rl`; immutable source snapshots live at `/home/lancewicki/projects/turing-rl-sources/<sha>`.
- Publish and launch committed code only through `scripts/cluster_launch.sh`. The legacy overlay deploy and global source stamp are retired.
- **Run cluster commands directly** over the SSH tunnel (see below) — no relay agent needed.

## Workflow
- Develop in a dedicated local Git worktree and commit before cluster execution. Dirty/WIP cluster runs are prohibited.
- Retained runs may use any clean commit containing current `lancewicki/main`. Older or divergent commits require `--debug --label <label>` and must write below `results/debug/<label>/`.
- Launch with `scripts/cluster_launch.sh --run-root <absolute-cluster-run-root> <launcher>`. It publishes a verified read-only snapshot, fingerprints external runtimes, and submits through the snapshot gateway.
- `scripts/snapshot_sbatch.sh` is the only maintained `sbatch` gateway. Never invoke `sbatch` directly for repository jobs.
- Repository snapshots freeze this repository only. Per-run manifests separately fingerprint veRL, Conda environments, package versions, model revisions where available, CUDA, GPUs and Slurm context.
- Always use the best model (latest Opus), both for yourself and sub-agents.
- Pull run results (plots, reports, metrics) locally into `results/<plan-name>/` (plan filename without `.md`); include a `README.txt` with provenance only: exact configuration and versions, job IDs and dates, cluster source paths, artifact filenames and checksums, mechanical validation status, and reproduction commands. Do not include results interpretation, scientific conclusions, hypothesis verdicts, or claims about what the results mean; those are for the user to decide.
- **Multi-agent integration.** Agents work on separate branches/worktrees. Only the designated integrator updates `lancewicki/main`, while holding `scripts/integration_lock.py` exclusively. Publication takes the same lock shared. Handoffs include commit SHA, changed files, tests and known conflicts.
- Same-file conflicts require semantic reconciliation with the originating agent or user; never discard a side wholesale merely to complete a merge. Reset/rebase/force-push, reverting another agent's work, or clearing another integrator's lock requires explicit user permission.
- Full workflow and command examples: `docs/cluster-workflow.md`.

## Cluster access (direct via SSH tunnel — primary)
The Mac agent reaches the cluster directly; the old Mac↔cluster relay-agent roundtrip is no longer required.
- The user keeps a tunnel open in a separate terminal: `ssh -L 2223:localhost:22 rfai-research-aws-use2-1 -N` (prompts 2FA; login pod has a 7-day TTL — refresh with `cloud_corp hpc login rfai-research-aws-use2-1` if it refuses).
- Run read-only/operational commands through it: `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<command>"`. Read remote files via SSH `cat`, not local file tools.
- Cluster state root: `/home/lancewicki/projects/turing-rl`; HF cache: `/home/lancewicki/data/hf_cache`; Slurm partition: `a100`.
- **Always run the `preflight-job-check` skill before any `sbatch`.** Don't exceed ~10 concurrent jobs; don't spam/loop Slurm commands. See the `rfai-cluster` skill for full details.
- **Never `scancel` a job you did not submit yourself in this session without explicit user approval** - I work with multiple agents so it is normal for a stray/unexpected job to appear - if it is a real blocker, ask me if I want to priortize your job and cancel the other one.
- Never edit, pull, or execute source from the cluster state root or a cluster Git worktree. Fixes require a new commit/snapshot and resubmission; running snapshots are intentionally not hot-patched.

## Cluster gotchas (V3)
- **Unset stale V2 proxy env vars in every job**: `unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY` (V3 uses transparent TLS egress; HF/PyPI allowlisted). Our sbatch scripts already do this.
- **A100s are 40GB, not 80GB.** bf16 8B + long-seq/LoRA can OOM — shard (FSDP) or quantize (QLoRA), and launch multi-GPU via `torch.distributed.run --nproc_per_node=8`, not plain `python` (plain python = single-GPU → wastes 7 GPUs and OOMs).
- **`/tmp` is a 1GB tmpfs** — for pip/heavy builds set `TMPDIR=~/tmp/build` and `PIP_CACHE_DIR=~/tmp/pip-cache`.
- **Slurm buffers stdout** — logs may lag; don't infer failure from an empty/short log.
- Prefer direct binary paths over `conda activate` in one-off commands (e.g. `/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python`).

## Fallback: agent comms (only if the tunnel is down)
If direct access isn't available, use the git relay: write instructions to `docs/agent-comms/<plan-name>/mac-to-cluster.md` (one subfolder per plan; `<plan-name>` = plan filename without `.md`), commit + push; the cluster agent runs it and replies in `cluster-to-mac.md` (don't edit that file). Pure local work (editing, planning, non-GPU code, local tests) skips any roundtrip.
