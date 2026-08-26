# Note for the RL-generator agent — deleted smoke launcher

**Deleted:** `bash_scripts/grpo/train_grpo_smoke.sh` (commit `3c1a475`).
**Why:** it was a near-copy of the canonical `bash_scripts/grpo/train_grpo.sh` — misleading as a
second source of truth.

**Impact for you:** the old smoke launchers `scripts/slurm/grpo_smoke.sh` /
`grpo_smoke_8b.sh` still call it → they're now **broken**. Treat them as **legacy** — don't
port them; build **fresh** launch scripts for the RL-generator work.

**Reuse this:** the deleted script's 40GB / small-slice overrides (batch 32, `rollout.n=2`,
shrunk lengths, `use_remove_padding=false`, `gpu_memory_utilization=0.45`) are preserved in
`our_patches.md` (section "DELETED: train_grpo_smoke.sh") — worth carrying into the fresh
scripts. `train_grpo.sh` forwards `"$@"` as Hydra overrides.
