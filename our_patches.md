# Our patches to the upstream repo

This file tracks every modification we make to files in
`/home/lancewicki/projects/turing-rl/` that originated upstream
(https://github.com/SusanWYS/turing-rl). Anything under `scripts/`, `CLAUDE.md`,
`summary/`, `.env`, or `our_patches.md` itself is **ours** and is not tracked here.

The intent is to keep this list small. Before patching an upstream file, prefer:
1. Wrapping in our own sbatch / Python script under `scripts/`
2. Overriding via env var or CLI flag if one exists
3. Only as a last resort: modify the upstream file (and document it here)

If a patch is temporary (e.g., flipped during a run, restored after via `trap`),
mark it `TEMP` and explain how/when it gets reverted. If it's permanent for the
duration of the repro, mark it `PERSISTENT`.

---

## TEMP: `training/sft/configs/qwen3_8b_lora.yaml` — `report_to`

- **Original**: `report_to: none` (line 17)
- **Patched to**: `report_to: wandb`
- **Where**: applied/reverted by `scripts/slurm/sft_smoke.sh` via inline `sed`
  + `trap EXIT` (with `.smoke-bak` snapshot).
- **Why**: `lora_sft.py` reads `report_to` from this YAML and exposes no CLI
  override. We want SFT loss/lr curves in wandb at
  `https://meta.wandb.io/lancewicki/turing-rl-smoke`. Their default of `none`
  is fine for their workflow but blocks our visibility goal.
- **Reverted**: yes, automatically at job end (success or failure).

---

## PERSISTENT (new file): `bash_scripts/grpo/train_grpo_smoke.sh`

- **Origin**: a copy of `bash_scripts/grpo/train_grpo.sh` with minimal deltas.
- **Deltas** (see the SMOKE_OVERRIDES block in the file):
  - `actor_rollout_ref.model.use_remove_padding=false` — our env has no
    flash_attn (no cu130 wheel; see `scripts/slurm/train_env_install.sh`);
    verl's `unpad_input` requires it, so the sequence-packed path can't run.
  - `actor_rollout_ref.actor.use_remove_padding=false` — same reason.
  - `data.train_batch_size=32`, `ppo_mini_batch_size=32`,
    `ppo_micro_batch_size_per_gpu=1`, `rollout.n=2` — our 138-row smoke
    slice can't sustain their default 128/128/4/4.
  - `data.max_prompt_length=6144`, `rollout.max_model_len=7168`,
    `max_num_batched_tokens=8192`, `max_num_seqs=16`,
    `gpu_memory_utilization=0.45` — 40GB A100 safety (paper defaults assume
    80GB headroom).
  - `trainer.total_epochs=1`, `trainer.save_freq=2` — smoke scale.
  - `trainer.project_name=${WANDB_PROJECT:-turing-rl-smoke}` — route smoke
    runs to our wandb project.
  - Passes `"$@"` through as extra Hydra overrides so callers can tune ad hoc
    without re-editing the script.
- **Why**: we tried to reimplement the invocation from scratch in
  `scripts/slurm/grpo_smoke.sh` and hit six preventable bugs (`--config-path`
  vs `--config-dir`, forgetting `PYTHONPATH`, batch-size math, etc.). Reusing
  their launcher shape is simpler and keeps parity with how they run.
- **NOT a modification of an upstream file** — it's an additional sibling
  script under `bash_scripts/grpo/`. Listed here anyway because it lives
  inside their tree and could confuse a `git status` reader.

