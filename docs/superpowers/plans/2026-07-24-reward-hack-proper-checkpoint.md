# Reward-Hacking Probe on a Proper SFT Checkpoint (+ 9B generator) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-run the RL-generator-vs-fixed-judge reward-hacking probe seeding GRPO from the
stop-token-supervised SFT checkpoint (removing the non-terminating-generator confound), as a
KL×LR overfit grid, for a Qwen3-8B generator (Arm A, now) and a Qwen3.5-9B generator (Arm B,
gated behind a feasibility spike).

**Architecture:** Arm A reuses the existing 2026-07-15 RL machinery unchanged
(`scripts/slurm/rl_generator_run.sh` → `rl_generator_train.sh` → `training.grpo.run_verl_main_ppo`,
atomic 2-node judge+trainer, DP-8 9B judge) — the only changes are the merged-SFT-reference path
(now checkpoint-78) and a 6-cell (KL×LR) submission grid. Arm B builds a new pinned veRL env and a
9B launcher variant (LoRA merge=True, attn+MLP target, TP4/FSDP8/offload), gated by a B0 spike that
asserts rollout weights actually track the actor.

**Tech Stack:** veRL/vLLM GRPO, PEFT LoRA, PRISM parquet data, Slurm (RFAI a100), pytest,
conda envs `turing-rl-train` (Arm A) and a new `turing-rl-rl-qwen35` (Arm B).

**Spec:** `docs/superpowers/specs/2026-07-24-reward-hack-proper-checkpoint-design.md`
**Precedent:** `docs/superpowers/specs/2026-07-15-rl-generator-vs-fixed-judge-design.md` +
`docs/superpowers/post-plans/2026-07-15-rl-generator-vs-fixed-judge/decisions.md`

## Global Constraints

- **Cap:** no cap — `TURING_JUDGE_SCORE_CLIP_MAX=7` (already set in the launcher). Never change.
- **Judge (both arms):** frozen `Qwen/Qwen3.5-9B`, served DP-8 (TP1×DP8) on one node; sampling
  `{"repetition_penalty":1.1,"temperature":0.6}`, thinking-on, `max_completion_tokens=8192`.
- **Grid:** KL ∈ {1e-3, 1e-4, 0} × LR ∈ {1e-5, 1e-4}, all no-cap, 50 overfit epochs, no early-stop.
- **SFT init + KL ref:** the **stop-token-supervised checkpoint-78** (ep3) — NOT the buggy
  `.../qwen3_8b_prism_full_s42_bf16_fsdp_nopack/final`. Merged into a standalone backbone;
  `lora_adapter_path=null`; fresh RL LoRA **r64 / α32**.
- **LoRA target (both arms, set EXPLICITLY for H2 parity):**
  `target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]` (attention + MLP).
  Set it on **Arm A too** — do NOT rely on the inherited `all-linear` from `qwen3_8b_grpo.yaml`; the
  two arms' configs must match literally. (On full-attention Qwen3-8B, `all-linear` expands to the
  same 7 modules, so this is a clarifying override, not a behavior change for 8B.) For Arm B, the
  explicit list matters: it **excludes the Gated-DeltaNet backbone** (`in_proj_*`, `out_proj`),
  which our SFT recipe found destructive (arXiv:2604.22127). Arm B additionally
  `exclude_modules='.*visual.*'`. (Note: veRL #6782's Qwen3.5-27B GRPO ran `all-linear` without
  crashing — so this is a parity + our-own-SFT-quality choice, not a hard upstream blocker.)
- **Arm B env:** pinned veRL SHA **≥ `c791da0b`** (must contain #7014 merged-weight sync + #5599
  Qwen3.5 GDN mappings) + vLLM 0.20.2 + transformers 5.4.0 + FLA 0.5.1, built in veRL Docker order.
  Candidate stack — validated only by B0. **Do not touch `turing-rl-train`.**
- **Gate metric:** `scripts/overfit_gate_check.py`, strict per-prompt majority (`frac > 0.5`, ties
  excluded), pass = ≥8/10 on final-epoch rollouts; also report a last-K-epoch average.
- **Cluster hygiene:** run `preflight-job-check` before every `sbatch`; ≤10 concurrent jobs; deploy
  via `scripts/sync_to_cluster.sh` (committed HEAD); additive commits only; never `scancel` others'
  jobs. Cluster access via SSH tunnel `ssh -p 2223 … lancewicki@localhost`.
- **Scope:** overfit-on-a-handful only. Full-split runs + 880-heldout eval are a post-plan follow-on.

---

# Phase 1 — Arm A (Qwen3-8B), existing stack, submit first

### Task 1: Arm-A grid definition + config-integrity test

Encode the 6 KL×LR cells as data so submission is a loop, not copy-paste, and lock the values with
a test (mirrors `tests/test_sweep_cell_config.py` / `tests/test_grpo_config.py`).

**Files:**
- Create: `scripts/rl_grid.py`
- Test: `tests/test_rl_grid.py`

**Interfaces:**
- Produces: `ARM_A_CELLS: list[dict]` and `cell_overrides(cell) -> str`. Each cell dict:
  `{"tag": str, "kl": float, "lr": float}`. `cell_overrides` returns the veRL Hydra override
  string `actor_rollout_ref.actor.kl_loss_coef=<kl> actor_rollout_ref.actor.optim.lr=<lr>`
  (the exact string passed as `EXTRA_OVERRIDES` to `rl_generator_run.sh`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_grid.py
from scripts.rl_grid import ARM_A_CELLS, cell_overrides

def test_grid_is_full_kl_by_lr_no_duplicates():
    kls = {1e-3, 1e-4, 0.0}
    lrs = {1e-5, 1e-4}
    got = {(c["kl"], c["lr"]) for c in ARM_A_CELLS}
    assert got == {(k, l) for k in kls for l in lrs}      # 6 cells, full cross
    tags = [c["tag"] for c in ARM_A_CELLS]
    assert len(tags) == len(set(tags)) == 6               # unique tags

def test_tags_encode_model_kl_lr_and_proper():
    for c in ARM_A_CELLS:
        assert c["tag"].startswith("8b_proper_")          # proper-checkpoint runs
    hack = next(c for c in ARM_A_CELLS if c["kl"] == 1e-3 and c["lr"] == 1e-4)
    assert hack["tag"] == "8b_proper_kl1e3_lr1e4"

def test_cell_overrides_string():
    hack = {"tag": "8b_proper_kl1e3_lr1e4", "kl": 1e-3, "lr": 1e-4}
    ovr = cell_overrides(hack)
    assert "actor_rollout_ref.actor.kl_loss_coef=0.001" in ovr
    assert "actor_rollout_ref.actor.optim.lr=0.0001" in ovr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_grid.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.rl_grid'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rl_grid.py
"""Arm-A / Arm-B GRPO overfit grid: KL x LR cells for the proper-checkpoint reward-hack repeat."""
from __future__ import annotations


def _fmt(x: float) -> str:
    # compact tag token: 1e-3 -> "1e3", 1e-4 -> "1e4", 0 -> "0"
    if x == 0:
        return "0"
    exp = round(-__import__("math").log10(x))
    return f"1e{exp}"


def _cells(model: str) -> list[dict]:
    kls = [1e-3, 1e-4, 0.0]
    lrs = [1e-5, 1e-4]
    out = []
    for kl in kls:
        for lr in lrs:
            out.append({"tag": f"{model}_proper_kl{_fmt(kl)}_lr{_fmt(lr)}", "kl": kl, "lr": lr})
    return out


ARM_A_CELLS = _cells("8b")
ARM_B_CELLS = _cells("9b")


def cell_overrides(cell: dict) -> str:
    """Hydra override string for EXTRA_OVERRIDES (KL + LR)."""
    return (
        f"actor_rollout_ref.actor.kl_loss_coef={cell['kl']:g} "
        f"actor_rollout_ref.actor.optim.lr={cell['lr']:g}"
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rl_grid.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/rl_grid.py tests/test_rl_grid.py
git commit -m "feat: Arm-A KL x LR overfit grid definition + config-integrity test"
```

---

### Task 2: Proper-checkpoint path resolver + distinctness test

Add a single source of truth for the proper (stop-token-supervised) checkpoint-78 adapter and its
merged output, and a test that it is **not** the buggy `/final` path. Keeps the "which checkpoint"
decision in code, not in a shell one-liner.

**Files:**
- Modify: `scripts/merge_sft_adapter.py` (add named constants; keep CLI defaults backward-compatible)
- Test: `tests/test_proper_checkpoint.py`

**Interfaces:**
- Produces (in `scripts/merge_sft_adapter.py`):
  `PROPER_ADAPTER_DIR_8B = REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78"`
  and `PROPER_MERGED_DIR_8B = REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3"`.
  (Existing `DEFAULT_ADAPTER_DIR` / `DEFAULT_OUTPUT_DIR` = the buggy paths — leave unchanged so old
  runs stay reproducible.)

> NOTE: the exact epochsave dir name is confirmed on the cluster in Task 4 (tunnel was down at
> plan-writing time). If it differs, update these two constants — the test only asserts *distinctness
> and shape*, not filesystem existence, so it stays green offline.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proper_checkpoint.py
from scripts.merge_sft_adapter import (
    DEFAULT_ADAPTER_DIR, DEFAULT_OUTPUT_DIR,
    PROPER_ADAPTER_DIR_8B, PROPER_MERGED_DIR_8B,
)

def test_proper_is_distinct_from_buggy():
    assert PROPER_ADAPTER_DIR_8B != DEFAULT_ADAPTER_DIR
    assert PROPER_MERGED_DIR_8B != DEFAULT_OUTPUT_DIR

def test_proper_points_at_epochsave_checkpoint78():
    assert PROPER_ADAPTER_DIR_8B.name == "checkpoint-78"
    assert "epochsave" in str(PROPER_ADAPTER_DIR_8B)
    assert str(DEFAULT_ADAPTER_DIR).endswith("/final")   # buggy stays the old one
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_proper_checkpoint.py -q`
Expected: FAIL — `ImportError: cannot import name 'PROPER_ADAPTER_DIR_8B'`.

- [ ] **Step 3: Write minimal implementation**

Add below the existing `DEFAULT_OUTPUT_DIR` line in `scripts/merge_sft_adapter.py`:

```python
# Stop-token-supervised (proper) SFT checkpoint from the 2026-07-21 trajectory run (job 10715),
# ep3. Distinct from the buggy DEFAULT_* above (whose completion mask excluded <|im_end|>).
PROPER_ADAPTER_DIR_8B = (
    REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78"
)
PROPER_MERGED_DIR_8B = (
    REPO_ROOT / "checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3"
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_proper_checkpoint.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/merge_sft_adapter.py tests/test_proper_checkpoint.py
git commit -m "feat: proper (stop-token) checkpoint-78 path constants, distinct from buggy final"
```

---

### Task 3: Arm-A grid submit script

A thin submitter that loops `ARM_A_CELLS`, pointing every run at the proper merged checkpoint. No new
training logic — just the correct env for `rl_generator_run.sh`. (Not idempotent: re-running
re-submits; check `squeue`/existing run dirs before re-invoking.)

**Files:**
- Create: `scripts/slurm/submit_arm_a_grid.sh`
- Test: `tests/test_submit_arm_a_grid.py` (static assertions on the script text — no cluster)

**Interfaces:**
- Consumes: `scripts/rl_grid.py::ARM_A_CELLS` (via `python -c`), `scripts/slurm/rl_generator_run.sh`.
- Produces: one `sbatch` per cell with `JUDGE=9b MODE=overfit OVERFIT_EPOCHS=50`,
  `RUN_TAG=<cell.tag>`, `MERGED_SFT_MODEL_PATH=<proper merged>`, `EXTRA_OVERRIDES=<cell_overrides>`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_arm_a_grid.py
import pathlib
S = pathlib.Path("scripts/slurm/submit_arm_a_grid.sh").read_text()

def test_uses_proper_merged_checkpoint_and_no_early_stop():
    assert "epochsave/merged_ep3" in S            # proper checkpoint, not /merged (buggy)
    assert "OVERFIT_EPOCHS=50" in S
    assert "MODE=overfit" in S and "JUDGE=9b" in S

def test_drives_grid_from_rl_grid_module():
    assert "rl_grid" in S                          # loops the SSOT cells, no hardcoded 6x copy-paste
    assert "rl_generator_run.sh" in S

def test_explicit_lora_target_for_h2_parity():
    # Arm A must set the target explicitly, not inherit all-linear (matches Arm B literally).
    assert "target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]" in S
    assert "all-linear" not in S
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_submit_arm_a_grid.py -q`
Expected: FAIL — `FileNotFoundError: scripts/slurm/submit_arm_a_grid.sh`.

- [ ] **Step 3: Write minimal implementation**

```bash
#!/bin/bash
# Submit the 6-cell (KL x LR) Arm-A overfit grid on the PROPER (stop-token) checkpoint-78.
# Run from the CLUSTER repo root after sync_to_cluster.sh + Task 4 (merge). preflight-job-check first.
set -euo pipefail
REPO=/home/lancewicki/projects/turing-rl
cd "$REPO"
PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
MERGED=checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
# Explicit LoRA target (H2 parity) — override the config's inherited all-linear so Arm A/B match
# literally. On full-attn Qwen3-8B this is the same 7 modules all-linear expands to.
TARGET='actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'

# Emit "tag<TAB>overrides" per cell from the SSOT grid module.
$PY - <<'PYEOF' | while IFS=$'\t' read -r TAG OVR; do
from scripts.rl_grid import ARM_A_CELLS, cell_overrides
for c in ARM_A_CELLS:
    print(f"{c['tag']}\t{cell_overrides(c)}")
PYEOF
  echo ">> submitting $TAG :: $OVR $TARGET"
  JUDGE=9b MODE=overfit OVERFIT_EPOCHS=50 \
    RUN_TAG="$TAG" \
    MERGED_SFT_MODEL_PATH="$MERGED" \
    EXTRA_OVERRIDES="$OVR $TARGET" \
    sbatch --export=ALL scripts/slurm/rl_generator_run.sh
done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_submit_arm_a_grid.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/slurm/submit_arm_a_grid.sh tests/test_submit_arm_a_grid.py
git commit -m "feat: Arm-A grid submitter (proper ckpt-78, 6 cells, 50-epoch overfit)"
```

---

### Task 4: Deploy + confirm cluster checkpoint path + merge checkpoint-78 (8B)

Cluster op. Produces the merged proper backbone the grid needs, and validates the merge parity that
the spec's "merged-SFT ref" test #3 requires.

**Files:** none (cluster commands). Uses `scripts/merge_sft_adapter.py`.

- [ ] **Step 1: Deploy committed HEAD to the cluster**

Run (Mac): `scripts/sync_to_cluster.sh`
Expected: ends with a `DEPLOYED_SHA` stamp equal to your worktree HEAD; `.py`/`.sh` syntax check passes.

- [ ] **Step 2: Confirm the proper checkpoint-78 path exists on the cluster**

Run: `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost 'ls -d /home/lancewicki/projects/turing-rl/checkpoints/sft/*epochsave*/checkpoint-78 && ls /home/lancewicki/projects/turing-rl/checkpoints/sft/*epochsave*/checkpoint-78/adapter_model.safetensors'`
Expected: prints the checkpoint-78 dir + its `adapter_model.safetensors`. **If the dir name differs
from `qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave`, update the constants in Task 2 and
`MERGED`/`MERGED_SFT_MODEL_PATH` in Task 3, re-commit, re-sync.**

- [ ] **Step 3: Merge the proper adapter into a standalone backbone (on the cluster)**

Run: `ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && HF_HUB_OFFLINE=1 /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/merge_sft_adapter.py --base-model Qwen/Qwen3-8B --adapter-dir checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78 --output-dir checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3'`
Expected: writes `merged_ep3/` with `config.json`, `tokenizer_config.json`,
`sft_merge_metadata.json`, and `model*.safetensors`. NOTE: the script's `validate_merged_artifact`
only checks **files + metadata** — it does NOT prove numerical parity. Numerical parity is Step 4.

- [ ] **Step 4: Verify logits parity (base+adapter ≈ merged) — the real merged-SFT-ref guard**

`validate_merged_artifact` is structural only; assert numerical equivalence on a fixed input so the
merged backbone truly equals `base+adapter` (the KL reference the GRPO run recovers by disabling the
fresh LoRA). **Compare the LIVE, UNMERGED PeftModel** (`base+adapter`, no `merge_and_unload`) against
the serialized merged model — merging in-memory first would compare merged-vs-merged and prove
nothing. **Run inside a Slurm GPU allocation** (`.cuda()` on the login node is forbidden):

```bash
ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && srun --partition=a100 --account=rfai --gres=gpu:1 --time=00:20:00 \
  bash -lc "HF_HUB_OFFLINE=1 /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python - <<PY
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
B=\"Qwen/Qwen3-8B\"
A=\"checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/checkpoint-78\"
M=\"checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3\"
tok=AutoTokenizer.from_pretrained(M)
ids=tok(\"The quick brown fox jumps over\", return_tensors=\"pt\").input_ids.cuda()
base=AutoModelForCausalLM.from_pretrained(B, torch_dtype=torch.bfloat16, device_map=\"cuda\")
ref=PeftModel.from_pretrained(base, A).eval()          # LIVE base+adapter, NOT merged
with torch.no_grad(): lo_ref=ref(ids).logits.float()
del base, ref; torch.cuda.empty_cache()
mer=AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16, device_map=\"cuda\").eval()
with torch.no_grad(): lo_mer=mer(ids).logits.float()
d=(lo_ref-lo_mer).abs().max().item()
print(\"max|Δlogits| =\", d); assert d < 1e-2, f\"merge parity FAILED: {d}\"
print(\"MERGE PARITY OK\")
PY"'
```
Expected: `max|Δlogits|` ≪ 1e-2 and `MERGE PARITY OK`. If it fails, the merge is wrong — stop.
(Run `preflight-job-check` before the srun; the merge itself in Step 3 is CPU-only and fine on login.)

- [ ] **Step 5: Sanity-check the merged dir shape**

Run: `ssh -p 2223 … lancewicki@localhost 'ls /home/lancewicki/projects/turing-rl/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3'`
Expected: contains `config.json tokenizer_config.json sft_merge_metadata.json` + `model-*.safetensors`.
(This is exactly what `rl_generator_train.sh` asserts before launching.)

- [ ] **Step 6: (No commit — cluster artifact.)** Record the confirmed path in the run ledger
`.sdd-progress.md` (untracked).

---

### Task 5: Ensure the overfit-10 dataset exists on the cluster

`rl_generator_train.sh` (MODE=overfit) reads
`data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet`. It should already
exist from the 2026-07-15 runs; build it if missing. The builder + its test already exist
(`tests/test_overfit10_builder.py`).

**Files:** none new (reuse the existing overfit-10 builder).

- [ ] **Step 1: Check presence on the cluster**

Run: `ssh -p 2223 … lancewicki@localhost 'ls -l /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/grpo/train_overfit10.parquet'`
Expected: the parquet exists (from prior runs). **If missing:** locate the builder
(`grep -rl overfit10 scripts/`) and run it on the cluster, then re-check. Expected result: a
10-row parquet, strict subset of `grpo/train.parquet`.

- [ ] **Step 2: Confirm the builder test passes locally** (guards the schema/subset invariant)

Run: `python -m pytest tests/test_overfit10_builder.py -q`
Expected: PASS.

---

### Task 6: Submit the Arm-A grid + verify liveness

Cluster op. The one deliberate deviation (no cap) and all fidelity knobs are already baked into
`rl_generator_run.sh`.

- [ ] **Step 1: Preflight**

Invoke the `preflight-job-check` skill. Resolve every flagged item before submitting. Confirm
current queue < ~4 idle-node budget (each cell = a 2-node job; stage submissions to stay ≤10
concurrent).

- [ ] **Step 2: Submit the grid**

Run: `ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && bash scripts/slurm/submit_arm_a_grid.sh'`
Expected: 6 `Submitted batch job <id>` lines, tags `8b_proper_kl{1e3,1e4,0}_lr{1e5,1e4}`.
(If node budget is tight, submit the two lr=1e-4 cells first — that is where the hack is expected.)

- [ ] **Step 3: Verify each run is training (not stalled on the judge handoff)**

Run: `ssh -p 2223 … lancewicki@localhost 'squeue -u lancewicki'` then tail one log:
`ssh … 'tail -40 /home/lancewicki/projects/turing-rl/logs/rl_gen-<jobid>.out'`
Expected: `judge endpoint:` line printed, then the trainer step banner
(`RL-gen trainer: MODE=overfit … cap=7`) and veRL step logs; reward-dump jsonl files begin
appearing under `results/grpo/rl-generator/<tag>/reward_dump/`.

- [ ] **Step 4: Verify the merged-SFT KL reference is correct (guards the old B4 bug)**

In the same log, check `actor/kl_loss` at the first step is **near zero** (the merged backbone with
its fresh RL LoRA disabled must recover the SFT policy = the KL reference). The buggy 2026-07-15
runs showed step-one `kl_loss` ~0.63–0.86 because the reference was bare Qwen3-8B.
Expected: step-1 `actor/kl_loss` ≈ 0 (≪ 0.1). If it is large, the merge/`lora_adapter_path=null`
wiring is wrong — stop and re-check Task 4.

- [ ] **Step 5: Record job ids** in `.sdd-progress.md` (tag → jobid → run_dir).

---

### Task 7: Arm-A analysis — gate, plots, comparison, results dir

Runs once the grid completes (~7h/cell). Produces the headline deliverable.

**Files:**
- Create: `results/2026-07-24-reward-hack-proper-checkpoint/README.txt`
- Reuse: `scripts/overfit_gate_check.py`, `scripts/plot_overfit_ratings.py`

- [ ] **Step 1: Per-cell overfit gate (strict >0.5)**

Run (per tag): `ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python scripts/overfit_gate_check.py --dump_dir results/grpo/rl-generator/<tag>/reward_dump'`
Expected: prints `wins/10`, `win_rate`, `passed`. Record all 6.

- [ ] **Step 2: Per-cell rating-trajectory plot**

Run (per tag): `ssh … 'cd … && python scripts/plot_overfit_ratings.py --dump_dir results/grpo/rl-generator/<tag>/reward_dump --out results/grpo/rl-generator/<tag>/rating_scatter.png'`
Expected: a 10-subplot scatter+mean PNG per cell.

- [ ] **Step 3: Pull results locally**

Run (Mac): `mkdir -p results/2026-07-24-reward-hack-proper-checkpoint && scp -P 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null 'lancewicki@localhost:/home/lancewicki/projects/turing-rl/results/grpo/rl-generator/8b_proper_*/rating_scatter.png' results/2026-07-24-reward-hack-proper-checkpoint/`
Expected: 6 PNGs land locally.

- [ ] **Step 4: Write the comparison + README**

Create `results/2026-07-24-reward-hack-proper-checkpoint/README.txt` with: repro commands (this
plan), cluster source paths (`results/grpo/rl-generator/8b_proper_*`), and the headline table —
each cell's clean-checkpoint win-rate/gate **next to the buggy-checkpoint number**
(kl1e3/lr1e5 → 5/10 ~0.60; kl1e4 → 4/10 0.590; kl0 → 4/10 0.575; **kl1e3/lr1e4 → 8/10 0.744**).
State the H1 verdict: does lr=1e-4 replicate the ≥8/10 hack on the clean checkpoint?

- [ ] **Step 5: Commit the README + plots.** `results/` is gitignored — use `git add -f` so the
plan-results dir + README are committed for the record.

```bash
git add -f results/2026-07-24-reward-hack-proper-checkpoint/README.txt results/2026-07-24-reward-hack-proper-checkpoint/*.png
git commit -m "results: Arm-A proper-checkpoint overfit grid (H1 verdict + buggy-vs-clean table)"
```

---

# Phase 2 — Arm B (Qwen3.5-9B generator), new stack, gated

> Do NOT start Phase 2 until Arm A jobs are submitted (Task 6). Arm A does not depend on any Phase-2
> work. Phase 2 gates on the B0 spike (Task 10).

### Task 8: B0 rollout-sync guard via LOGPROB parity + unit test

The critical B0 assertion: the rollout policy must track the actor (the pre-#7014 bug ran crash-free
while vLLM served the *base* policy). **Do NOT compare raw weights** — vLLM does not expose an actor-
comparable weight API, its params are TP-sharded and fused (HF actor weights are split), and hashing
all 9B params as float32 would move ~36GB through CPU. Instead compare **per-token logprobs on a
fixed prompt**: veRL already logs both the rollout logprobs (from vLLM at generation) and the actor's
recomputed logprobs (`old_log_prob`) for the same tokens. The guard uses those. Build a pure helper
now (unit-testable offline); wire it into the spike in Task 10.

**Files:**
- Create: `scripts/rollout_sync_guard.py`
- Test: `tests/test_rollout_sync_guard.py`

**Interfaces:**
- Produces:
  `logprob_parity(rollout_lp, actor_lp, atol=0.1) -> dict` — compares two per-token logprob arrays
  for the same generated tokens; returns `{"max_abs_diff": float, "close": bool}`
  (`close` = max abs diff ≤ atol). If rollout and actor disagree, vLLM is serving different weights
  than the actor holds (the stale-base symptom).
  `assert_rollout_synced(step0: dict, step1: dict, tf_lp0, tf_lp1) -> dict` — `step0`/`step1` are the
  within-step logprob-parity dicts (rollout vs actor on the SAME generated tokens, valid).
  `tf_lp0`/`tf_lp1` are the actor's **teacher-forced** logprobs on a **FIXED prompt+continuation**
  evaluated at step0 and step1 — identical tokens, so a cross-step diff is meaningful (unlike sampled
  rollouts, which differ every step). Returns `{"synced": bool, "policy_moved": bool, "ok": bool}`:
  `synced` = rollout≈actor at BOTH steps; `policy_moved` = the teacher-forced logprobs changed after
  the update; `ok = synced and policy_moved`. Raises if `tf_lp0`/`tf_lp1` shapes differ (that would
  mean the "fixed" sequence wasn't actually fixed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rollout_sync_guard.py
import numpy as np
from scripts.rollout_sync_guard import logprob_parity, assert_rollout_synced

def test_parity_close_and_far():
    a = np.array([-0.10, -1.20, -0.03])
    assert logprob_parity(a, a + 0.02, atol=0.1)["close"] is True      # within tol
    r = logprob_parity(a, a + 0.5, atol=0.1)
    assert r["close"] is False and r["max_abs_diff"] > 0.4             # rollout != actor

def test_ok_when_synced_both_steps_and_policy_moved():
    # within-step parity dicts (rollout vs actor, same sampled tokens — may differ per step)
    s0 = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))   # close @ step0
    s1 = logprob_parity(np.array([-0.4, -0.9, -0.3]), np.array([-0.41, -0.9, -0.29]))  # close @ step1
    # teacher-forced logprobs on a FIXED 4-token continuation, evaluated at both steps (same shape)
    tf0 = np.array([-0.10, -0.20, -0.30, -0.40])
    tf1 = np.array([-0.50, -0.60, -0.70, -0.80])                            # actor moved
    out = assert_rollout_synced(s0, s1, tf0, tf1)
    assert out == {"synced": True, "policy_moved": True, "ok": True}

def test_stale_base_flagged():           # rollout frozen at base while actor trains -> step1 desync
    s0 = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))   # ok @ step0
    s1 = logprob_parity(np.array([-0.4, -0.9]), np.array([-0.4, -0.2]))     # rollout != actor @ step1
    tf0 = np.array([-0.1, -0.2, -0.3, -0.4]); tf1 = np.array([-0.5, -0.6, -0.7, -0.8])
    out = assert_rollout_synced(s0, s1, tf0, tf1)
    assert out["synced"] is False and out["ok"] is False

def test_frozen_policy_flagged():        # rollout tracks actor but teacher-forced logprobs unchanged
    s = logprob_parity(np.array([-0.1, -0.2]), np.array([-0.11, -0.19]))
    tf = np.array([-0.1, -0.2, -0.3, -0.4])
    out = assert_rollout_synced(s, s, tf, tf.copy())                        # no movement on fixed seq
    assert out["policy_moved"] is False and out["ok"] is False

def test_teacher_forced_shape_must_match():
    s = logprob_parity(np.array([-0.1]), np.array([-0.1]))
    import pytest
    with pytest.raises((ValueError, AssertionError)):
        assert_rollout_synced(s, s, np.array([-0.1, -0.2]), np.array([-0.1]))  # not a fixed seq
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rollout_sync_guard.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.rollout_sync_guard'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rollout_sync_guard.py
"""B0 guard: verify the vLLM rollout policy tracks the actor via per-token LOGPROB parity.

Weight hashing is unusable here (vLLM weights are TP-sharded/fused, no actor-comparable API, ~36GB).
veRL logs rollout logprobs (vLLM) and the actor's recomputed old_log_prob for the same tokens;
if they diverge, vLLM is serving different weights than the actor holds (the pre-#7014 stale base).
"""
from __future__ import annotations

import numpy as np


def logprob_parity(rollout_lp, actor_lp, atol: float = 0.1) -> dict:
    """Max abs diff between rollout and actor per-token logprobs for the same tokens."""
    r = np.asarray(rollout_lp, dtype=np.float64)
    a = np.asarray(actor_lp, dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(r - a))) if r.size else float("inf")
    return {"max_abs_diff": max_abs_diff, "close": max_abs_diff <= atol}


def assert_rollout_synced(step0: dict, step1: dict, tf_lp0, tf_lp1, move_atol: float = 1e-3) -> dict:
    """ok iff rollout≈actor at BOTH steps (synced) AND the actor moved on a FIXED teacher-forced seq.

    step0/step1 : within-step logprob_parity dicts (rollout vs actor on the SAME sampled tokens).
    tf_lp0/tf_lp1: actor teacher-forced logprobs on ONE fixed prompt+continuation at step0 & step1 —
                   identical tokens, so the cross-step delta is meaningful. (Do NOT diff sampled
                   rollout logprobs across steps: different tokens/shapes -> meaningless.)
    """
    a0 = np.asarray(tf_lp0, dtype=np.float64)
    a1 = np.asarray(tf_lp1, dtype=np.float64)
    if a0.shape != a1.shape:
        raise ValueError(f"teacher-forced logprobs must be the same fixed sequence: {a0.shape} vs {a1.shape}")
    synced = bool(step0["close"] and step1["close"])
    policy_moved = bool(a0.size and float(np.max(np.abs(a0 - a1))) > move_atol)
    return {"synced": synced, "policy_moved": policy_moved, "ok": synced and policy_moved}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rollout_sync_guard.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/rollout_sync_guard.py tests/test_rollout_sync_guard.py
git commit -m "feat: B0 rollout-sync guard via logprob parity (catches pre-#7014 stale rollout)"
```

> **Wiring note:** these arrays are NOT in veRL's default reward dump — an explicit instrumentation
> hook (Task 10, Step 3b) must capture them. `calculate_log_probs=True` attaches vLLM's generation
> logprobs to the batch and the actor's `old_log_prob` is computed every PPO step (same tokens →
> within-step parity); the hook additionally runs a teacher-forced pass on ONE fixed
> prompt+continuation at step0 & step1 (→ movement) and writes `rollout_sync.json` via this helper.
> Complement with the cheap **IPC-side fingerprint**: a hash of the small set of merged tensors veRL
> pushes to vLLM in `update_weights` (not all 9B params) — independent evidence the payload is
> non-empty and changing.

---

### Task 9: Build the pinned Arm-B env (candidate stack)

Cluster op. Mirrors the `turing-rl-sft-qwen35` pattern: a dedicated env, never touching
`turing-rl-train`.

**Files:**
- Create: `scripts/slurm/rl_qwen35_env_install.sh` (documents the exact build, like `train_env_install.sh`)

- [ ] **Step 1: Write the install script** — executable, ordered, no free pip-resolve:

```bash
#!/bin/bash
# Build the dedicated Arm-B GRPO env for Qwen3.5-9B. CANDIDATE STACK — validated only by B0.
# Never modify turing-rl-train. Ordered install; --no-deps where noted to stop vllm/transformers
# metadata from clobbering our pins. V3: unset stale proxy vars; use ~/tmp for builds.
set -euo pipefail
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

ENV=turing-rl-rl-qwen35
VERL_SHA=c791da0bfcd7d7b560b1e461d2c188145b39c353  # FULL SHA; contains #7014 + #5599
VERL_DIR=/home/lancewicki/src/verl
CONDA=/home/lancewicki/miniconda3
PY=$CONDA/envs/$ENV/bin/python
PIP="$PY -m pip"

# 0. Fresh env (Python 3.11; do NOT clone turing-rl-train — keep it pristine).
$CONDA/bin/conda create -y -n "$ENV" python=3.11
$PIP install -U pip setuptools wheel packaging ninja

# 1. Clone veRL at the pinned SHA and read ITS torch/CUDA + flash-attn pins (source of truth).
[ -d "$VERL_DIR" ] || git clone https://github.com/verl-project/verl "$VERL_DIR"
git -C "$VERL_DIR" fetch --all && git -C "$VERL_DIR" checkout "$VERL_SHA"
echo ">> veRL pins (install EXACTLY these next; edit the literals below if they differ):"
grep -iRhoE '(torch(vision|audio)?|flash-attn|vllm)[=<>~!]=[0-9][^ ]*' "$VERL_DIR"/requirements*.txt "$VERL_DIR"/setup.py 2>/dev/null | sort -u || true

# 2. Torch FIRST — vLLM 0.20.2 REQUIRES torch 2.11.0 / CUDA 13 (NOT 2.6; that is an ABI mismatch).
#    Include torchaudio (some veRL data utils import it). All exact, no ranges.
$PIP install "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" --index-url https://download.pytorch.org/whl/cu130

# 3. Kernels built against torch 2.11 (order matters, --no-build-isolation). Exact versions.
$PIP install --no-build-isolation "flash-attn==2.8.3"
$PIP install --no-build-isolation "causal-conv1d==1.5.0.post8"
$PIP install "flash-linear-attention==0.5.1"

# 4. vLLM 0.20.2 with its FULL deps (torch 2.11 already satisfied so it won't fight).
$PIP install "vllm==0.20.2"

# 5. veRL from source with its complete requirements (do NOT --no-deps here — install the full set).
$PIP install -r "$VERL_DIR/requirements.txt"
$PIP install -e "$VERL_DIR"

# 6. Force transformers==5.4.0 LAST, --no-deps, so it overrides whatever vllm/veRL resolved
#    (5.4.0 is the GDN-crash workaround, veRL #6549). This is the candidate-stack gamble — B0 validates it.
$PIP install --no-deps --force-reinstall "transformers==5.4.0"

# 7. turing-rl runtime deps not covered above (exact, match the train env where they overlap).
$PIP install "peft==0.17.1" "openai==1.58.1"

# 8. Freeze the FULL resolved env for reproducibility (commit this artifact alongside the script).
$PIP freeze > "/home/lancewicki/projects/turing-rl/docs/superpowers/plans/arm_b_env_freeze.txt"
$PY -c "import torch,vllm,transformers,fla; print('VERSIONS', torch.__version__, vllm.__version__, transformers.__version__, fla.__version__)" \
  | tee -a "/home/lancewicki/projects/turing-rl/docs/superpowers/plans/arm_b_env_freeze.txt"
echo ">> built $ENV @ veRL $VERL_SHA — freeze saved; verify with Step 2."
```

> Build this env **on a compute node** (`srun --partition=a100 --gres=gpu:1 --time=02:00:00 --pty bash`),
> not the login node: flash-attn / causal-conv1d compile CUDA kernels and need the toolchain + a GPU
> present. Commit `docs/superpowers/plans/arm_b_env_freeze.txt` so the exact resolved stack is recorded.

> The `torch==2.11.0/cu130` and `flash-attn==2.8.3` literals are the vLLM-0.20.2-required stack;
> **cross-check against step-1's printed pins** and adjust if the pinned veRL differs. A torch/vLLM
> ABI mismatch here is the #1 silent failure mode; the transformers-5.4.0 force-reinstall (step 6)
> is deliberately last so vLLM's own transformers pin can't win.

- [ ] **Step 2: Build the env on the cluster** following the script, then record exact installed
versions:

Run: `ssh -p 2223 … lancewicki@localhost '/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python -c "import verl,vllm,transformers,fla; print(verl.__version__, vllm.__version__, transformers.__version__)"'`
Expected: prints veRL (≥ c791da0b build), vllm 0.20.2, transformers 5.4.0. Record in `.sdd-progress.md`.

- [ ] **Step 3: Verify vLLM can serve the arch** (static, before GRPO). Run **inside a Slurm GPU
alloc** (TP=4 → 4 GPUs), with a PID + trap so the server is always killed. Confirm the 9B merged
path first (it may have been deleted for quota — see Step 3.0).

- [ ] **Step 3.0: Confirm/rebuild the 9B merged checkpoint.** The trajectory README notes the 9B
`merged_ep{1,2,3}` (~19GB each) may have been deleted to reclaim quota.
Run: `ssh … 'ls -d /home/lancewicki/projects/turing-rl/checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3'`
If missing, re-splice from `checkpoint-78` via `scripts/splice_sft_into_full.py` (per the trajectory
README repro, in env `turing-rl-sft-qwen35`), then re-check.

- [ ] **Step 3.1: Disable MTP in the merged checkpoint config (single source of truth).** The 9B
`config.json` has `text_config.mtp_num_hidden_layers=1`; MTP is unresolved under FSDP (veRL #6483)
and must be off. Patch the on-disk config so **actor, ref, AND the vLLM rollout** (all load from this
dir) see 0 — the launcher's `override_config` (Step 3) is only a belt for the actor:

```bash
ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && M=checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3 && python - <<PY
import json, pathlib
p=pathlib.Path("'"'"'$M'"'"'")/"config.json"; c=json.loads(p.read_text())
tc=c.get("text_config", c)
tc["mtp_num_hidden_layers"]=0
if "num_nextn_predict_layers" in tc: tc["num_nextn_predict_layers"]=0   # alt key name, if present
p.write_text(json.dumps(c, indent=2)); print("MTP layers set to 0 in", p)
PY'
```
Expected: prints the patched path. (Confirm the exact key name in the checkpoint's `config.json`
first — it is `mtp_num_hidden_layers` or `num_nextn_predict_layers` depending on the arch version.)

```bash
ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && srun --partition=a100 --account=rfai --gres=gpu:4 --time=00:30:00 bash -lc "
set -e
M=checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/vllm serve \$M --tensor-parallel-size 4 --enforce-eager --port 8000 &
SRV=\$!; trap \"kill \$SRV 2>/dev/null || true\" EXIT
for i in \$(seq 1 60); do curl -sf localhost:8000/v1/models && break || sleep 10; done
curl -s localhost:8000/v1/models
"'
```
Expected: the model id is listed (arch `Qwen3_5ForConditionalGeneration` served); the trap kills the
server on exit. If vLLM cannot load the arch, the pinned stack is wrong — fix before Task 10.

- [ ] **Step 4: Commit the install script**

```bash
git add scripts/slurm/rl_qwen35_env_install.sh
git commit -m "feat: Arm-B pinned candidate env install (veRL c791da0b + vllm 0.20.2 + tf 5.4)"
```

---

### Task 10: Arm-B launcher + B0 feasibility spike (GATE)

Create the 9B GRPO launcher and run the gated spike. Fold config, env wiring, and the parity check
into one task — a reviewer accepts/rejects "can we GRPO the 9B at all" as a unit.

**Files:**
- Create: `scripts/slurm/rl_generator_train_9b.sh` (9B variant of `rl_generator_train.sh`)
- Create: `scripts/slurm/rl_generator_run_9b.sh` (2-node driver; or parameterize the existing driver)
- Create: `training/grpo/b0_rollout_sync_hook.py` (Step 3b — captures the logprob-parity data path)
- Modify: `training/grpo/run_verl_main_ppo.py` (guarded `if os.environ.get("B0_ROLLOUT_SYNC")` call)
- Test: `tests/test_rl_9b_launcher.py` (static assertions on the launcher text)

**Interfaces:**
- Consumes: the merged 9B checkpoint (`merged_ep3`), `scripts/rollout_sync_guard.py`,
  `scripts/slurm/judge_serve_9b_replicas.sh` (same frozen 9B judge).
- Produces: reward dumps under `results/grpo/rl-generator/9b_b0_spike/reward_dump/` + a
  `rollout_sync.json` written by the spike (the B0 gate artifact: logprob parity @ step0/step1 +
  policy-moved).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_9b_launcher.py
import pathlib
S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()

def test_lora_target_is_attn_mlp_not_all_linear_excludes_visual_and_mtp():
    assert "all-linear" not in S                       # never LoRA the GDN backbone
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert m in S
    assert "visual" in S and "mtp" in S                # exclude vision tower AND MTP head
    assert "lora_rank=64" in S and "lora_alpha=32" in S

def test_mtp_disabled_via_override_config():
    assert "override_config.text_config.mtp_num_hidden_layers=0" in S

def test_merge_key_is_model_lora_merge():
    # correct veRL key is model.lora.merge (Hydra-appended), NOT rollout.lora.merge
    assert "actor_rollout_ref.model.lora.merge=True" in S
    assert "rollout.lora.merge" not in S

def test_offload_and_cache_clear():
    assert "param_offload=True" in S
    assert "optimizer_offload=True" in S
    assert "actor.fsdp_config.offload_policy=True" in S       # FSDP2-specific offload policy
    assert "ref.fsdp_config.offload_policy=True" in S
    assert "free_cache_engine=True" in S and "enforce_eager=True" in S

def test_checkpoint_engine_override_has_no_plus_prefix():
    # key already exists in current veRL -> `+` would error "already exists"
    assert "checkpoint_engine.update_weights_bucket_megabytes=3072" in S
    assert "+actor_rollout_ref.rollout.checkpoint_engine" not in S

def test_required_fsdp2_and_qwen35_overrides():
    for k in (
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.ref.strategy=fsdp2",
        "actor_rollout_ref.actor.fsdp_config.fsdp_size=8",
        "actor_rollout_ref.actor.use_dynamic_bsz=False",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        "actor_rollout_ref.rollout.max_model_len=13524",
        "actor_rollout_ref.rollout.calculate_log_probs=True",   # feeds the B0 logprob guard
        "checkpoint_engine.update_weights_bucket_megabytes=3072",
    ):
        assert k in S, f"missing required override: {k}"

def test_uses_merged_9b_and_no_cap():
    assert "merged_ep3" in S
    assert "TURING_JUDGE_SCORE_CLIP_MAX=7" in S or "cap" in S.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_9b_launcher.py -q`
Expected: FAIL — `FileNotFoundError: scripts/slurm/rl_generator_train_9b.sh`.

- [ ] **Step 3: Write the 9B trainer launcher**

Copy `rl_generator_train.sh` → `rl_generator_train_9b.sh` and make exactly these edits (do NOT keep
the 8B `OVR=(...)` array or its trainer invocation — **replace** them with the block below, so there
is a single `OVR` and a single `run_verl_main_ppo` call). Keep the header, the merged-dir existence
guards, `DATA_BASE`/`TRAIN_FILE`/`VAL_FILE`, and the `MODE=overfit` branch identical to the 8B
version; keep the reward-env inheritance identical. Change `PY` and `MERGED_SFT_MODEL_PATH` at top:

```bash
# scripts/slurm/rl_generator_train_9b.sh  — edits vs the 8B version
PY=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python   # pinned Arm-B env
MERGED_SFT_MODEL_PATH=${MERGED_SFT_MODEL_PATH:-checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}

# ---- REPLACES the 8B OVR array (single definition) ----
# `+key=...` appends keys not present in qwen3_8b_grpo_turing.yaml (lora.merge, strategy, chunked).
OVR=(
  actor_rollout_ref.model.path="$MERGED_SFT_MODEL_PATH"
  actor_rollout_ref.model.lora_adapter_path=null
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  actor_rollout_ref.model.exclude_modules='.*(visual|mtp).*'   # skip vision tower AND MTP head
  +actor_rollout_ref.model.override_config.text_config.mtp_num_hidden_layers=0  # belt: disable MTP on actor/ref
  +actor_rollout_ref.model.lora.merge=True                 # merged dense sync (NOT rollout.lora.merge)
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  # FSDP2 + Qwen3.5/GDN requirements (from veRL's 27B FSDP2 recipe) --------------------------------
  actor_rollout_ref.actor.strategy=fsdp2
  actor_rollout_ref.ref.strategy=fsdp2
  actor_rollout_ref.actor.fsdp_config.fsdp_size=8
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.fsdp_config.offload_policy=True    # FSDP2 offload policy (official 27B recipe)
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  actor_rollout_ref.ref.fsdp_config.offload_policy=True
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
  # rollout (vLLM) --------------------------------------------------------------------------------
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.free_cache_engine=True
  actor_rollout_ref.rollout.enforce_eager=True
  actor_rollout_ref.rollout.enable_prefix_caching=False
  actor_rollout_ref.rollout.enable_chunked_prefill=True   # REQUIRED: prompts ~12.5k > 4096 batch cap
  actor_rollout_ref.rollout.max_model_len=13524
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40
  actor_rollout_ref.rollout.calculate_log_probs=True       # feeds the B0 logprob-parity guard
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072   # no + (key exists)
  # trainer / data --------------------------------------------------------------------------------
  trainer.default_local_dir="$CKPT_DIR"
  trainer.n_gpus_per_node=8
  trainer.nnodes=1
  data.train_files="$TRAIN_FILE"
  data.val_files="$VAL_FILE"
)
# TURING_JUDGE_SCORE_CLIP_MAX=7 and PERSONA_* inherited from the driver, unchanged.
$PY -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs \
  --config-name qwen3_8b_grpo_turing "${OVR[@]}" ${EXTRA_OVERRIDES:-}
```

Also create `scripts/slurm/rl_generator_run_9b.sh` as a copy of `rl_generator_run.sh` that srun-launches
`rl_generator_train_9b.sh` (trainer) alongside `judge_serve_9b_replicas.sh` (frozen judge, unchanged).

> **Validate the exact Hydra key names against the pinned veRL SHA** (Task 9 Step 3): if the pinned
> veRL renamed any of `lora.merge`, `strategy`, `enable_chunked_prefill`,
> `checkpoint_engine.update_weights_bucket_megabytes`, adjust the key while keeping the semantics.
> The `+` prefix is required only for keys absent from the base yaml — drop it for keys already
> present (Hydra errors on `+` for an existing key).

- [ ] **Step 3b: Write the B0 rollout-sync instrumentation hook (the data path)**

The guard from Task 8 is pure logic — nothing yet *produces* its inputs. veRL's default reward dump
does NOT contain aligned rollout/actor logprob arrays, and `run_verl_main_ppo.py` never owns the
actor or the batch (it just builds config + launches Ray). So the hook must attach at the trainer:

**Integration point (concrete):** in `run_verl_main_ppo.py`, AFTER `apply_verl_runtime_patch()` and
BEFORE `trainer.fit()`, when `os.environ.get("B0_ROLLOUT_SYNC")` is set, monkeypatch
`verl.trainer.ppo.ray_trainer.RayPPOTrainer.update_actor` (the method that runs the optimizer step;
confirm the exact name in the pinned veRL — it may be `_update_actor`/`update_policy`) with a wrapper
that, for the first two calls only, captures the signals below from the `DataProto` batch it receives
and then calls the original.

**Files:** Create `training/grpo/b0_rollout_sync_hook.py` (the wrapper + `write_rollout_sync(run_dir)`);
add the guarded monkeypatch in `training/grpo/run_verl_main_ppo.py`.

What the wrapper captures, at update-call 0 and 1 only:
1. **Within-step parity:** from the batch `DataProto`, read vLLM's generation logprobs
   (`batch.batch["rollout_log_probs"]`, present because `calculate_log_probs=True`) and the actor's
   recomputed `batch.batch["old_log_probs"]` for the SAME response tokens; mask via
   `attention_mask`/`response_mask`; call `logprob_parity(rollout_lp, actor_lp)` → `step0`/`step1`.
2. **Movement (teacher-forced):** call the **actor worker group's** logprob API
   (`actor_rollout_wg.compute_log_prob` on a FIXED prompt+continuation `DataProto`, hard-coded token
   ids saved once) at call 0 and call 1 → `tf_lp0`, `tf_lp1` — identical tokens, comparable.
3. `assert_rollout_synced(step0, step1, tf_lp0, tf_lp1)` → `write_rollout_sync` dumps
   `rollout_sync.json` into `RL_RUN_DIR`.
4. **Optional** IPC fingerprint: only if a clean insertion exists in the pinned veRL's
   `checkpoint_engine`/`update_weights` path (hash the small merged-tensor payload, not 36GB). If no
   clean hook point, **omit it** — the logprob signals (1–2) are the gate; do not fake it.

Confirm the exact veRL symbol names (`update_actor`, `rollout_log_probs`, `old_log_probs`,
`compute_log_prob`, `apply_verl_runtime_patch`) against `$VERL_DIR` at the pinned SHA and adjust;
keep the guard-call semantics fixed. No new unit test (exercised live in Step 7); the pure inputs are
covered by `tests/test_rollout_sync_guard.py`.

- [ ] **Step 4: Run the launcher test to verify it passes**

Run: `python -m pytest tests/test_rl_9b_launcher.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit the launcher + instrumentation**

```bash
git add scripts/slurm/rl_generator_train_9b.sh scripts/slurm/rl_generator_run_9b.sh training/grpo/b0_rollout_sync_hook.py training/grpo/run_verl_main_ppo.py tests/test_rl_9b_launcher.py
git commit -m "feat: Arm-B 9B GRPO launcher + B0 rollout-sync instrumentation (logprob parity data path)"
```

- [ ] **Step 5b: Hydra composition smoke (before any GPU run).** Static substring tests do not prove
the overrides actually compose. Dry-resolve the full config with the pinned Arm-B env and inspect the
Qwen3.5-sensitive inherited settings:

```bash
ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && OVR="$(sed -n "s/^  \(actor_rollout_ref[^ ]*\|trainer[^ ]*\|data[^ ]*\|+[^ ]*\).*/\1/p" scripts/slurm/rl_generator_train_9b.sh)"; /home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python -m training.grpo.run_verl_main_ppo --config-dir training/grpo/configs --config-name qwen3_8b_grpo_turing $OVR --cfg job --resolve 2>&1 | grep -iE "lora|merge|strategy|attn_implementation|remove_padding|chunked_prefill|max_model_len|mtp|target_modules|exclude_modules|offload"'
```
Expected: resolves with NO Hydra error; every override present at its intended value; verify
Qwen3.5's inherited `attn_implementation` and `use_remove_padding` are sane (remove-padding off unless
flash-attn path confirmed). Fix any mis-resolved key before spending GPUs.

- [ ] **Step 5c: Verify MTP is actually off in the INSTANTIATED config** (Hydra grep does NOT inspect
the loaded HF config — MTP lives there). Confirm the patched checkpoint reports 0:

```bash
ssh -p 2223 … lancewicki@localhost 'cd /home/lancewicki/projects/turing-rl && /home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python - <<PY
from transformers import AutoConfig
c=AutoConfig.from_pretrained("checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3", trust_remote_code=True)
tc=getattr(c,"text_config",c)
n=getattr(tc,"mtp_num_hidden_layers", getattr(tc,"num_nextn_predict_layers", None))
print("mtp layers =", n); assert n in (0, None), f"MTP still on: {n}"
print("MTP OFF OK")
PY'
```
Expected: `mtp layers = 0` (or None) and `MTP OFF OK`. If nonzero, re-run Step 3.1 with the correct
key name before B0.

- [ ] **Step 6: Deploy + run the B0 spike (5–10 updates) on the cluster**

10 overfit rows at batch 10 = **1 optimizer update per epoch**, so use `OVERFIT_EPOCHS=8` (→ 8
updates), not 3. Run (Mac): `scripts/sync_to_cluster.sh` then
`ssh … 'cd … && preflight then B0_ROLLOUT_SYNC=1 JUDGE=9b MODE=overfit OVERFIT_EPOCHS=8 RUN_TAG=9b_b0_spike sbatch --export=ALL scripts/slurm/rl_generator_run_9b.sh'`
Expected: job queued; log shows judge endpoint + trainer banner in the Arm-B env. `B0_ROLLOUT_SYNC=1`
turns on the Step-3b hook so `rollout_sync.json` is written.

- [ ] **Step 7: Evaluate the B0 gate**

**Hard gate (both MUST hold):**
1. Steps complete, no NaN/crash, `reward_dump/*.jsonl` populated, generations terminate
   (char lengths bounded — reuse the length check from the trajectory README).
2. **Rollout sync (logprob parity) — the reliable gate:** `rollout_sync.json` shows `ok: true` from
   `assert_rollout_synced` — vLLM rollout logprobs match the actor's `old_log_probs` (within tol) at
   BOTH update 0 and update 1 AND the teacher-forced logprobs moved after the update. Needs
   `calculate_log_probs=True` (Step 3) + the Step-3b hook. **A large parity gap = the pre-#7014
   stale-base symptom → gate FAILS.**

**Soft signal (record, do NOT hard-gate on it):** reward direction over ~8 updates — too noisy to
gate on at this scale; a flat/negative reward with `ok: true` sync is still a PASS (optimization
tuning, not a plumbing failure). Record the verdict + both signals in `.sdd-progress.md`.

- [ ] **Step 8: Gate decision.**
   - **Pass** → proceed to Task 11.
   - **merge=True fails** → try full-param FSDP2 FT (mark any resulting 9B run **exploratory**, not
     part of H2 — per the spec) and re-evaluate the gate.
   - **Both fail within the time box** → STOP Phase 2; write an Arm-B-deferred note in the results
     README with the exact failure + upstream issue links. Arm A stands alone.

---

### Task 11: Arm-B overfit grid + analysis (only if B0 passed)

Mirror Arm A on the 9B, same judge, same grid — the controlled H2 comparison.

**Files:**
- Create: `scripts/slurm/submit_arm_b_grid.sh` (copy of Task 3's submitter → `ARM_B_CELLS`,
  `rl_generator_run_9b.sh`, 9B merged path)
- Test: `tests/test_submit_arm_b_grid.py` (assert `ARM_B_CELLS`, `9b_proper_*`, `rl_generator_run_9b.sh`)
- Extend: `results/2026-07-24-reward-hack-proper-checkpoint/README.txt`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_submit_arm_b_grid.py
import pathlib
S = pathlib.Path("scripts/slurm/submit_arm_b_grid.sh").read_text()

def test_uses_arm_b_cells_and_9b_launcher():
    assert "ARM_B_CELLS" in S
    assert "rl_generator_run_9b.sh" in S
    assert "merged_ep3" in S and "OVERFIT_EPOCHS=50" in S
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_submit_arm_b_grid.py -q`
Expected: FAIL — `FileNotFoundError`.

- [ ] **Step 3: Write the submitter** (copy `submit_arm_a_grid.sh`, swap the module import to
`ARM_B_CELLS`, the launcher to `rl_generator_run_9b.sh`, and `MERGED` to the 9B `merged_ep3`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest tests/test_submit_arm_b_grid.py -q`
Expected: PASS.

- [ ] **Step 5: Deploy, preflight, submit** (Mac `scripts/sync_to_cluster.sh`; cluster
`bash scripts/slurm/submit_arm_b_grid.sh`). Stage to stay ≤10 concurrent; lr=1e-4 cells first.

- [ ] **Step 6: Analyze** (reuse Task 7 steps: `overfit_gate_check.py` + `plot_overfit_ratings.py`
per `9b_proper_*` tag; pull PNGs locally).

- [ ] **Step 7: Write the H2 comparison** into the results README: 8B vs 9B, per cell — peak
overfit win-rate and the first (KL, LR) cell each clears ≥8/10. State the H2 verdict (does the
larger/DeltaNet 9B game the frozen judge more readily?).

- [ ] **Step 8: Commit**

```bash
git add scripts/slurm/submit_arm_b_grid.sh tests/test_submit_arm_b_grid.py
git add -f results/2026-07-24-reward-hack-proper-checkpoint/README.txt results/2026-07-24-reward-hack-proper-checkpoint/*.png
git commit -m "results: Arm-B 9B overfit grid + 8B-vs-9B H2 comparison"
```

---

## Final: PR

- [ ] Run the full local test suite: `python -m pytest tests/test_rl_grid.py tests/test_proper_checkpoint.py tests/test_submit_arm_a_grid.py tests/test_rollout_sync_guard.py tests/test_rl_9b_launcher.py tests/test_submit_arm_b_grid.py -q` → all PASS.
- [ ] Push the branch and open a **draft** PR (`gh pr create --draft`) with a summary linking the
  spec + this plan. Never push to `main`/`lancewicki/main` directly; additive commits only.

---

## Notes for the implementer

- **Cluster reads:** use SSH `cat`/`ls`, not the Read tool. Tunnel: `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<cmd>"`. If refused, ask the user to refresh the tunnel.
- **Arm A is almost pure config** — the 2026-07-15 machinery is unchanged; the only substantive
  cluster op is the checkpoint-78 merge (Task 4). If a run misbehaves, the decisions doc
  `2026-07-15-rl-generator-vs-fixed-judge/decisions.md` documents every prior gotcha (PYTHONPATH,
  wandb .env, epoch-end checkpoint quota, KL-ref merge).
- **Arm B is the risk.** Time-box the env pinning (~1–2 days). The B0 rollout-sync gate (logprob
  parity, Task 8) is non-negotiable — "no crash" is not enough (pre-#7014 served the base policy
  silently, crash-free). Do NOT weight-hash: vLLM weights are TP-sharded/fused and ~36GB.
- **veRL override names** (`lora.merge`, `target_modules`, `exclude_modules`, offload flags) must be
  validated against the pinned veRL SHA's config schema during Task 10 Step 3 — adjust the exact
  Hydra keys if the pinned veRL uses different names, keeping the semantics fixed.
