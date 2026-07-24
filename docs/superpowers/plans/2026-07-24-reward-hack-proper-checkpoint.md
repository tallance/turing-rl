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
- **LoRA target (both arms):** attention + MLP only (`q/k/v/o_proj`, `gate/up/down_proj`).
  **Never `all-linear`** — it hits the Gated-DeltaNet backbone (`in_proj_*`, `out_proj`), which is
  destructive (arXiv:2604.22127). Arm B additionally `exclude_modules='.*visual.*'`.
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

A thin, idempotent submitter that loops `ARM_A_CELLS`, pointing every run at the proper merged
checkpoint. No new training logic — just the correct env for `rl_generator_run.sh`.

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

# Emit "tag<TAB>overrides" per cell from the SSOT grid module.
$PY - <<'PYEOF' | while IFS=$'\t' read -r TAG OVR; do
from scripts.rl_grid import ARM_A_CELLS, cell_overrides
for c in ARM_A_CELLS:
    print(f"{c['tag']}\t{cell_overrides(c)}")
PYEOF
  echo ">> submitting $TAG :: $OVR"
  JUDGE=9b MODE=overfit OVERFIT_EPOCHS=50 \
    RUN_TAG="$TAG" \
    MERGED_SFT_MODEL_PATH="$MERGED" \
    EXTRA_OVERRIDES="$OVR" \
    sbatch --export=ALL scripts/slurm/rl_generator_run.sh
done
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_submit_arm_a_grid.py -q`
Expected: PASS (2 passed).

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
`sft_merge_metadata.json`, and `model*.safetensors`; the script's `validate_merged_artifact` prints
OK. (This is the merged-SFT-ref parity guard — the merge script validates the artifact.)

- [ ] **Step 4: Sanity-check the merged dir shape**

Run: `ssh -p 2223 … lancewicki@localhost 'ls /home/lancewicki/projects/turing-rl/checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3'`
Expected: contains `config.json tokenizer_config.json sft_merge_metadata.json` + `model-*.safetensors`.
(This is exactly what `rl_generator_train.sh` asserts before launching.)

- [ ] **Step 5: (No commit — cluster artifact.)** Record the confirmed path in the run ledger
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

- [ ] **Step 5: Commit the README + plots** (per project convention: `results/` is gitignored on
the cluster, but the local plan-results dir + README are committed for the record).

```bash
git add results/2026-07-24-reward-hack-proper-checkpoint/README.txt
git commit -m "results: Arm-A proper-checkpoint overfit grid (H1 verdict + buggy-vs-clean table)"
```

---

# Phase 2 — Arm B (Qwen3.5-9B generator), new stack, gated

> Do NOT start Phase 2 until Arm A jobs are submitted (Task 6). Arm A does not depend on any Phase-2
> work. Phase 2 gates on the B0 spike (Task 10).

### Task 8: B0 weight-sync parity helper + unit test

The critical B0 assertion: rollout weights must track the actor (the pre-#7014 bug ran crash-free
while vLLM served the base policy). Build a pure helper now (unit-testable offline); wire it into
the spike in Task 10.

**Files:**
- Create: `scripts/rollout_weight_parity.py`
- Test: `tests/test_rollout_weight_parity.py`

**Interfaces:**
- Produces:
  `param_fingerprint(named_tensors: Iterable[tuple[str, "Tensor"]]) -> str` — order-independent
  hash of parameter values (float32 bytes), used to compare an actor state-dict vs the weights the
  rollout engine reports.
  `assert_weights_tracked(before_fp, after_fp, actor_after_fp) -> dict` — returns
  `{"changed": bool, "matches_actor": bool, "ok": bool}` where `ok = changed and matches_actor`
  (rollout weights changed after the optimizer step AND equal the post-step actor).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rollout_weight_parity.py
import numpy as np
from scripts.rollout_weight_parity import param_fingerprint, assert_weights_tracked

class T:  # minimal tensor stand-in: .detach().cpu().float().numpy() chain
    def __init__(self, a): self._a = np.asarray(a, dtype=np.float32)
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self
    def numpy(self): return self._a

def test_fingerprint_order_independent_and_value_sensitive():
    a = [("w2", T([3.0, 4.0])), ("w1", T([1.0, 2.0]))]
    b = [("w1", T([1.0, 2.0])), ("w2", T([3.0, 4.0]))]
    assert param_fingerprint(a) == param_fingerprint(b)          # order-independent
    c = [("w1", T([1.0, 2.0])), ("w2", T([3.0, 4.001]))]
    assert param_fingerprint(a) != param_fingerprint(c)          # value-sensitive

def test_ok_only_when_changed_and_matches_actor():
    before, after, actor = "aaa", "bbb", "bbb"
    r = assert_weights_tracked(before, after, actor)
    assert r == {"changed": True, "matches_actor": True, "ok": True}

def test_stale_rollout_flagged():   # the pre-#7014 bug: rollout never changed
    r = assert_weights_tracked("aaa", "aaa", "bbb")
    assert r["changed"] is False and r["ok"] is False

def test_desync_flagged():          # rollout changed but not to the actor's weights
    r = assert_weights_tracked("aaa", "ccc", "bbb")
    assert r["matches_actor"] is False and r["ok"] is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rollout_weight_parity.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.rollout_weight_parity'`.

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/rollout_weight_parity.py
"""B0 guard: verify GRPO rollout weights actually track the actor (catches pre-#7014 stale base)."""
from __future__ import annotations

import hashlib
from typing import Iterable, Tuple


def param_fingerprint(named_tensors: Iterable[Tuple[str, "object"]]) -> str:
    """Order-independent, value-sensitive hash of (name -> tensor) params."""
    h = hashlib.sha256()
    for name, t in sorted(named_tensors, key=lambda kv: kv[0]):
        arr = t.detach().cpu().float().numpy()
        h.update(name.encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def assert_weights_tracked(before_fp: str, after_fp: str, actor_after_fp: str) -> dict:
    """ok iff rollout weights CHANGED after the optimizer step AND equal the post-step actor."""
    changed = before_fp != after_fp
    matches_actor = after_fp == actor_after_fp
    return {"changed": changed, "matches_actor": matches_actor, "ok": changed and matches_actor}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_rollout_weight_parity.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add scripts/rollout_weight_parity.py tests/test_rollout_weight_parity.py
git commit -m "feat: B0 rollout-vs-actor weight-parity guard (catches pre-#7014 stale rollout)"
```

---

### Task 9: Build the pinned Arm-B env (candidate stack)

Cluster op. Mirrors the `turing-rl-sft-qwen35` pattern: a dedicated env, never touching
`turing-rl-train`.

**Files:**
- Create: `scripts/slurm/rl_qwen35_env_install.sh` (documents the exact build, like `train_env_install.sh`)

- [ ] **Step 1: Write the install script** capturing the pinned stack (veRL Docker build order):

```bash
#!/bin/bash
# Build the dedicated Arm-B GRPO env for Qwen3.5-9B. CANDIDATE STACK — validated only by B0.
# Never modify turing-rl-train. Follow veRL's Docker build order; do NOT let pip free-resolve.
set -euo pipefail
export TMPDIR=/home/lancewicki/tmp/build PIP_CACHE_DIR=/home/lancewicki/tmp/pip-cache
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"
ENV=turing-rl-rl-qwen35
VERL_SHA=c791da0b   # MUST contain #7014 (merged-weight sync) + #5599 (Qwen3.5 GDN mappings)
# 1. clone env base, 2. pin: transformers==5.4.0, vllm==0.20.2, flash-linear-attention==0.5.1,
#    causal-conv1d, flash-attn (from veRL image), 3. install veRL @ $VERL_SHA from source.
# (Fill exact conda/pip lines per the SFT-env recipe in results/2026-07-15-generator-sweep/README.txt.)
echo "See docstring — build $ENV with veRL@$VERL_SHA, vllm 0.20.2, transformers 5.4.0, FLA 0.5.1."
```

- [ ] **Step 2: Build the env on the cluster** following the script, then record exact installed
versions:

Run: `ssh -p 2223 … lancewicki@localhost '/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python -c "import verl,vllm,transformers,fla; print(verl.__version__, vllm.__version__, transformers.__version__)"'`
Expected: prints veRL (≥ c791da0b build), vllm 0.20.2, transformers 5.4.0. Record in `.sdd-progress.md`.

- [ ] **Step 3: Verify vLLM can serve the arch** (static, before GRPO): serve `merged_ep3` (9B) and
hit `/v1/models`.

Run: `ssh … 'cd … && <env>/bin/vllm serve checkpoints/sft/<qwen35_9b_epochsave>/merged_ep3 --tensor-parallel-size 4 --enforce-eager & sleep 240; curl -s localhost:8000/v1/models'`
Expected: the model id is listed (arch `Qwen3_5ForConditionalGeneration` served). Kill the server.

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
- Test: `tests/test_rl_9b_launcher.py` (static assertions on the launcher text)

**Interfaces:**
- Consumes: the merged 9B checkpoint (`merged_ep3`), `scripts/rollout_weight_parity.py`,
  `scripts/slurm/judge_serve_9b_replicas.sh` (same frozen 9B judge).
- Produces: reward dumps under `results/grpo/rl-generator/9b_b0_spike/reward_dump/` + a
  `weight_parity.json` written by the spike (the B0 gate artifact).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rl_9b_launcher.py
import pathlib
S = pathlib.Path("scripts/slurm/rl_generator_train_9b.sh").read_text()

def test_lora_target_is_attn_mlp_not_all_linear_excludes_visual():
    assert "all-linear" not in S                       # never LoRA the GDN backbone
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert m in S
    assert "visual" in S                               # exclude the vision tower
    assert "lora_rank=64" in S and "lora_alpha=32" in S

def test_merge_true_and_offload_and_cache_clear():
    assert "merge=true" in S.lower() or "lora.merge=true" in S.lower()
    assert "param_offload=true" in S.lower() or "param_offload=True" in S
    assert "free_cache_engine" in S and "enforce_eager" in S

def test_uses_merged_9b_and_no_cap():
    assert "merged_ep3" in S
    assert "TURING_JUDGE_SCORE_CLIP_MAX=7" in S or "cap" in S.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_rl_9b_launcher.py -q`
Expected: FAIL — `FileNotFoundError: scripts/slurm/rl_generator_train_9b.sh`.

- [ ] **Step 3: Write the 9B trainer launcher**

Adapt `rl_generator_train.sh` (copy it) with these concrete changes — start from veRL's 27B FSDP2
GRPO example, keep the reward-env inheritance identical:

```bash
# scripts/slurm/rl_generator_train_9b.sh  (delta from the 8B version)
PY=/home/lancewicki/miniconda3/envs/turing-rl-rl-qwen35/bin/python   # pinned Arm-B env
MERGED_SFT_MODEL_PATH=${MERGED_SFT_MODEL_PATH:-checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3}
# ... same merged-dir existence guards, DATA_BASE, overfit branch as the 8B version ...
OVR=(
  actor_rollout_ref.model.path="$MERGED_SFT_MODEL_PATH"
  actor_rollout_ref.model.lora_adapter_path=null
  actor_rollout_ref.model.lora_rank=64
  actor_rollout_ref.model.lora_alpha=32
  actor_rollout_ref.model.target_modules=[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]
  actor_rollout_ref.model.exclude_modules='.*visual.*'
  actor_rollout_ref.rollout.lora.merge=true
  actor_rollout_ref.model.enable_gradient_checkpointing=true
  actor_rollout_ref.actor.fsdp_config.param_offload=true
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true
  actor_rollout_ref.ref.fsdp_config.param_offload=true
  actor_rollout_ref.rollout.tensor_model_parallel_size=4
  actor_rollout_ref.rollout.free_cache_engine=true
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.enable_prefix_caching=false
  actor_rollout_ref.rollout.gpu_memory_utilization=0.40
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
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

- [ ] **Step 4: Run the launcher test to verify it passes**

Run: `python -m pytest tests/test_rl_9b_launcher.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit the launcher**

```bash
git add scripts/slurm/rl_generator_train_9b.sh scripts/slurm/rl_generator_run_9b.sh tests/test_rl_9b_launcher.py
git commit -m "feat: Arm-B 9B GRPO launcher (LoRA merge=true, attn+MLP excl visual, TP4/offload)"
```

- [ ] **Step 6: Deploy + run the B0 spike (5–10 steps) on the cluster**

Run (Mac): `scripts/sync_to_cluster.sh` then
`ssh … 'cd … && preflight then JUDGE=9b MODE=overfit OVERFIT_EPOCHS=3 RUN_TAG=9b_b0_spike sbatch --export=ALL scripts/slurm/rl_generator_run_9b.sh'`
Expected: job queued; log shows judge endpoint + trainer banner in the Arm-B env.

- [ ] **Step 7: Evaluate the B0 gate (ALL must hold)**

Tail the log + inspect artifacts:
1. Steps complete, no NaN/crash, `reward_dump/*.jsonl` populated, generations terminate
   (char lengths bounded — reuse the length check from the trajectory README).
2. **Weight parity:** `weight_parity.json` shows `ok: true` (rollout weights changed after an
   optimizer step AND match the actor — via `scripts/rollout_weight_parity.py` fingerprints logged
   by the trainer at step N vs N+1). If the trainer does not yet emit fingerprints, add a small
   hook logging `param_fingerprint(...)` for the actor and the vLLM-reported weights around
   `update_weights`, and re-run.
3. Reward moves in the expected direction over the 3 epochs.
Record the verdict in `.sdd-progress.md`.

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
git add scripts/slurm/submit_arm_b_grid.sh tests/test_submit_arm_b_grid.py results/2026-07-24-reward-hack-proper-checkpoint/README.txt
git commit -m "results: Arm-B 9B overfit grid + 8B-vs-9B H2 comparison"
```

---

## Final: PR

- [ ] Run the full local test suite: `python -m pytest tests/test_rl_grid.py tests/test_proper_checkpoint.py tests/test_submit_arm_a_grid.py tests/test_rollout_weight_parity.py tests/test_rl_9b_launcher.py tests/test_submit_arm_b_grid.py -q` → all PASS.
- [ ] Push the branch and open a **draft** PR (`gh pr create --draft`) with a summary linking the
  spec + this plan. Never push to `main`/`lancewicki/main` directly; additive commits only.

---

## Notes for the implementer

- **Cluster reads:** use SSH `cat`/`ls`, not the Read tool. Tunnel: `ssh -p 2223 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null lancewicki@localhost "<cmd>"`. If refused, ask the user to refresh the tunnel.
- **Arm A is almost pure config** — the 2026-07-15 machinery is unchanged; the only substantive
  cluster op is the checkpoint-78 merge (Task 4). If a run misbehaves, the decisions doc
  `2026-07-15-rl-generator-vs-fixed-judge/decisions.md` documents every prior gotcha (PYTHONPATH,
  wandb .env, epoch-end checkpoint quota, KL-ref merge).
- **Arm B is the risk.** Time-box the env pinning (~1–2 days). The B0 weight-parity gate is
  non-negotiable — "no crash" is not enough (pre-#7014 served the base policy silently).
- **veRL override names** (`lora.merge`, `target_modules`, `exclude_modules`, offload flags) must be
  validated against the pinned veRL SHA's config schema during Task 10 Step 3 — adjust the exact
  Hydra keys if the pinned veRL uses different names, keeping the semantics fixed.
