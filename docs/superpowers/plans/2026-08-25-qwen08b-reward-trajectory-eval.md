# Qwen 0.8B Reward Trajectory Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit a reproducible 440-pair trajectory evaluation for the 20-epoch generator trained against Qwen3.5-0.8B.

**Architecture:** Extend the shared judge-cell registry and model-major evaluator only where necessary, then configure them through a dedicated thin launcher. Reuse verified step-0 artifacts and serialize the four judge families in the requested order.

**Tech Stack:** Bash, Python 3.12, `unittest`/pytest, Slurm, vLLM, immutable cluster snapshots.

## Global Constraints

- Use a clean commit containing current `lancewicki/main`.
- Launch only through `scripts/cluster_launch.sh` with the `eval` dependency profile.
- Evaluate steps `0 12 24 36 48 60 72 84 96 108 120` on the frozen 440-pair seed-42 held-out subset.
- Judge order and modes are `qwen35-0.8b=off gemma4-12b=on gemma4-31b=on qwen35-9b=on`.
- Preserve the corrected full ordered schema and existing sampling settings.

---

### Task 1: Mixed-mode judge matrix

**Files:**
- Modify: `configs/judge_sweep_cells.py`
- Modify: `scripts/launch_full_schema_eval.sh`
- Test: `tests/test_launch_full_schema_eval.py`

**Interfaces:**
- Consumes: `JUDGES`, `STEPS`, existing `resolve_cell()`.
- Produces: opt-in `qwen35-0.8b` and optional `JUDGE_MODES` aligned one-to-one with `JUDGES`.

- [ ] Add failing tests for 0.8B resolution and mixed per-judge thinking modes.
- [ ] Run the focused tests and confirm the failures describe the missing behavior.
- [ ] Add the 0.8B cell and minimal mode-selection logic, defaulting unspecified modes to ON.
- [ ] Run focused tests and shell syntax checks.
- [ ] Commit the tested change.

### Task 2: Dedicated trajectory launcher

**Files:**
- Create: `scripts/launch_qwen08b_trained_frac10_eval.sh`
- Test: `tests/test_launch_qwen08b_trained_frac10_eval.py`

**Interfaces:**
- Consumes: `scripts/launch_frac10_test50_eval.sh`, verified step-0 reuse, the completed training checkpoint root.
- Produces: a pinned staged evaluation with the requested steps, judges, modes, and result naming.

- [ ] Add a failing dry-run test asserting run tag, steps, judge order, thinking modes, and step-0 reuse.
- [ ] Run it and confirm failure because the launcher is absent.
- [ ] Implement the thin launcher and any narrowly-scoped controller support required to skip verified reused cells.
- [ ] Run focused tests and shell syntax checks.
- [ ] Commit the launcher.

### Task 3: Preflight and submission

**Files:**
- No repository changes unless preflight exposes a defect.

**Interfaces:**
- Consumes: committed launcher and current cluster inputs.
- Produces: immutable snapshot, run manifest, ordered Slurm chain, and job IDs.

- [ ] Verify clean ancestry against current `lancewicki/main`.
- [ ] Run all applicable preflight checks, including real disk write and 440-row/key validation.
- [ ] Dry-run the exact launch command and inspect the complete cell order.
- [ ] Submit through `scripts/cluster_launch.sh --dependency-profile eval`.
- [ ] Confirm the first job starts healthily and report the chain/job IDs.
