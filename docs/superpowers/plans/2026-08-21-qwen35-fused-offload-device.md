# Qwen3.5 Fused Offload Device Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make veRL's Qwen3.5 fused PPO path compatible with FSDP2 CPU parameter offload so the long-sequence 9B memory experiment can reach an actor update.

**Architecture:** Extend the existing repository runtime patcher with a version-sensitive source transformation for `verl.models.transformers.qwen3_5`. Install it before model construction in both the driver and Ray workers. Verify the transformation and differentiability locally, then run one immutable debug smoke.

**Tech Stack:** Python, pytest, PyTorch autograd, veRL 0.9, FSDP2, Hydra, Slurm.

## Global Constraints

- Work only in the existing dedicated `worktree-judge-only-rlvr-spec` worktree.
- Do not modify the shared cluster veRL checkout directly.
- Run retained/debug jobs only through `scripts/cluster_launch.sh` after `preflight-job-check`.
- Keep experiment README files provenance-only.

---

### Task 1: Runtime source transformation

**Files:**
- Modify: `training/grpo/verl_runtime_patch.py`
- Test: `tests/test_verl_09_compat.py`

**Interfaces:**
- Produces: `_insert_qwen35_fused_vocab_weight_device_transfer(source: str) -> str`
- Produces: `_patch_qwen35_fused_vocab_weight_device_source() -> bool`

- [ ] Add a failing unit test using a minimal Qwen3.5 source fixture containing both `full_tensor()` call sites; assert both become `full_tensor().to(hidden_states.device)`.
- [ ] Run the focused test and confirm it fails because the transformer is absent.
- [ ] Implement the minimal idempotent source transformer and installed-module patch function.
- [ ] Call the installed-module patch from `apply_verl_runtime_patch()` before model construction.
- [ ] Add tests for idempotence and rejection of unknown source.
- [ ] Run `pytest -q tests/test_verl_09_compat.py` and confirm it passes.

### Task 2: Gradient-preservation check

**Files:**
- Test: `tests/test_verl_09_compat.py`

**Interfaces:**
- Consumes: PyTorch's differentiable `.to(device)` operation.

- [ ] Add a tensor-level test that transfers a leaf weight, computes a matrix product and backward pass, and asserts the original weight receives the expected gradient.
- [ ] Run the focused test and the complete compatibility test file.

### Task 3: Cluster smoke

**Files:**
- No code changes.
- Artifacts: `results/debug/qwen35-fused-offload-device-9b-8192/` on the cluster.

**Interfaces:**
- Consumes: committed runtime patch and the existing `MODE=valsmoke` launcher.

- [ ] Commit the tested repository changes.
- [ ] Run the full Slurm preflight checklist with the fresh debug root.
- [ ] Submit exactly one 9B smoke with thinking ON, `data.max_response_length=8192`, `actor_rollout_ref.model.use_fused_kernels=True`, and the longest-prompt subset.
- [ ] Verify the effective log contains `max_new_tokens: 8192`, thinking enabled, fused backend enabled, and the patched-source marker.
- [ ] Inspect whether step 1 completes, recording peak actor memory or the exact failure site.
