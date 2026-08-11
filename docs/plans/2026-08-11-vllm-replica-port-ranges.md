# vLLM Replica Port Ranges Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent concurrent judge replicas from racing on the same internal vLLM/PyTorch rendezvous port, then resume the stopped full-schema evaluation at Qwen 9B step 224.

**Architecture:** Keep the existing OpenAI API ports unchanged. Give every vLLM replica a deterministic, non-overlapping internal port band through a per-process `VLLM_PORT` value; validate the configured range before launching any server. Deploy the repaired committed tree as a new immutable cluster checkout and restart only the uncompleted model-major batch.

**Tech Stack:** Bash, vLLM OpenAI-compatible servers, PyTorch distributed TCPStore, pytest, Slurm.

---

### Task 1: Add a regression test

**Files:**
- Modify: `tests/test_gemma4_judge_runtime.py`
- Test: `tests/test_gemma4_judge_runtime.py`

**Step 1: Write the failing test**

Add a source-level launcher test requiring configurable internal port base/stride values and requiring both Qwen and Gemma vLLM commands to receive the per-replica `VLLM_PORT` value.

**Step 2: Run the test to verify it fails**

Run: `python -m pytest -q tests/test_gemma4_judge_runtime.py`

Expected: FAIL because the current launcher sets only the public API port.

### Task 2: Assign disjoint internal port bands

**Files:**
- Modify: `scripts/slurm/judge_sweep_cell.sh`
- Test: `tests/test_gemma4_judge_runtime.py`

**Step 1: Add the minimal implementation**

Define `VLLM_INTERNAL_PORT_BASE` and `VLLM_INTERNAL_PORT_STRIDE`, validate that each replica's range remains within TCP port bounds, compute `internal_port` per replica, and prefix both vLLM launch paths with `VLLM_PORT=$internal_port`.

**Step 2: Run focused verification**

Run: `bash -n scripts/slurm/judge_sweep_cell.sh`

Run: `python -m pytest -q tests/test_gemma4_judge_runtime.py tests/test_run_judge_sweep_cell.py tests/test_launch_full_schema_eval.py`

Expected: shell syntax passes and all tests pass.

**Step 3: Commit**

Commit the launcher, regression test, and this plan together.

### Task 3: Deploy and resume safely

**Files:**
- Deploy committed tree to: `/home/lancewicki/projects/turing-rl-runs/full-schema-<sha>`
- Reuse results: `/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema`

**Step 1: Deploy and verify**

Archive the committed integration tree into a new immutable cluster checkout, stamp `DEPLOYED_SHA`, run shell syntax checks, and run the focused pytest suite in the target environment.

**Step 2: Run the Slurm preflight**

Verify the failed step-224 reward directory is empty, all pair inputs exist with 880 rows, the held-out split guard still passes, the queue is bounded, and the new launcher contains disjoint internal port configuration.

**Step 3: Replace only the stalled dependency batch**

Cancel jobs `15281` through `15287`, which can never run after failed job `15280`. Submit `OFFSET=7 BATCH_SIZE=7` from the repaired checkout, keeping the same output root, models, pair sets, sampling parameters, and model-major order.

**Step 4: Verify the resumed cell**

Confirm all eight Qwen 9B replicas reach readiness with distinct internal port starts and that step 224 begins writing reward rows without `EADDRINUSE`.
