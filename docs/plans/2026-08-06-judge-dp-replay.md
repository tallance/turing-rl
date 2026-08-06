# Judge Data-Parallel Replay Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Verify whether vLLM's official eight-API-server launcher prevents the long-output judge workload from collapsing onto one data-parallel rank.

**Architecture:** Treat the first 512 completed validation calls from GRPO job 14217 as the legacy-launch control. Replay those prompts at concurrency 64 against the same vLLM 0.18 environment, model, sampling settings, and eight DP ranks, changing only the entrypoint to `vllm serve --api-server-count 8`; retain request, server, and GPU-utilization telemetry for comparison.

**Tech Stack:** Python 3.12, `aiohttp`, vLLM 0.18.0, Slurm, eight A100-40GB GPUs.

---

### Task 1: Add a production-fidelity replay client

**Files:**
- Create: `scripts/benchmark_judge_dp_replay.py`
- Create: `tests/test_benchmark_judge_dp_replay.py`

1. Add tests that require selection of the first `N` validation prompts and the exact production payload: Qwen3.5-9B, 8,192 completion tokens, temperature 0.6, repetition penalty 1.1, thinking enabled, and `response_format=json_object`.
2. Run the focused tests and confirm they fail because the replay client does not yet exist.
3. Implement bounded asynchronous replay with incremental per-call JSONL output and an aggregate JSON summary.
4. Record prompt hashes, HTTP status, timing, token usage, finish reason, and response-size metadata without duplicating the large response text.
5. Run the focused tests and compile the script.

### Task 2: Add the isolated Slurm harness

**Files:**
- Create: `scripts/slurm/judge_dp_replay.sh`

1. Request one eight-A100 node and use the existing `turing-rl-train` environment and cached Qwen3.5-9B checkpoint.
2. Launch `vllm serve` with DP=8 and `--api-server-count 8`, retaining all other serving flags from job 14217.
3. Wait for model-verified readiness, record GPU utilization throughout the replay, run 512 validation-derived requests at concurrency 64, and cleanly terminate the server.
4. Validate the shell syntax.

### Task 3: Commit and deploy only additive files

1. Confirm that no unrelated dirty files are staged.
2. Commit the plan, client, tests, and Slurm harness as one additive experiment commit.
3. Deploy only the new runtime files to the cluster so the divergent deployed training tree is not overwritten.
4. Confirm deployed hashes and syntax.

### Task 4: Preflight and run

1. Execute every applicable item in the repository's Slurm preflight checklist, including resource shape, A100 memory, proxy cleanup, direct environment paths, input existence, output uniqueness, walltime, disk, syntax, vLLM logprob settings, and exact command echoing.
2. Touch `/tmp/sbatch_preflight_ok` immediately before submission.
3. Submit one job and monitor startup, request progress, and completion without disturbing job 14217.

### Task 5: Compare and preserve artifacts

1. Pull the replay's request summary, server log, GPU telemetry, and Slurm log into `results/2026-08-06-judge-dp-replay/`.
2. Compute the same per-rank load concentration and effective-engine metrics used for the legacy validation log.
3. Add a provenance-only `README.txt` with configuration, versions, job/date/path metadata, artifact checksums, mechanical validation, and reproduction commands.
4. Report whether the new launcher materially improves balance and wall time; keep interpretation out of `README.txt`.
