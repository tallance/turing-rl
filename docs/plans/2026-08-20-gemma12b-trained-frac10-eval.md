# Gemma-12B-Trained Frac10 Evaluation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Launch a separately named, reproducible evaluation of the partial Gemma-12B-rewarded generator trajectory at steps 0, 12, 24, 36, 48, 60, and 72 using Gemma 4 12B and Qwen3.5-9B judges, while preserving stage timings for future estimates.

**Architecture:** A Python helper safely copies and verifies the already-finalized step-0 pair and two judge cells into a fresh result root. A thin shell launcher calls that helper and then delegates steps 12–72 to the existing frac10 merge/generate/build pipeline with narrowed steps and judges. Judge jobs write machine-readable startup/scoring timing JSON, and a standalone summarizer combines captured Slurm accounting with those timing files.

**Tech Stack:** Bash launchers, Python standard library, pandas/Parquet through the existing train environment, Slurm accounting, pytest.

---

### Task 1: Safe step-0 reuse

**Files:**
- Create: `scripts/reuse_test_eval_step.py`
- Test: `tests/test_reuse_test_eval_step.py`

**Steps:**

1. Write tests that build a miniature source result tree and assert the helper copies one pair file plus only the requested judge cells, writes SHA-256 provenance, and preserves byte identity.
2. Add failing tests for a nonempty destination, incomplete pair coverage, mismatched pair keys between judge output and the pair parquet, and missing source metadata/job IDs.
3. Run `python3 -m unittest tests.test_reuse_test_eval_step -v` and confirm the tests fail because the helper is absent.
4. Implement a standard-library CLI that validates the source tree, computes deterministic file/tree hashes, copies through a temporary directory, atomically publishes the step-0 artifacts, and writes `provenance/step0_reuse.json`.
5. Rerun the test module and `python3 -m py_compile scripts/reuse_test_eval_step.py`.
6. Commit the helper and tests.

### Task 2: Machine-readable judge phase timings

**Files:**
- Modify: `scripts/slurm/judge_sweep_cell.sh`
- Test: `tests/test_judge_sweep_timing_markers.py`

**Steps:**

1. Write a source-level test requiring timing fields for job start, all-replicas-ready, clients-finished, model-startup seconds, scoring seconds, and total seconds.
2. Run the test and confirm it fails.
3. Add timestamps around the existing server readiness and client-wait boundaries. On successful completion, atomically write `timing.json` inside the cell mode directory. Do not alter model, sampling, or scoring behavior.
4. Run the timing-marker test, existing judge sweep tests, `bash -n scripts/slurm/judge_sweep_cell.sh`, and `git diff --check`.
5. Commit the instrumentation.

### Task 3: Timing summary generator

**Files:**
- Create: `scripts/summarize_eval_timings.py`
- Test: `tests/test_summarize_eval_timings.py`

**Steps:**

1. Write fixtures containing Slurm pipe-delimited accounting rows and judge `timing.json` files.
2. Test computation of queue wait, active elapsed time, GPU allocation, per-stage totals/minimum/median/maximum, model-startup and scoring durations, the topology-based active estimate, and the observed active-interval union.
3. Run the test and confirm it fails because the summarizer is absent.
4. Implement a CLI producing `pipeline_jobs.csv` and `timing_summary.json` without querying Slurm itself.
5. Run the unit test and compile check.
6. Commit the summarizer.

### Task 4: Dedicated Gemma-trained evaluation launcher

**Files:**
- Create: `scripts/launch_gemma12b_trained_frac10_eval.sh`
- Test: `tests/test_launch_gemma12b_trained_frac10_eval.py`

**Steps:**

1. Write dry-run tests asserting the wrapper pins the run tag, steps `12 24 36 48 60 72`, judges `gemma4-12b qwen35-9b`, frozen 440-row subset, unique generator-key prefix, and distinct result root.
2. Assert it invokes the reuse helper before launching and delegates to `launch_frac10_test50_eval.sh` at `PHASE=merge`.
3. Run the test and confirm it fails because the launcher is absent.
4. Implement the wrapper with no new pipeline logic. It refuses stale output, reuses step 0, then exports the approved settings and calls the existing launcher.
5. Run the new test, existing frac10 launcher tests, shell syntax checks, and `git diff --check`.
6. Commit the launcher.

### Task 5: Integrate, preflight, and submit

**Files:**
- Update only if verification exposes an issue: the files above.

**Steps:**

1. Run all new tests plus the existing frac10/full-schema/judge-sweep tests.
2. Run a complete `DRY=1` launcher test and verify six merges, six generations, six builds, six Gemma cells, and six Qwen cells, in that order after reused step 0.
3. Review the complete diff against the approved design, commit any fixes, and integrate the branch into `lancewicki/main` under `scripts/integration_lock.py`.
4. Run the `preflight-job-check` checklist on the integrated commit: clean descendant, scripts present in the snapshot plan, six actor checkpoints, subset shape/hash, model snapshots/environments, queue pressure, storage, syntax, port isolation, and `afterok` propagation.
5. Launch through `scripts/cluster_launch.sh --dependency-profile eval` into:
   `/home/lancewicki/projects/turing-rl/results/2026-08-20-test-eval-9b-train10pct-through12ep-gemma12b-reward-test50pct-full-schema`.
6. Confirm the step-0 reuse manifest, first merge jobs, and continuation are present and healthy. Report job IDs and the historical 5.5–6 hour active-time estimate separately from queue delay.
