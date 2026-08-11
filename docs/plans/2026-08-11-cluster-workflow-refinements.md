# Cluster Workflow Refinements Implementation Plan

**Goal:** Scope runtime-change enforcement to declared dependencies, separate canonical input data from run-generated data, and make the Slurm script boundary unambiguous.

**Architecture:** The run manifest keeps a complete external inventory but derives a second profile-specific fingerprint for startup enforcement. Runtime setup exports separate input and generated-data roots while retaining the old data-root name as a generated-data compatibility alias. The submission gateway accepts only `snapshot_sbatch [options] -- script [args]` and passes the canonical immutable script path to Slurm.

**Tech Stack:** Python 3, Bash, pytest, Git snapshots, Slurm.

---

### Task 1: Add dependency profiles

**Files:**
- Modify: `scripts/cluster_workflow.py`
- Modify: `scripts/record_runtime_manifest.py`
- Modify: `scripts/cluster_launch.py`
- Test: `tests/test_cluster_workflow.py`

1. Add named environment paths and `eval`, `training`, `sft`, `data`, and `all` profiles.
2. Write tests showing that changing an unused inventory entry does not change an enforced fingerprint, while changing a selected entry does.
3. Record the full inventory and profile-specific enforced dependency set in every manifest.
4. Require `--dependency-profile` at launch and compare source SHA, profile, and enforced fingerprint at startup.
5. Run `pytest -q tests/test_cluster_workflow.py` remotely; expect all tests to pass.

### Task 2: Split input and generated data roots

**Files:**
- Modify: `scripts/cluster_runtime.sh`
- Modify: maintained launchers under `scripts/slurm/` and `scripts/launch*.sh`
- Modify: `scripts/verify_prism_split.sh`
- Test: `tests/test_cluster_workflow.py`

1. Export canonical `TURING_RL_INPUT_DATA_ROOT` and run-class-aware `TURING_RL_GENERATED_DATA_ROOT`.
2. Keep `TURING_RL_DATA_ROOT` temporarily as an alias of the generated root.
3. Route canonical eval/training inputs through the input root and pipeline-produced datasets through the generated root.
4. Make the GRPO trainer data base absolute so it never resolves into the source-only `data/` Python package.
5. Add static and runtime-view tests for the path contract.

### Task 3: Require an explicit Slurm boundary

**Files:**
- Modify: `scripts/snapshot_sbatch.py`
- Modify: `scripts/submit_snapshot_job.sh`
- Modify: all maintained orchestrators that call the gateway
- Test: `tests/test_cluster_workflow.py`

1. Write tests for a missing boundary and for script arguments ending in `.sh`.
2. Parse options before `--`, take the first following token as the script, and preserve remaining tokens as script arguments.
3. Pass the resolved immutable script path to `sbatch`.
4. Migrate every gateway call to `[options] -- script [args]`.

### Task 4: Documentation and verification

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`
- Modify: `docs/cluster-workflow.md`
- Modify: `docs/test-set-eval.md`

1. Update examples with dependency profiles, split data-root semantics, and the explicit Slurm boundary.
2. Run Python compilation, Bash syntax checks, `git diff --check`, focused tests, and the repository suite excluding only the independently reproduced baseline vLLM failure.
3. Commit, publish the clean commit, and repeat the read-only Hydra/runtime-manifest smoke without submitting a Slurm job.
