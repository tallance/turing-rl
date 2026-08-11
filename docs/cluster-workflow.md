# Multi-agent and cluster workflow

## Development and integration

Each agent develops and commits in a separate local Git worktree. One designated integrator at a
time updates `lancewicki/main` while holding the repository lock:

```bash
python scripts/integration_lock.py --owner <session> -- <merge/test command>
```

The handoff to the integrator contains the commit SHA, changed files, verification commands and
known conflicts. Same-file conflicts are reconciled semantically with their originators; neither
side is silently discarded. Rebase, reset, force-push and reverting another agent's work require
explicit user approval.

## Publishing and launching

Cluster source is content-addressed and read-only:

```text
/home/lancewicki/projects/turing-rl-sources/<40-character-sha>
```

The mutable path `/home/lancewicki/projects/turing-rl` contains state only: datasets, results,
checkpoints, logs, WandB state and `.env`. Do not execute code from it.

After running the `preflight-job-check` skill, launch a retained experiment from any clean commit
that contains current `lancewicki/main`:

```bash
scripts/cluster_launch.sh \
  --run-root /home/lancewicki/projects/turing-rl/results/<run> \
  --env MODE=full5 --env JUDGE=9b \
  scripts/submit_snapshot_job.sh -- \
  --export=ALL scripts/slurm/rl_generator_run_9b.sh
```

Launch a multi-job orchestrator the same way:

```bash
scripts/cluster_launch.sh \
  --run-root /home/lancewicki/projects/turing-rl/results/<eval> \
  --env EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/<eval> \
  scripts/launch_test_eval.sh
```

For a disposable experiment whose commit does not contain current main:

```bash
scripts/cluster_launch.sh --debug --label probe-a \
  --run-root /home/lancewicki/projects/turing-rl/results/debug/probe-a/run-1 \
  scripts/submit_snapshot_job.sh -- <sbatch options> <script>
```

Dirty and partial deployments are unsupported. A source bug is repaired with another commit and a
new job; a running snapshot is never hot-patched.

## Isolation and provenance

Publication holds a remote lock, extracts into a temporary sibling, verifies every Git path, mode
and digest, and atomically renames the result. Jobs receive a writable runtime view: source entries
point into the immutable snapshot, state entries point into the state root, and Hydra writes below
the run root.

Every run records the repository SHA and tree, submission arguments, job dependencies, veRL SHA
and dirty diff digest, Conda package-list digests, key package versions, model identifiers, CUDA,
GPU and Slurm context. Secrets come from the state-root `.env` and are never copied into source or
provenance. A runtime fingerprint change between submission and job startup fails the job.

`SOURCE_MANIFEST.json` describes a source snapshot; `provenance/launch.json`,
`provenance/expected_runtime.json`, and `provenance/jobs/<job-id>/` describe a particular run.
There is deliberately no global mutable source stamp or current-source pointer.

Snapshots are retained indefinitely. The legacy mixed checkout is converted to state-only only
after all jobs executing from it finish, using `scripts/retire_cluster_checkout.py` first without
and then with `--execute` after reviewing its inventory.
