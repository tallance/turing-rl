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
  --dependency-profile training \
  --run-root /home/lancewicki/projects/turing-rl/results/<run> \
  --env MODE=full5 --env JUDGE=9b \
  scripts/submit_snapshot_job.sh \
  --export=ALL -- scripts/slurm/rl_generator_run_9b.sh
```

Launch a multi-job orchestrator the same way:

```bash
scripts/cluster_launch.sh \
  --dependency-profile eval \
  --run-root /home/lancewicki/projects/turing-rl/results/<eval> \
  --env EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/<eval> \
  scripts/launch_test_eval.sh
```

For a disposable experiment whose commit does not contain current main:

```bash
scripts/cluster_launch.sh --debug --label probe-a \
  --dependency-profile all \
  --run-root /home/lancewicki/projects/turing-rl/results/debug/probe-a/run-1 \
  scripts/submit_snapshot_job.sh <sbatch options> -- <script> [script arguments]
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
provenance. The full external inventory is always recorded; only dependencies selected by the
declared profile are compared between submission and job startup:

| Profile | Intended use | Enforced dependencies |
|---|---|---|
| `eval` | generator and multi-model judge evaluation | train, RL-Qwen, SFT-Qwen, judge-vLLM and Gemma environments |
| `training` | GRPO runs | train, RL-Qwen and judge-vLLM environments plus the veRL tree |
| `sft` | supervised training | train and SFT-Qwen environments |
| `data` | dataset preparation | train and upstream veRL environments |
| `all` | unusual mixed/debug workflows | every recorded environment plus the veRL tree |

Pick the profile from the environments a run actually touches, not from its category. A GRPO
run is not automatically `training`: that profile omits the Gemma environment, so a
Gemma-judged GRPO run needs `all`, which is the only profile carrying both the veRL tree and a
non-Qwen judge environment. Enforcement compares only the declared profile, so an
under-declared run still executes -- it simply stops guarding the environment most likely to
drift, which for the Gemma judge is a CUDA-13 nightly build.

Canonical datasets are read through `TURING_RL_INPUT_DATA_ROOT`. Dataset artifacts produced by a
workflow use `TURING_RL_GENERATED_DATA_ROOT`, which points to shared state for retained runs and
the run root for debug runs. `TURING_RL_DATA_ROOT` remains only as a temporary generated-data alias.

## Reclaiming disk from derived model artifacts

Each generator step of a test-set eval leaves ~37 GB under
`results/<eval-run>/models/step<N>/`: `hf_base/` (~18 GB, the reconstructed SFT backbone, the same
content at every step), `hf_base/lora_adapter/` (~223 MB, the only step-unique bytes) and
`hf_dense/` (~19 GB, what is actually served). Both large directories are reproducible from
`merged_ep3` plus the adapter, so they are the first thing to delete when the per-user quota bites
-- but `rm -rf hf_base` also removes the adapter nested inside it, which makes every sibling
`hf_dense` unrecoverable too, silently and after the fact.

`scripts/safe_delete_derived.py` deletes them only after re-deriving the proof from the bytes on
disk. Per target it checks the path guard (must be a directory named `hf_base`/`hf_dense` under a
`results/` root), the adapter (parses, `r`/`alpha` sane, the expected paired tensor count), the base
container, and then spot-checks the reconstruction arithmetic on a seeded random sample of target
modules read lazily out of the shards: `hf_dense[k] == base[k] + (alpha/r) * B@A`, and `hf_base[k]`
bit-identical to the container. Any failure skips that target loudly and the process exits nonzero.
Before deleting an `hf_base` it copies everything that is not a weight shard -- the adapter,
`merge_provenance.json`, config and tokenizer -- into `hf_base.preserved/`, verifies the copy by
SHA-256, and rewrites the other manifests in the batch to cite the surviving adapter path.

```bash
# prove only; nothing is touched (dry run is the default)
python scripts/safe_delete_derived.py results/<eval-run>/models/step*/hf_dense

# prove, then delete
python scripts/safe_delete_derived.py --delete \
  results/<eval-run>/models/step*/hf_base results/<eval-run>/models/step*/hf_dense
```

Each deletion leaves `<target>.deleted.json` beside where the directory was, recording the file
inventory, the adapter (path and SHA-256), the base container, the modules that were verified, and
the literal `merge_grpo_adapter.py` / `validate_grpo_merge.py` commands that rebuild it. Useful
flags: `--sample N` (default 3) to widen the spot-check, `--base` when
`grpo_merge_report.json` records a container under a `work/launcher-*/` directory that has since
been cleaned up and cannot be re-anchored automatically, `--allowed-root` to permit targets outside
the current directory, and `--json` for a machine-readable run report. It is pure safetensors
math on the CPU, so it runs fine on the login node; only the hashing of the preserved adapter is
appreciable work.

## Constraints a job script must respect

Two properties of the runtime view are load-bearing and fail in ways that do not name their
cause. Both were found the hard way; see the git history of the scripts named below.

**One runtime view per job.** `turing_rl_prepare_runtime` derives its work directory from
`SLURM_JOB_ID` and aborts with `FATAL: runtime work directory already exists` if that directory
is present. It is therefore single-use: a script that sources `cluster_job_bootstrap.sh` must
not invoke another script that also sources it. `rl_generator_run_9b.sh` prepares the view and
then `srun`s both a judge and a trainer, so each child guards its bootstrap on
`[ -z "${TURING_RL_WORK_ROOT:-}" ]` and reuses the parent's view; the parent exports that
variable, and children inherit it. Standalone invocation still prepares a view, because nothing
exported it. Roughly forty scripts source the bootstrap, so any new parent/child pairing needs
the same guard until the helper itself becomes idempotent.

**Secrets are not reachable from the source view.** `.env` lives in the state root and is
deliberately absent from the snapshot, but `shared/load_env.py` resolves it as
`Path(__file__).resolve().parents[1]/.env` -- and `.resolve()` follows the runtime view's
symlinks back into the snapshot, where it does not exist. `~/.env` does not exist on this
cluster either, so both candidates miss. `get_openai_api_key` compounds this by calling the
file loader *before* inspecting the process environment, so exporting `OPENAI_API_KEY` does not
help; only a file satisfies it. Jobs that reach a judge must therefore export
`ENV_FILE=$REPO/.env`, the runtime view's symlink to the state-root file. Ray workers inherit
environment variables, so setting it in the driver reaches the reward workers that raise.
Without it a run starts normally and dies at its first reward call, minutes in.

The Slurm gateway syntax is deliberately unambiguous: options precede a mandatory `--`; the next
token is the immutable script and remaining tokens are script arguments.

`SOURCE_MANIFEST.json` describes a source snapshot; `provenance/launch.json`,
`provenance/expected_runtime.json`, and `provenance/jobs/<job-id>/` describe a particular run.
There is deliberately no global mutable source stamp or current-source pointer.

Snapshots are retained indefinitely. The legacy mixed checkout is converted to state-only only
after all jobs executing from it finish, using `scripts/retire_cluster_checkout.py` first without
and then with `--execute` after reviewing its inventory.
