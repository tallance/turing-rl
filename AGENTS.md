# Agent Instructions

Follow the repository instructions in `CLAUDE.md`.

## Multi-Agent and Cluster Safety

- Work in a dedicated local branch/worktree. Do not update `lancewicki/main` unless designated as the integrator for the session.
- Preserve every originator's intent when resolving same-file conflicts; stop and escalate if that intent is uncertain.
- Never use `scripts/sync_to_cluster.sh`, direct `sbatch`, dirty source, the mutable state root, or a cluster Git worktree to run repository code.
- Retained runs require a clean descendant of current `lancewicki/main`; divergent commits are debug-only and use the debug results namespace.
- Before cluster submission, use the `preflight-job-check` skill, then launch through `scripts/cluster_launch.sh`.

## Experiment READMEs

Experiment `README.txt` files contain provenance only: exact configuration and versions, job IDs and dates, cluster source paths, artifact filenames and checksums, mechanical validation status, and reproduction commands.

Do not include results interpretation, scientific conclusions, hypothesis verdicts, or claims about what the results mean. Those are for the user to decide.
