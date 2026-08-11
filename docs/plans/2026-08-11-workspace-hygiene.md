# Workspace Hygiene Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Preserve presentation WIP on its own branch, keep local scratch state out of Git status, and publish a clean integrated main snapshot.

**Architecture:** A dedicated `worktree-presentation` branch owns the six presentation Markdown/Python files. Repository ignore rules cover `.codex/` and `temp/`, while the temporary assets remain local and unmodified. A small hygiene branch is verified, merged into `lancewicki/main` under the integration lock, and published through the immutable source publisher.

**Tech Stack:** Git worktrees, Git ignore rules, Python compilation, immutable cluster source publication.

---

### Task 1: Preserve presentation WIP

**Files:**
- Create on `worktree-presentation`: `docs/plans/2026-07-28-adversarial-user-simulation-presentation-draft.md`
- Create on `worktree-presentation`: `docs/presentations/adversarial-user-simulation-initial-draft.md`
- Create on `worktree-presentation`: `scripts/make_field_map_figures.py`
- Create on `worktree-presentation`: `scripts/make_presentation_plots.py`
- Create on `worktree-presentation`: `scripts/plot_differences_rating_accuracy_reordered.py`
- Create on `worktree-presentation`: `scripts/plot_scores_rating_accuracy_reordered.py`

1. Create the presentation worktree from integrated main.
2. Move the six exact untracked files into it without changing their bytes.
3. Compile the four Python scripts and run `git diff --check`.
4. Commit the WIP checkpoint on the presentation branch.

### Task 2: Ignore local-only state

**Files:**
- Modify: `.gitignore`

1. Add `.codex/` beside the existing Claude local-state rule.
2. Add `temp/` as local scratch; do not move or delete its current files.
3. Verify ignored paths with `git check-ignore -v`.
4. Commit the hygiene change and this plan.

### Task 3: Integrate and publish

1. Merge `codex/workspace-hygiene` into `lancewicki/main` while holding `scripts/integration_lock.py` exclusively.
2. Verify main has no tracked or visible untracked changes.
3. Run `git diff --check`, Python compilation, and focused workflow tests.
4. Publish the clean main commit with `scripts/publish_cluster_source.py --json`.
5. Verify the published manifest SHA and that the source tree has no writable paths. Do not submit Slurm jobs.
