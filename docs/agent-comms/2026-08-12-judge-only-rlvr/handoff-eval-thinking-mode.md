# Handoff: `launch_judge_eval.sh` needs a THINKING_MODE parameter

**To:** whoever owns branch `worktree-judge-4b-eval` (the 4B held-out evaluation fork)
**From:** branch `worktree-judge-only-rlvr-spec`
**Date:** 2026-08-18

## Why this is a handoff and not a commit

`scripts/launch_judge_eval.sh` exists only on your branch. Changing it on mine would create a
same-file conflict that CLAUDE.md says must be reconciled with the originating agent rather than
resolved by whoever merges second. The equivalent change is already made on the two launchers I
own, so the pattern below is copy-able.

## The problem

Every judge-GRPO run completed so far was trained with hidden thinking **disabled**, despite the
config asking for it enabled — `data.apply_chat_template_kwargs.enable_thinking: true` was never
read. Details in `docs/judge-thinking-off-ablation.md`.

Consequence for evaluation: those judges are **train-OFF**, and evaluating them thinking-ON is a
cross-mode measurement. Going forward there are two families that must not be mixed:

| family | judges | eval mode |
|---|---|---|
| OFF | the four retained round-1 runs | thinking **OFF** |
| ON | corrected retrains | thinking **ON** |

Within a family, every model — trained and zero-shot alike — must use one pinned inference
policy.

`scripts/launch_judge_eval.sh:87` currently hardcodes:

```
sweep_exports="ALL,MODEL=$dense,TP=$TP,REPLICAS=$REPLICAS,THINKING_MODE=on"
```

## What to change

Mirror what landed in `2e50b6f` on `launch_full_schema_eval.sh` and
`launch_brier_judge_trajectory.sh`:

1. **Accept and validate the mode**

   ```bash
   THINKING_MODE=${THINKING_MODE:-on}
   case "$THINKING_MODE" in on|off) ;; *) echo "FATAL: THINKING_MODE must be on|off, got $THINKING_MODE" >&2; exit 2 ;; esac
   ```
   Defaulting to `on` keeps your existing invocations behaving identically.

2. **Use it in the export**, replacing the literal `THINKING_MODE=on`.

3. **Check any stale-output guard.** `scripts/slurm/judge_sweep_cell.sh:108` writes results to
   `$SWEEP_ROOT/$CELL_NAME/$THINKING_MODE`. A guard that inspects a hardcoded `/on/` path will
   wave through a rerun and let two families overwrite or be mistaken for each other. This bit
   both of my launchers.

4. **Export it if any continuation or dependent job re-invokes the launcher.**
   `launch_full_schema_eval.sh` sets the mode and passes `--export=ALL`; because the variable was
   set but not exported, continuation batches silently reverted to the default after the first
   eight cells, splitting one sweep across two modes. Worth checking whether your script has the
   same shape.

5. **Do not reuse a step-0 cell scored in the other mode.** If your script copies a completed
   baseline cell rather than rescoring it, add the guard I used — the leaf directory is named
   after the mode, so a mismatch is mechanically detectable:

   ```bash
   baseline_mode=$(basename "$BASELINE_CELL_ROOT")
   [ "$baseline_mode" = "$THINKING_MODE" ] || { echo "FATAL: cross-mode baseline" >&2; exit 2; }
   ```

## Also worth knowing

- **There is no 2B zero-shot cell.** The full-schema baseline set covers qwen35-27b (n=313),
  gemma4-31b, gemma4-12b, qwen35-9b, qwen35-4b. If a 2B judge is ever evaluated it has nothing to
  be compared against, so a 2B cell is needed in whichever family it lands in.
- The **9B graded step-52 checkpoint** (val accuracy 0.752, `2026-08-14-judge-r1-9b-graded`) is
  the strongest artifact from round 1 — 0.7404 on the 880-pair set against a 0.631 best zero-shot.
  It is thinking-OFF-trained, so it belongs to the OFF family.
- Checkpoints are FSDP-sharded veRL format. Serving needs `verl.model_merger` →
  `scripts/merge_grpo_adapter.py` → `scripts/validate_grpo_merge.py`; skipping the merge silently
  scores the pre-RL base.

## Reference

- Commit `ef217fa` — thinking fix and dense format reward
- Commit `2e50b6f` — the launcher parameterization to mirror
- `tests/test_eval_thinking_mode.py` — static guards; extend the `LAUNCHERS` tuple with
  `launch_judge_eval.sh` once your change lands and the same assertions will cover it
- `docs/judge-thinking-off-ablation.md` — mechanism, scope, and the round-1 measurements
