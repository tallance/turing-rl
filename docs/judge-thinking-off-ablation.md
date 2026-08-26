# Judge-RLVR round 1: the thinking-OFF ablation

The four completed judge-GRPO runs (2× 4B, 2× 9B) were trained with hidden thinking **disabled**,
although their config asked for it enabled, and under a format reward that made partial JSON
near-optimal. They are retained and labelled as an ablation rather than discarded. This note
records what they are, so the numbers are not later mistaken for the corrected configuration.

Provenance belongs in `results/2026-08-12-judge-only-rlvr/README.txt`; this file carries the
mechanism and the measurements.

## Defect 1 — training was thinking-OFF

`training/grpo/configs/qwen35_judge_grpo.yaml` sets
`data.apply_chat_template_kwargs.enable_thinking: true`. Nothing read it:

```
SingleTurnAgentLoop.run  (patched_single_turn_run, verl_runtime_patch.py)
  -> _render_text_prompt_ids
  -> get_chat_template_kwargs_for_prompt_mode      (shared/prompt_utils.py)
  -> prompt_mode_uses_chat_template_thinking()
  -> resolve_chat_template_thinking_override(default=False)   # reads PERSONA_ENABLE_THINKING only
```

`PERSONA_ENABLE_THINKING` is set by no launcher. `scripts/slurm/judge_grpo_train.sh` exports
`PERSONA_JUDGE_ENABLE_THINKING`, a different variable that governs only the served-judge path.
Every rollout therefore rendered with `enable_thinking=False`, ending the prompt in a pre-closed
empty block:

| | prompt tail |
|---|---|
| intended (thinking ON) | `<think>\n` |
| actual | `<think>\n\n</think>\n\n` |

**Scope: judge runs only.** The generator parent config `qwen3_8b_grpo.yaml` sets
`enable_thinking: false`, which matches the buggy default, so generator RL received what it
intended. Only a config asking for `true` was silently overridden. All six GRPO configs inherit
from that parent, so no other run was affected.

Fixed in `ef217fa`: the driver seeds `PERSONA_ENABLE_THINKING` from the resolved config in
`patched_get_ppo_ray_runtime_env` and propagates it to Ray workers unconditionally; unresolvable
config aborts rather than defaulting to False.

## Defect 2 — the format reward paid for partial JSON

`format_score` was the unweighted mean of four booleans, against `0.9·task + 0.1·format`:

| output | task | format | total |
|---|---|---|---|
| correct compact `{"score_gap","rating"}` | 1.0 | 0.50 | **0.950** |
| correct full 37 fields, imperfect arithmetic | 1.0 | 0.75 | **0.975** |

All 37 fields were therefore worth 0.025 of total reward. Both 2B runs found the shortcut: as
`fmt_all_fields` fell to 0.000, the aggregate format reward *rose*.

| 2B run 18452, step | 1 | 12 |
|---|---|---|
| `judge_format_score` | 0.33 | 0.44 |
| `judge_fmt_json_valid` | 0.66 | 0.89 |
| `judge_fmt_all_fields` | 0.05 | 0.00 |
| `judge_rung_score_gap` | 0.59 | 0.88 |

The model was optimising the reward as written. Fixed in `ef217fa`.

## What the retained runs measured

Trained thinking-OFF under the flat-mean format reward (`0.9/0.1`); evaluated thinking-ON with
the full schema. They are **train-OFF / evaluate-ON**.

### 9B, held-out validation (1,410 rows / 705 contexts, steps 0/13/26/39/52)

| metric | graded (job 17888) | directional (job 17893) |
|---|---|---|
| accuracy | 0.461 → 0.519 → 0.623 → 0.727 → **0.752** | 0.443 → 0.468 → 0.498 → 0.500 → **0.500** |
| tie rate | 0.135 → 0.221 → 0.078 → 0.001 → 0.084 | 0.138 → 0.090 → 0.928 → **1.000** → **1.000** |
| confidence | 0.465 → 0.365 → 0.432 → 0.996 → 0.901 | 0.462 → 0.452 → 0.025 → 0.000 → 0.000 |
| pred_B | 0.327 → 0.308 → 0.491 → 0.568 → 0.456 | 0.318 → 0.466 → 0.051 → 0.000 → 0.000 |
| all-37-fields | 0.809 → 0.996 → 1.000 → 0.999 → 0.997 | 0.819 → 0.987 → 0.993 → 0.998 → 1.000 |

The directional arm emits rating 4 on 100% of validation rows from step 39 onward.

### 4B, training-rollout metrics only (first-10 → last-10 of 52 steps)

`trainer.test_freq` defaulted to −1 on these two runs, so no validation ran. Each context appears
8× within the single epoch (416 contexts × 4 generations × 2 orders), so these are not held-out.

| metric | graded (16269) | directional (16244) |
|---|---|---|
| accuracy | 0.490 → 0.652 | 0.482 → 0.552 |
| tie rate | 0.162 → 0.031 | 0.070 → 0.517 |
| confidence | 0.407 → 0.921 | 0.626 → 0.163 |
| pred_B | 0.322 → 0.307 | 0.465 → 0.099 |
| response length | 980 → 630 | 945 → 609 |

### 880-pair held-out evaluation

Trained 9B graded judge scoring `merged_ep3` generations: **0.7404** (n=880, 3 parse errors).
Zero-shot cells from `2026-08-10-test-eval-9b-full5ep-full-schema`, same pair set and schema:

| judge | accuracy | n |
|---|---|---|
| qwen35-27b | 0.631 | 313 (partial) |
| gemma4-31b | 0.593 | 880 |
| gemma4-12b | 0.542 | 880 |
| qwen35-9b | 0.518 | 880 |
| qwen35-4b | 0.501 | 880 |

There is **no 2B zero-shot cell**, so a 2B judge currently has no baseline to be measured against.

## Reading these numbers alongside corrected runs

- These are **train-OFF / evaluate-ON**. Corrected runs will be train-ON / evaluate-ON.
- Evaluation must not mix modes. `judge_sweep_cell.sh` writes to `$CELL_NAME/$THINKING_MODE`, and
  both eval launchers now take `THINKING_MODE`, so an OFF family and an ON family coexist. The
  Brier trajectory refuses a step-0 baseline cell whose mode differs from the sweep's.
- The corrected runs change **two** variables at once (thinking and format reward), so they are
  not a one-variable ablation against these. That was a deliberate choice to avoid paying for two
  retraining campaigns.
- No 2B run completed. Five attempts: two OOMs from configuration deviations, one
  `expandable_segments`/vLLM-sleep-mode startup failure, and two format collapses now attributed
  to defect 2.
