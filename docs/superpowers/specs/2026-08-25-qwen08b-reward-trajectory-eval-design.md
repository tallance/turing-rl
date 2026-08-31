# Qwen 0.8B Reward Trajectory Evaluation Design

## Goal

Evaluate the completed 9B generator run trained on 10% of the data for 20 epochs with the Qwen3.5-0.8B reward judge, using the same frozen 440-pair held-out subset and generation protocol as the existing 10%-dataset trajectory evaluations.

## Evaluation matrix

- Generator run: `9b_frac10_20ep_qwen08b_nothink_kl1e4_lr1e4_temp1`.
- Checkpoints: steps `0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120` (two-epoch spacing).
- Judges, in strict model-major order:
  1. `qwen35-0.8b`, thinking OFF, matching the training reward configuration.
  2. `gemma4-12b`, thinking ON.
  3. `gemma4-31b`, thinking ON.
  4. `qwen35-9b`, thinking ON.
- Pair set: the frozen seed-42 440-row subset used by the existing frac10 evaluations.
- Generation sampling and judge schema remain unchanged from the corrected full-schema pipeline.

## Implementation

Add Qwen3.5-0.8B as an opt-in judge cell. Generalize the existing model-major judge launcher to accept one thinking mode per judge while retaining thinking ON as the default. Add a dedicated thin launcher that pins the run, steps, judge ordering, modes, result naming, and verified step-0 reuse.

Step 0 reuses the existing pair parquet and the already-completed thinking-ON Gemma/Qwen9B cells only after the existing provenance and key-set checks pass. The 0.8B thinking-OFF step-0 cell is newly evaluated. Existing verified cells are skipped by the dedicated trajectory controller; all other cells run in the requested order through the immutable snapshot submission gateway.

## Validation

- Unit tests cover the 0.8B cell shape, mixed mode mapping, judge ordering, checkpoint selection, and dry-run submission plan.
- Preflight checks validate checkpoint directories, held-out subset identity and row count, cached judge models, shell syntax, environment imports, disk writes, and queue size.
- Completion requires exact 440-key coverage per new cell, with no duplicate or stray keys, before results are packaged.
