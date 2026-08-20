# Gemma-12B-trained frac10 trajectory evaluation

## Goal

Evaluate the partial generator trajectory trained against the Gemma 4 12B
judge on the frozen 440-pair held-out subset. Score it with Gemma 4 12B and
Qwen3.5-9B, without confusing it with the Qwen-9B-trained frac10 trajectory.

## Inputs and scope

- Training run tag:
  `9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1`.
- Available epoch-aligned checkpoints: global steps 6 through 72. The retained
  evaluation samples step 0 and every two epochs: `0 12 24 36 48 60 72`.
- Evaluation data is the existing deterministic 440-row subset
  `test_seed42_n440.parquet` (123 users), whose held-out split guard passed.
- Judge order is model-major: all Gemma 4 12B cells, then all Qwen3.5-9B cells.
- Generation and judging settings remain identical to the finalized frac10
  evaluation: one generation per pair; generation temperature 0.7, top-p 0.8,
  top-k 20 and maximum 1024 tokens; judges use thinking on, temperature 0.6,
  repetition penalty 1.1, the full ordered schema, maximum 8192 completion
  tokens, score clip 7, and a 1800-second timeout.

The result root is distinct and names both the training judge and partial
training extent:

`results/2026-08-20-test-eval-9b-train10pct-through12ep-gemma12b-reward-test50pct-full-schema`

## Step-0 reuse

Both generator trajectories start from the same
`qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3` SFT model.
The finalized 2026-08-12 frac10 evaluation also used the same frozen subset,
generation settings, judge models, thinking mode, and response schema.

Therefore step 0 is copied mechanically from
`2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema` rather than
rerun. The setup copies the step-0 pair parquet and only the Gemma 12B and Qwen
9B score trees. It records source paths, source Slurm job IDs, and SHA-256 tree
hashes in a reuse manifest, then verifies the copied hashes. It refuses a
nonempty destination or any source with incomplete pair coverage.

## Pipeline

1. Reuse and verify step 0.
2. Merge and hard-gate checkpoints `12 24 36 48 60 72`.
3. Generate fresh responses on the frozen subset for those six checkpoints.
4. Build 440 aligned human/generated pairs per checkpoint.
5. Run Gemma 12B over all six new checkpoints.
6. Run Qwen 9B over all six new checkpoints.
7. Require exact 440-key coverage for all 14 cells, including reused step 0.

The existing frac10 evaluation and full-schema judge launchers provide steps
2–6. A small committed wrapper performs step-0 reuse, records provenance, and
starts the existing launcher at its merge phase with the narrowed checkpoint
and judge lists. Jobs remain serialized through `afterok`, so a failed cell
halts the chain.

## Timing and capacity record

The result package records enough timing detail to estimate later evaluations
without conflating queue delay with compute time:

- `pipeline_jobs.csv` records job ID, stage, checkpoint, judge, submit time,
  eligible time, start, end, queue wait, active elapsed time, state, exit code,
  and allocated GPUs.
- Merge, generation, and pair building already run as separate jobs, so their
  active durations come directly from Slurm.
- Judge cells emit structured timestamps at job start, after all model servers
  are ready, and after all scoring clients finish. This separates model startup
  from scoring time. Per-shard scoring durations are retained to expose replica
  imbalance.
- A generated timing summary reports total, median, minimum, and maximum active
  time by stage, plus a topology-based compute estimate: concurrent merge jobs
  contribute their maximum duration, while serialized generation/build/judge
  stages contribute their sums. The observed union of active job intervals is
  recorded separately and is not labeled as a dependency-graph critical path.
  Queue wait is reported separately. Reused step 0 is labeled as zero new
  compute and retains its original source job IDs as provenance.

Historical 440-pair cells give this pre-run active-time estimate:

| Stage | Expected active time |
|---|---:|
| Reuse and verify step 0 | under 1 minute |
| Merge and hard-gate six checkpoints | about 5–7 minutes critical path |
| Generate six checkpoints | about 7–9 minutes each; 45–55 minutes total |
| Build six pair sets | about 10 seconds each; about 1 minute total |
| Gemma 12B judge, six cells | about 28–30 minutes each; 2.8–3.0 hours total |
| Qwen 9B judge, six cells | about 17–20 minutes each; 1.7–2.0 hours total |

Expected serialized active time is approximately 5.5–6 hours. Cluster queue
delay is additional and may dominate when the per-user GPU quota is saturated.

## Verification

- Unit tests cover refusal of stale destinations and mismatched/incomplete
  reused artifacts.
- A dry run must show six merges, six generations/builds, and twelve judge
  cells in the requested order.
- Preflight verifies all six actor checkpoints, the 440-row subset, both model
  environments, shell syntax, storage, and queue pressure.
- After completion, the strict completeness verifier and summarizer generate
  the two CSVs directly from raw outputs; plots read those CSVs without manual
  transcription.
- Each dense model must pass the merge gate and carry the expected source run
  tag, step, actor path, and adapter fingerprint before generation starts.
- Each generation and pair parquet must contain exactly 440 unique keys on the
  frozen subset. Every judge shard must exit successfully, with zero missing,
  extra, or duplicate keys after aggregation.
- The final package preserves phase timings, output hashes, split-guard status,
  source/runtime manifests, and the exact commands required to regenerate the
  summaries and plot.
