# Held-out test-set eval (runbook)

Score generator checkpoints on the 880-prompt held-out set (128 users, disjoint from SFT and
GRPO train/val) with a frozen judge, and get a comparable table across checkpoints.

Everything runs on the cluster. Deploy first: `scripts/sync_to_cluster.sh`.

## The one thing that will bite you

**veRL GRPO checkpoints are not dense models.** They persist an *unmerged* LoRA — `lora_A`/`lora_B`
next to frozen `base_layer` weights. `verl.model_merger` pops the LoRA into `lora_adapter/` and
saves the stripped remainder, which is **the SFT base**. Point a generator at that and every
checkpoint scores like step 0: plausible numbers, no error, wrong conclusion.

So Stage 0 is mandatory, and it ends in a hard gate. Do not skip it, and do not generate from a
model whose merge job did not exit 0.

## Run it

Stage 0 — build a dense model per checkpoint and gate it (CPU, ~3 min each):

```bash
sbatch --export=ALL,STEP=8  scripts/slurm/merge_grpo_ckpt.sh
sbatch --export=ALL,STEP=16 scripts/slurm/merge_grpo_ckpt.sh
```

Optional `DISTINCT_FROM=<other hf_dense>` adds a check that two checkpoints actually differ.
`RUN_TAG` selects the GRPO run (default `9b_half_kl1e4_lr1e4_temp1`). A gate failure exits 5 —
`hf_dense` still exists on disk, so **gate on the job's exit status, not on the directory**.

Stage 1+2 — generate, build pairs, judge (`STEPS` must name checkpoints Stage 0 built; `0` means
the pre-RL SFT init and needs no merge):

```bash
DRY=1 STEPS="0 8 16" bash scripts/launch_test_eval.sh   # print the plan
STEPS="0 8 16" bash scripts/launch_test_eval.sh
```

Submits 1-GPU generations in parallel, CPU pair-builds, then 8-GPU judge cells serialized.

Stage 3 — verify, then summarize:

```bash
python scripts/verify_judge_completeness.py --eval_root results/<EVAL_ROOT> --max_missing_frac 0.03
python scripts/summarize_test_eval.py       --eval_root results/<EVAL_ROOT> \
    --out_csv results/<EVAL_ROOT>/summary.csv --out_md results/<EVAL_ROOT>/summary.md
```

Verify **before** you believe any table. Judge cells swallow per-pair errors into a counter and
still exit 0, so a cell can score 850/880 and look successful to Slurm.

## Adding another judge later

Generation is decoupled from judging, so re-run only the judge stage over the existing pairs:

```bash
DO_GEN=0 JUDGES="qwen35-27b qwen35-397b" STEPS="0 8 16" bash scripts/launch_test_eval.sh
```

## If pairs are missing

Timeouts drop ~1-3% of pairs. Write the gaps and re-judge just those, then re-verify with
`--allow_multi_job` (the dump dir will legitimately span two job ids; the unique-key check still
enforces exactly-once scoring):

```bash
python scripts/verify_judge_completeness.py --eval_root results/<EVAL_ROOT> \
    --write_missing results/<EVAL_ROOT>/raw/pairs_missing
# then resubmit judge_sweep_cell.sh with PAIRS=<...>_missing.parquet and a raised timeout
```

The verifier tolerates up to 3% so the pipeline keeps moving, but **`summarize_test_eval.py` is
strict by default** — a published table should not rest on incomplete scoring. If you do raise
`--max_missing_frac` there, the threshold applies to the **common subset**, not just to each cell:
gaps are usually disjoint, so the intersection shrinks with their union. Real example from this
run — cells at 861/857/870 (worst 97.4%) intersected to only 831/880 = 94.4%.

Note the summarizer discovers any reward directory that exists, including one a judge is still
writing. The strict default catches that (a partial cell fails the count); if you run with a
tolerance, confirm the judge jobs have actually finished first.

## Keeping numbers comparable

- Generation sampling defaults to the model-card values used by GRPO validation
  (`0.7 / 0.8 / 20`, 1024 max tokens, 12500 prompt truncation). `eval/generate_trained.py`
  otherwise forces the PRISM domain default of 0.6, so a run without the overrides is **not**
  comparable to one with them.
- Judge env matches the training run's judge (`temperature 0.6`, `repetition_penalty 1.1`,
  thinking ON). Changing it invalidates comparison against the val curve.
- Re-judging into a directory that already has output is refused; use `FORCE_REJUDGE=1` only when
  you mean it, since reward dumps accumulate rather than overwrite.

## Where things land

```
results/<EVAL_ROOT>/
  models/step<N>/hf_base/     verl.model_merger output + lora_adapter/ + merge_provenance.json
  models/step<N>/hf_dense/    the servable dense policy (~19G) + grpo_merge_report.json
  raw/generator/<key>/        heldout_inference.pkl + gen_metadata.json (records sampling used)
  raw/pairs/                  gen_<key>_880.parquet + .meta.json
  raw/<key>/sweep/<cell>/<mode>/reward/   judge reward dumps (the per-pair records)
  summary.csv / summary.md
```

`hf_base` weights are redundant with the SFT container once validated — only `lora_adapter/` and
`merge_provenance.json` are worth keeping if disk is tight.

Add a provenance-only `README.txt` when pulling results locally (commands, job ids, checksums,
validation status; no interpretation) — see the project `CLAUDE.md`.
