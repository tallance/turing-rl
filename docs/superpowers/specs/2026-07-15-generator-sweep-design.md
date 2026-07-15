# Design: generator sweep (fixed judge matrix, varied generator)

**Date:** 2026-07-15
**Builds on:** `docs/superpowers/plans/2026-07-08-judge-sweep-implementation.md` and its
post-plans (esp. `post-plans/2026-07-08-judge-sweep/2026-07-14-cot-failure-diagnostic.md`,
which established the judge "fixes": uniform `temperature=0.6`, `repetition_penalty=1.1`,
`qwen3` reasoning parser, 1800s client timeout).

## Goal

The completed judge-sweep held the **generator fixed** (SFT qwen3-8B) and varied the
**judge size**. This experiment **inverts** that: hold the full judge matrix fixed (with
our fixes) and vary the **generator** that produces the candidate user-turns. It measures
how "human-like" each generator's output is, as scored across every judge size — i.e. how
hard each generator is to tell apart from a real human.

## Experimental matrix (2×2, 4 generators)

| gen key | generator | source | needs |
|---|---|---|---|
| `qwen3-8b-base`   | qwen3-8B, no SFT      | HF base, adapter=None            | generate |
| `qwen3-8b-sft`    | qwen3-8B + SFT        | existing ckpt `checkpoints/sft/qwen3_8b_prism_full_s42_bf16_fsdp_nopack` | **reuse existing pairs** |
| `qwen35-9b-base`  | qwen3.5-9B, no SFT    | HF base, adapter=None            | generate |
| `qwen35-9b-sft`   | qwen3.5-9B + SFT      | **new LoRA — train first**       | train → generate |

The four cells complete a clean **{qwen3-8B, qwen3.5-9B} × {base, SFT}** grid. `qwen3-8b-sft`
is the generator behind the existing 880 pair-set; it is **re-swept with the fixes** (its
original sweep predates them → non-uniform judge temp, no rep_pen) so all four cells use
identical judge settings and are apples-to-apples.

## Fixed judge sweep (per generator)

Reuse the canonical **qwen3.5 family + 397B anchor** cell list from
`configs/judge_sweep_cells.py` (`cell_list("qwen3.5")`): `qwen35-4b`, `qwen35-9b`,
`qwen35-27b`, `qwen35-35b-a3b`, `qwen35-122b`, `qwen35-397b` — **6 cells × {off, on} = 12
scoring jobs per generator**. Each cell serves at its footprint-based shape
(`tp*replicas=8`, i.e. one whole node per cell).

**Fixes applied uniformly to every cell** (this is what makes it "the sweep, with the
fixes"):
- `PERSONA_JUDGE_SAMPLING='{"repetition_penalty":1.1,"temperature":0.6}'` — pinned uniform
  (exported so it reaches `reward.py:_openai_chat`; `judge_sweep_cell.sh` inherits it via
  `--export=ALL`).
- `--reasoning-parser qwen3` on thinking-on servers (already the cell-script default).
- `PERSONA_JUDGE_JSON_SCHEMA=1`, `PERSONA_JUDGE_MAX_COMPLETION_TOKENS=8192`,
  `PERSONA_OPENAI_TIMEOUT_SECONDS=1800` (cell-script / default-params defaults).

Frozen inputs otherwise unchanged: same 880 heldout targets, same rubric/schema, heldout
split only.

## Per-generator pipeline (all reuse existing code; only inputs change)

1. **Generate candidates** — `eval/generate_trained.py` via a 1-GPU vLLM job. Sampling held
   **identical to the existing baseline** so only the model varies: prism domain temp 0.6,
   `presence_penalty=0.5`, `repetition_penalty=1.0`, `gen_num=1`, `--conditioning_mode history`,
   max 2048 tokens, `PROMPT_MODE=reasoning`. SFT generators pass `--checkpoint_dir <ckpt>`
   (adapter auto-resolved). **Base generators need a small additive patch:**
   `generate_trained.py:645` hard-raises when no adapter is found, so add a `--base_model`
   flag that sets `adapter_path=None` (downstream `build_llm_kwargs`/`build_vllm_lora_request`
   already handle `None` → `enable_lora=False`) and guard the adapter-name output block
   against `None`. Documented in `our_patches.md`.
   Output pickle → `results/2026-07-15-generator-sweep/raw/generator/<gen>/heldout_inference.pkl`.
   - `qwen3-8b-sft` skips this step — its pickle/pairs already exist from the judge-sweep.
2. **Build pair-set** — `scripts/build_judge_pairs.py --inference_pkl <pkl> --test_parquet
   <same test.parquet> --out raw/pairs/gen_<gen>_880.parquet`. Same 880 targets, same schema
   the sweep client already loads. Emits `.meta.json` with `exact_match_frac` (guards against
   a generator copying the human turn).
   - `qwen3-8b-sft` reuses the existing
     `results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet`.
3. **Full judge sweep** — for each (cell, mode): submit `scripts/slurm/judge_sweep_cell.sh`
   with `MODEL/TP/REPLICAS/THINKING_MODE` per cell, `PAIRS=raw/pairs/gen_<gen>_880.parquet`,
   `CELL_NAME=gen_<gen>__<cell>`. Dumps land at
   `raw/sweep/gen_<gen>__<cell>/<mode>/{reward,http}` (namespaced by `CELL_NAME` — no
   clobbering, no cell-script change needed).

## Single-node serialized chaining

Every cell needs all 8 GPUs (`tp*replicas=8`), so "one node" ⇒ cells run **sequentially**,
not fanned out like the original 3-node sweep. New wrapper
`scripts/launch_generator_sweep.sh` submits the entire experiment as **one
`--dependency=afterany` chain**, so **at most one node is ever allocated** (2 nodes stay
free for the other agent):

```
[SFT-9B train] → gen(qwen3-8b-base) → build → {12 sweep jobs}
              → gen(qwen35-9b-base) → build → {12 sweep jobs}
              → gen(qwen35-9b-sft)  → build → {12 sweep jobs}
              → {12 sweep jobs for qwen3-8b-sft on existing pairs}
              → analysis
```

- **`afterany`, not `afterok`:** a 397B-on job that hits the 12h wall (known partial-
  completion mode) must not abort the whole chain — partial dumps are still analyzable and
  the sweep is per-pair idempotent on re-run.
- Each `sbatch` requests exactly one node's worth of GPUs (`--gres=gpu:8`). The chain is
  built by capturing each `--parsable` job id and passing it as the next job's dependency.
- The generation + build + SFT steps are 1-GPU / CPU but still submitted as their own
  chained jobs (build can be a tiny CPU job or folded into the front of the next gen job).

## 9B SFT dependency

`qwen35-9b-sft` has no checkpoint and the SFT config is 8B-only. Add:
- `training/sft/configs/qwen35_9b_lora.yaml` — clone `qwen3_8b_lora.yaml`, swap the base to
  `Qwen/Qwen3.5-9B`, keep LoRA **r=64 / α=128 / dropout 0.05, bf16 (no QLoRA)**, 3 epochs,
  same PRISM CoT data (`data/sft/prism_full_s42_sft_cot.jsonl`), `report_to: wandb`.
- Run via existing `scripts/slurm/sft_variant.sh` with `VARIANT=bf16_fsdp NOPACK=1` (the
  verified 40GB recipe) → `checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack`.
  `sft_variant.sh` currently hardcodes the 8B config/output paths per VARIANT; extend it to
  accept the config + output dir (or add a `MODEL=qwen35-9b` branch) — additive only.
- Front of the chain; ~1h, gpu:8, one node.

## New / changed files

**New (ours):**
- `training/sft/configs/qwen35_9b_lora.yaml` — 9B SFT config.
- `scripts/launch_generator_sweep.sh` — the serialized dependency-chain orchestrator.
- `scripts/analyze_generator_sweep.py` — comparison analyzer (see below).
- `tests/test_analyze_generator_sweep.py` — unit tests mirroring
  `tests/test_analyze_judge_sweep.py`.

**Changed (additive):**
- `eval/generate_trained.py` — add `--base_model` flag (no-adapter generation) + guard the
  adapter-name block; log in `our_patches.md`.
- `scripts/slurm/sft_variant.sh` — parameterize base config + output dir (or add a 9B branch).
- (Optional) `configs/judge_sweep_cells.py` `SIZE_MAP` — only if the analyzer needs new keys;
  cell keys are unchanged so likely no edit.

**Reused unchanged:** `eval/generate_trained.py`, `scripts/build_judge_pairs.py`,
`scripts/slurm/judge_sweep_cell.sh`, `scripts/run_judge_sweep_cell.py`,
`configs/judge_sweep_cells.py` (`cell_list`), `scripts/slurm/heldout_inference.sh` (as a
template for the per-generator generation jobs).

## Outputs & analysis

Root: `results/2026-07-15-generator-sweep/`.
- `raw/generator/<gen>/heldout_inference.pkl` (+ metadata) — per-generator candidates.
- `raw/pairs/gen_<gen>_880.parquet` (+ `.meta.json`).
- `raw/sweep/gen_<gen>__<cell>/<mode>/{reward,http}` — per-cell judge dumps.
- `derived/` + `plots/` from `scripts/analyze_generator_sweep.py`.

`analyze_generator_sweep.py` reuses `analyze_judge_sweep.py` internals (per-call features,
`aggregate_cell`, accuracy / penalized-accuracy / parse-error). **Primary output:** curves
of judge-size (x) vs metric (y) with **one line per generator** — the money plot showing
which generator best fools each judge, and whether SFT closes the human-gap more for
qwen3-8B or qwen3.5-9B. Also a 2×2 summary table at the anchor (397B-on).

Write `README.txt` (repro commands, input paths, upstream job ids) per the reports-repro
rule.

## Verification

1. `pytest tests/test_analyze_generator_sweep.py -q` (cluster env; laptop lacks numpy).
2. SFT: confirm `checkpoints/sft/qwen35_9b_...` has a resolvable adapter; loss curve in wandb.
3. Per generator: `build_judge_pairs.py` reports 880 rows, `exact_match_frac < 0.01`
   (generator isn't copying the human turn).
4. Sweep: `raw/sweep/gen_<gen>__<cell>/<mode>/reward/*.jsonl` populated; spot-check a few
   ratings parse; parse-error rate sane (rep_pen fix keeps 397B-on ≈0.03, not 0.11).
5. Only ever one node allocated during the chain (`squeue --me` shows ≤1 running sweep job).
6. Deploy via `scripts/sync_to_cluster.sh` (commit first; Mac is sole author).

## Open items / caveats

- **Compute magnitude.** 4 generators × 12 cells serialized on one node is a **multi-day
  chain** (397B on/off are the long poles). This is the accepted cost of the single-node
  constraint. **Run the full 12 cells for every generator — no cell is dropped unless the
  user explicitly says so.**
- **Base generators + reasoning format.** qwen3.5-9B / qwen3-8B *base* were not trained on
  the `<reasoning>…</reasoning>` SFT format; they may not emit it, so `build_judge_pairs`
  falls back to raw text. That's expected ("before SFT" = worse, less-formatted candidates)
  and is exactly what the experiment measures. `exact_match_frac` and a manual spot-check
  guard against degenerate output.
- **Chain robustness.** ~55 chained jobs (1 SFT + 3 gen + 3 build + 48 sweep;
  `qwen3-8b-sft` skips gen/build and reuses existing pairs) is long; `afterany` keeps it
  moving through partial 397B timeouts. A failed *earlier* step
  (SFT/gen) should hard-stop its generator's branch — build the chain so gen→build→sweep for
  a generator are `afterok` within the branch, but branch-to-branch and cell-to-cell are
  `afterany`.

## Deviations (from the original judge-sweep plan)

- DG1: judge sampling is now an explicit wire override (`repetition_penalty=1.1`,
  `temperature=0.6`) for **all** cells — intended, per the cot-failure fix; supersedes the
  original "no wire override" Task-1 policy.
- DG2: cells run **serialized on one node** (dependency chain) instead of fanned across 3
  nodes — to leave 2 nodes for a concurrent agent.
- DG3: new results root `2026-07-15-generator-sweep/` (the 2026-07-08 tree stays the frozen
  judge-sweep record).
