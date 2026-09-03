Six-judge evaluation of the single-token-trained 9B GRPO generator
==================================================================
Provenance and mechanical validation only. Finalized 2026-09-03.

WHAT THIS IS
------------
The 9B GRPO generator trained against a SINGLE-TOKEN Qwen3.5-9B judge, evaluated
at every 2 epochs by six judges: four single-token and two full-schema
(thinking ON).

Generator run tag  9b_frac10_20ep_qwen9b_st_kl1e4_lr1e4_temp1
  results/grpo/rl-generator/9b_frac10_20ep_qwen9b_st_kl1e4_lr1e4_temp1
  complete at global_step_120 (latest_checkpointed_iteration.txt = 120)

Unlike the sibling packages, generations did NOT already exist: the run holds
only FSDP-sharded LoRA actor checkpoints, so every evaluated step was merged to
a dense model and generated fresh here.

Checkpoints: every 2 epochs, 11 points
  0 12 24 36 48 60 72 84 96 108 120
Step 0 is the pre-RL SFT init (merged_ep3), not a GRPO checkpoint.

GENERATOR TRAINING JUDGE
------------------------
  Qwen/Qwen3.5-9B, SINGLE-TOKEN protocol, thinking off
Confirmed from the run's own reward dump: judge_prompt_style="single_token",
judge_model="Qwen/Qwen3.5-9B", judge_raw_content="A".

That judge IS plotted here, as cell qwen35-9b-st, marked "*". This differs from
the two sibling packages, whose training judge (full-schema 9B) is not on their
figures and whose "*" therefore carries a caveat.

JUDGES
------
  cell                  model                                      protocol      TP  repl
  qwen35-9b-st          Qwen/Qwen3.5-9B                            single_token   1  8
  gemma4-12b-st         google/gemma-4-12B-it                      single_token   1  8
  judge-9b-ce-st        checkpoints/sft/judge_qwen35_9b_ce_dense   single_token   1  8
  judge-gemma12b-ce-st  checkpoints/sft/judge_gemma4_12b_ce_dense  single_token   1  8
  qwen35-9b             Qwen/Qwen3.5-9B                            full, think ON 1  8
  gemma4-12b            google/gemma-4-12B-it                      full, think ON 1  8

SOURCE CODE
-----------
Snapshot 0fa7d26e4808a2d3f927b429cf29bcf776121a98 (retained)
  /home/lancewicki/projects/turing-rl-sources/0fa7d26e4808a2d3f927b429cf29bcf776121a98
  contains lancewicki/main at a9770a49d9acab8613bdd9a45ed6db633d2d89c6
Used for generation, both judge arms, and both table builders.

Generation  scripts/launch_frac10_test50_eval.sh (PHASE=merge onward)
  merge     scripts/slurm/merge_grpo_ckpt.sh
  generate  scripts/slurm/generator_infer.sh -> scripts/slurm/build_pairs.sh
Single-token judges  scripts/launch_single_token_trajectory.sh
Full-schema judges   scripts/launch_full_schema_eval.sh (wrapper judge phase)
Tables  scripts/build_single_token_trajectory.py, scripts/summarize_test_eval.py

TWO RUN ROOTS
-------------
  generation + full-schema arm
    results/2026-09-02-st-trained-gen-eval-9b-frac10-20ep
  single-token arm
    results/2026-09-02-st-trained-gen-eval-9b-frac10-20ep-stjudges
    manifests copied here as provenance_stjudges/

Both used the SAME source snapshot. The single-token arm has its own root
because record_runtime_manifest.py refused to re-initialize the first root: its
enforced_fingerprint_sha256 had changed between the two submissions. The five
enforced environments were byte-identical in their key packages at that point,
so the change was elsewhere in the fingerprinted inventory ({profile, verl,
environments}; runtime_context is not fingerprinted). The guard was respected
rather than overridden, and the generation and full-schema jobs were unaffected.

DATA
----
Frozen held-out subset, reused verbatim (NOT re-derived):
  data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
  440 rows / 123 users
  sha256 0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2
  split_guard.json: PASS, expect=heldout
This is the same file the sibling package used, so the two trajectories are
scored on identical pairs. PHASE=prepare was skipped deliberately to guarantee
that rather than rely on the sampler being reproducible.

Per-step pair sources and sha256: provenance/pair_sources.psv (single-token arm
staged from the generation root).

GENERATION
----------
Merge, per step: verl.model_merger -> hf_base/ + lora_adapter/, then
merge_grpo_adapter.py -> hf_dense/ (W + 0.5*B@A over 128 targets; the run's
lora_train_meta.json is r=64 / alpha=32, so scaling 0.5), then
validate_grpo_merge.py as a hard gate.

Sampling (unchanged from the sibling packages so the numbers are comparable):
vLLM TP=1, temperature 0.7, top_p 0.8, top_k 20, max_tokens 1024,
prompt truncation 12500, max model length 13524.

Judge sampling for the full-schema arm: repetition_penalty 1.1, temperature 0.6,
max_completion_tokens 8192, score_clip_max 7, timeout 1800 s.
Single-token arm: max_completion_tokens 1, logprobs, top_logprobs 20,
enable_thinking false, MIN_AB_MASS 0.01.

JOBS
----
125 jobs, 2026-09-02 to 2026-09-03, ALL COMPLETED with exit code 0:0:
  10  merge          (st_gen_merge_*)
  11  generation     (st_gen_gen_*)
  11  pair build     (st_gen_build_*)
  27  orchestration  (st_gen_continue)
  11  full-schema qwen35-9b     (st_gen_qwen35-9b_*)
  11  full-schema gemma4-12b    (st_gen_gemma4-12b_*)
  44  single-token, 4 chains    (stj_*)
Per-job elapsed and node are in judge_jobs.psv.
Full-schema cells ran about 18 minutes each and were serialized by the wrapper's
continuation chain; the single-token cells ran about 2.5 minutes each in four
parallel chains.

MECHANICAL VALIDATION
---------------------
  66/66 judge cells present (6 judges x 11 checkpoints)
  every cell has exactly 440 unique pair keys; zero missing, extra or duplicated
  all 66 cells share an IDENTICAL pair-key set, ACROSS BOTH ARMS
  that pair-key set is identical to the sibling package's
  0 hard failures in the single-token arm
  no single-token checkpoint exceeds the |a_rate_excess| > 0.2 threshold
  every single-token cell's pair_source matches its own checkpoint

  MERGE APPLIED: step 48's hf_dense differs from the SFT base in exactly 128
  tensors -- the documented LoRA target count -- with identical keysets across
  775 tensors. This is the check that distinguishes a real trajectory from an
  unmerged-LoRA run, which would silently reproduce step-0 numbers at every
  checkpoint and draw a flat but plausible curve.

  Cross-package anchor: step 0 is the same SFT init as the sibling package.
  Sibling qwen35-9b-st reads 0.459 there, this run reads 0.502. Generations are
  fresh at temperature 0.7, so they are not identical; on n=440 that gap is
  about 1.8 standard errors.

COLUMNS
-------
summary_<st-cell>.csv (single-token arm) carries judge_accuracy, gen_win_rate,
p_gen_mean, a_rate, expected_a_rate, a_rate_excess, n_hard_fail.
summary_<fs-cell>.csv (full-schema arm) is summarize_test_eval.py's schema:
likert_mean, win_rate_ge5, pct_7, judge_accuracy, gen_win_rate, n_tie,
n_parse_error.

gen_win_rate is the only column common to both and is the only quantity plotted
for all six judges. p_gen_mean exists only for single-token; likert_mean only
for full schema.

ARTIFACTS
---------
  test_eval_st_trained_generator.png
  summary_qwen35-9b-st.csv, summary_gemma4-12b-st.csv,
  summary_judge-9b-ce-st.csv, summary_judge-gemma12b-ce-st.csv
  summary_qwen35-9b.csv, summary_gemma4-12b.csv
  judge_jobs.psv
  split_guard.json
  plot/plot_st_trained_generator.py, plot/plotstyle.py
  provenance/ and provenance_stjudges/ launch manifests, runtime inventories,
    per-step pair sources
  SHA256SUMS

REPRODUCTION
------------
Generate (10 merges + 11 generations, ~1 h):
  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root <run root> \
    --env RUN_TAG=9b_frac10_20ep_qwen9b_st_kl1e4_lr1e4_temp1 \
    --env EVAL_PARQUET=<.../eval_subsets/test_seed42_n440.parquet> \
    --env 'STEPS=0 12 24 36 48 60 72 84 96 108 120' \
    --env 'MERGE_STEPS=12 24 36 48 60 72 84 96 108 120' \
    --env GEN_KEY_PREFIX=9b-st-trained-step --env PAIRS_TAG=440 --env EVAL_ROWS=440 \
    --env 'JUDGES=qwen35-9b gemma4-12b' --env PHASE=merge --env OFFSET=0 \
    --env GEN_BATCH_SIZE=4 \
    scripts/launch_frac10_test50_eval.sh

PHASE=merge skips prepare so the frozen subset above is used verbatim. The judge
phase runs automatically after generation and produces the full-schema arm.

Single-token arm (44 cells, ~30 min):
  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root <run root>-stjudges \
    --env SOURCE_ROOTS=<generation run root> \
    --env 'STEPS=0 12 24 36 48 60 72 84 96 108 120' \
    --env GEN_KEY_PREFIX=9b-st-trained-step --env PAIRS_TAG=440 \
    --env 'JUDGES=qwen35-9b-st|Qwen/Qwen3.5-9B;gemma4-12b-st|google/gemma-4-12B-it;judge-9b-ce-st|<...>/judge_qwen35_9b_ce_dense;judge-gemma12b-ce-st|<...>/judge_gemma4_12b_ce_dense' \
    scripts/launch_single_token_trajectory.sh

Tables:
  python scripts/build_single_token_trajectory.py --run-root <stjudges root> \
    --out-dir <stjudges root>/tables --prefix 9b-st-trained-step --pairs 440 \
    --steps "0 12 24 36 48 60 72 84 96 108 120"
  python scripts/summarize_test_eval.py --eval_root <generation root> \
    --cell <qwen35-9b|gemma4-12b> --mode on --expect_pairs 440 \
    --out_csv <generation root>/tables/summary_<cell>.csv

Figure, from this directory:
  python plot/plot_st_trained_generator.py --eval_root .

Verify this directory:
  shasum -a 256 -c SHA256SUMS
