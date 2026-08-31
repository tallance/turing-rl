Single-token judge evaluation of the 9B train-10% trajectory through 20 epochs
==============================================================================
Provenance and mechanical validation only. Finalized 2026-08-31.

WHAT THIS IS
------------
The same generator trajectory as
  results/2026-08-19-test-eval-9b-train10pct-20ep-every2ep-test50pct-full-schema
scored again by SINGLE-TOKEN judges: the rubric and the JSON schema are removed
and the judge answers with one A/B token, read from its logprobs. Thinking off.

Nothing was regenerated. The generated turns are the ones the full-schema
evaluation already produced, so the generator side is byte-identical between the
two packages and only the judge protocol differs.

Checkpoints: every 2 epochs (6 GRPO steps per epoch)
  0 12 24 36 48 60 72 84 96 108 120

JUDGES
------
  cell             model                                                    TP  replicas
  qwen35-9b-st     Qwen/Qwen3.5-9B                                           1  8
  gemma4-12b-st    google/gemma-4-12B-it                                     1  8
  judge-9b-ce-st   checkpoints/sft/judge_qwen35_9b_ce_dense                  1  8

judge-9b-ce-st is the LoRA cross-entropy judge trained on the single-token A/B
task (see docs/plans/2026-08-26-single-token-judge.md). The other two are
zero-shot and have a full-schema counterpart in the package named above.

Not run here: qwen35-27b, gemma4-31b, and the 5 odd-epoch checkpoints.

SOURCE CODE
-----------
Run snapshot   0640e4307f38a7621f0d101212097d9687404c2d  (retained)
  /home/lancewicki/projects/turing-rl-sources/0640e4307f38a7621f0d101212097d9687404c2d
Analysis snapshot 508505263ce8818c83e5fef5dcc03169d09fcae5  (retained)
  built the summary tables; adds expected_a_rate / a_rate_excess columns that
  postdate the submission. Scoring itself is entirely from the run snapshot.
Both contain lancewicki/main at c5c859bb3e6d6de8345c0cb99b7eaa01d40331fb.

Launcher   scripts/launch_single_token_trajectory.sh
Cell       scripts/slurm/judge_sweep_cell.sh (JUDGE_PROMPT_STYLE=single_token)
Scorer     eval/single_token_judge.py
Tables     scripts/build_single_token_trajectory.py

DATA
----
Frozen held-out subset (unchanged from the full-schema package):
  /home/lancewicki/projects/turing-rl/data/prism/
    full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
  440 rows / 123 users
  sha256 0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2
  split_guard.json is copied here from the full-schema package; it PASSES.

The 11 pair sets span two source roots, because the 20-epoch run is an extension
of the 10-epoch one. Per-step sources and sha256 are in
provenance/pair_sources.psv:
  steps 0-60    results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
  steps 72-120  results/2026-08-19-test-eval-9b-train10pct-20ep-every2ep-test50pct-full-schema
The launcher resolves each step against both roots and refuses a step that
neither provides, or that both provide with differing content.

CONFIGURATION
-------------
Judge calls: max_completion_tokens=1, logprobs=true, top_logprobs=20,
chat_template_kwargs={"enable_thinking": false}, no sampling override (each
model serves at its own generation_config defaults). The verdict is the argmax
of the renormalized A/B mass, not the sampled token. A call hard-fails when the
A/B mass falls below MIN_AB_MASS=0.01.

Side assignment is the same deterministic hash the full arm uses
(shared/judge_utils._stable_turing_generated_is_b), so a given pair sits in the
same A/B slot in both packages.

RUNTIME
-------
Serving env for qwen35-9b-st and judge-9b-ce-st:
  /home/lancewicki/miniconda3/envs/turing-rl-train
  Python 3.12, torch 2.10.0+cu130, transformers 4.57.6, vLLM 0.18.0+cu130
Serving env for gemma4-12b-st (selected by judge_sweep_cell.sh from the model id):
  /home/lancewicki/miniconda3/envs/turing-rl-gemma4-vllm-nightly
  torch 2.13.0, transformers 5.14.1, vLLM 0.26.1rc1.dev335+gc687c1abb
GPUs: NVIDIA A100-SXM4-40GB. Full inventory in provenance/expected_runtime.json.

JOBS
----
Smoke      19440  COMPLETED 0:0, 20 pairs, qwen35-9b-st at step 0
Trajectory 19441-19473, 33 jobs, 2026-08-31, all COMPLETED with exit code 0:0
  qwen35-9b-st    19441-19451
  gemma4-12b-st   19452-19462
  judge-9b-ce-st  19463-19473
Three independent afterok chains, one per judge, so a failure in one judge
cannot strand the others. Per-job elapsed time and node are in judge_jobs.psv.
Wall clock for the whole matrix: about 40 minutes.

MECHANICAL VALIDATION
---------------------
  33/33 cells present (3 judges x 11 checkpoints)
  every cell has exactly 440 unique pair keys; zero missing, extra or duplicated
  all 33 cells share an IDENTICAL pair-id set
  0 hard failures across all 33 cells
  every cell's recorded pair_source matches the parquet for its own checkpoint
  no checkpoint exceeds the |a_rate_excess| > 0.2 degeneracy threshold
The table builder enforces each of these and refuses to emit a summary
otherwise; the run above passed without overrides.

COLUMNS
-------
summary_<cell>.csv, one row per checkpoint:
  judge_accuracy   fraction of votes where the judge picked the real human
  gen_win_rate     fraction where it picked the generated turn (= 1 - accuracy)
  p_gen_mean       mean probability mass on the generated side,
                   p_gen = p_a if the human is B else 1 - p_a
  a_rate           fraction of votes answering "A"
  expected_a_rate  what an unbiased judge would score on this sample
  a_rate_excess    a_rate - expected_a_rate; |excess| > 0.2 flags position bias
  n_hard_fail      calls with A/B mass below the floor; not votes, and excluded
                   from every denominator above

There is no likert_mean column. A single-token verdict maps to rating 1 or 7, so
a mean likert would be 1 + 6*gen_win_rate rather than an independent measurement.
p_gen_mean is the graded signal in its place.

ARTIFACTS
---------
  test_eval_single_token.png
  summary_qwen35-9b-st.csv, summary_gemma4-12b-st.csv, summary_judge-9b-ce-st.csv
  judge_jobs.psv
  split_guard.json
  plot/plot_single_token_trajectory.py, plot/plotstyle.py
  provenance/ launch manifest, runtime inventory, per-step pair sources, claim
  SHA256SUMS

REPRODUCTION
------------
Submit (33 cells, ~40 min):
  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root <absolute cluster run root> \
    scripts/launch_single_token_trajectory.sh

Add --env DRY=1 for a submission-shape check; DRY writes a claim and stages the
pair sets, so give it a run root of its own.

Build the tables (from the analysis snapshot, against the run root):
  python scripts/build_single_token_trajectory.py \
    --run-root <run root> --out-dir <run root>/tables

Render the figure from this directory:
  python plot/plot_single_token_trajectory.py --eval_root .

Verify this directory:
  shasum -a 256 -c SHA256SUMS
