Single-token judge evaluation of the full-dataset 5-epoch 9B GRPO trajectory
============================================================================
Provenance and mechanical validation only. Finalized 2026-08-31.

WHAT THIS IS
------------
The full-dataset 5-epoch generator trajectory scored by SINGLE-TOKEN judges: the
rubric and the JSON schema are removed and the judge answers with one A/B token,
read from its logprobs. Thinking off.

Nothing was regenerated. The generated turns are the ones
  results/2026-08-10-test-eval-9b-full5ep-full-schema
already produced, so the generator side is byte-identical to every other judge
scored against that run and only the judge protocol differs.

This is the same procedure as the sibling package
  results/2026-08-31-single-token-trajectory-9b-train10pct-20ep
applied to a different generator run. It is configuration only: the launcher and
the table builder are unchanged, driven by SOURCE_ROOTS / STEPS /
GEN_KEY_PREFIX / PAIRS_TAG.

Checkpoints: every 32 GRPO steps
  0 32 64 96 128 160 192 224 256 288 320

GENERATOR TRAINING JUDGE
------------------------
The generator (run tag 9b_full5ep_kl1e4_lr1e4_temp1) was trained against
  Qwen/Qwen3.5-9B, FULL schema, thinking ON
from JUDGE=9b in scripts/slurm/rl_generator_run_9b.sh, where JUDGE_MODEL
resolves to Qwen/Qwen3.5-9B and PERSONA_JUDGE_ENABLE_THINKING defaults to 1.

That judge is NOT one of the three plotted here. The qwen35-9b-st line is the
same base model under the single-token protocol, which is a different judge.
The figure marks it "*" and says so in the subtitle.

JUDGES
------
  cell             model                                                    TP  replicas
  qwen35-9b-st     Qwen/Qwen3.5-9B                                           1  8
  gemma4-12b-st    google/gemma-4-12B-it                                     1  8
  judge-9b-ce-st   checkpoints/sft/judge_qwen35_9b_ce_dense                  1  8

  judge-gemma12b-ce-st  checkpoints/sft/judge_gemma4_12b_ce_dense           1  8

judge-9b-ce-st and judge-gemma12b-ce-st are LoRA cross-entropy judges trained on
the single-token A/B task (see docs/plans/2026-08-26-single-token-judge.md). The
other two are zero-shot. Same four judges as the sibling package, so the two
trajectories are directly comparable.


FOURTH JUDGE, ADDED 2026-09-01
------------------------------
judge-gemma12b-ce-st was run after the other three and is recorded separately
because it used a later source snapshot; cluster_launch refuses to mix snapshots
in one run root, so it has its own.

  source snapshot  5c34cce127d8fa2d1aa5eddb46dd01b56f830af2 (retained)
                   contains lancewicki/main at a645d610
  run root         results/2026-09-01-single-token-trajectory-9b-full5ep-gemma12b
                   manifests copied here as provenance_gemma12b/
  jobs             19605-19615, all COMPLETED 0:0

The served model was merged locally from the adapter another agent trained; the
merge command and its verification are recorded in the sibling package
  results/2026-08-31-single-token-trajectory-9b-train10pct-20ep/README.txt
and the same dense model was used here.

Validation for these 11 cells: exactly 880 unique pair keys each, an identical
pair-id set across them, 0 hard failures, no checkpoint over the degeneracy
threshold. Their pair-id set was additionally checked against the other three
judges' and is identical, which the per-root builder cannot see.

SOURCE CODE
-----------
Snapshot 508505263ce8818c83e5fef5dcc03169d09fcae5  (retained)
  /home/lancewicki/projects/turing-rl-sources/508505263ce8818c83e5fef5dcc03169d09fcae5
  contains lancewicki/main at c5c859bb3e6d6de8345c0cb99b7eaa01d40331fb
Used for both scoring and table building. (The sibling package records two
snapshots because its scoring predates the a_rate_excess columns; this run does
not have that split.)

Launcher   scripts/launch_single_token_trajectory.sh
Cell       scripts/slurm/judge_sweep_cell.sh (JUDGE_PROMPT_STYLE=single_token)
Scorer     eval/single_token_judge.py
Tables     scripts/build_single_token_trajectory.py

DATA
----
Frozen held-out set:
  /home/lancewicki/projects/turing-rl/data/prism/
    full_s42_history_sft40_grpo60_test10/test.parquet
  880 rows / 128 users
  sha256 c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32
  split_guard.json is copied here from the full-schema package; it PASSES.

Note this is the 880-pair full test set, NOT the 440-pair 50% subset the sibling
package uses. The two trajectories are not scored on the same pairs.

All 11 pair sets come from a single source root; per-step paths and sha256 are
in provenance/pair_sources.psv:
  results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/
    gen_9b-full5ep-step<N>_880.parquet

CONFIGURATION
-------------
Judge calls: max_completion_tokens=1, logprobs=true, top_logprobs=20,
chat_template_kwargs={"enable_thinking": false}, no sampling override (each
model serves at its own generation_config defaults). The verdict is the argmax
of the renormalized A/B mass, not the sampled token. A call hard-fails when the
A/B mass falls below MIN_AB_MASS=0.01.

Side assignment is the same deterministic hash the full arm uses
(shared/judge_utils._stable_turing_generated_is_b), so a given pair sits in the
same A/B slot here as in the full-schema evaluation of this run.

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
19476-19508, 33 jobs, 2026-08-31, all COMPLETED with exit code 0:0
  qwen35-9b-st    19476-19486
  gemma4-12b-st   19487-19497
  judge-9b-ce-st  19498-19508
Three independent afterok chains, one per judge, so a failure in one judge
cannot strand the others. Per-job elapsed time and node are in judge_jobs.psv.
Wall clock for the whole matrix: about 35 minutes; each cell about 2.5 minutes,
essentially the same as the 440-pair sibling because serving startup dominates.

MECHANICAL VALIDATION
---------------------
  33/33 cells present (3 judges x 11 checkpoints)
  every cell has exactly 880 unique pair keys; zero missing, extra or duplicated
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
Submit (33 cells, ~35 min):
  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root <absolute cluster run root> \
    --env SOURCE_ROOTS=<.../2026-08-10-test-eval-9b-full5ep-full-schema> \
    --env 'STEPS=0 32 64 96 128 160 192 224 256 288 320' \
    --env GEN_KEY_PREFIX=9b-full5ep-step \
    --env PAIRS_TAG=880 \
    scripts/launch_single_token_trajectory.sh

Add --env DRY=1 for a submission-shape check; DRY writes a claim and stages the
pair sets, so give it a run root of its own.

Build the tables:
  python scripts/build_single_token_trajectory.py \
    --run-root <run root> --out-dir <run root>/tables \
    --prefix 9b-full5ep-step --pairs 880 \
    --steps "0 32 64 96 128 160 192 224 256 288 320"

Render the figure from this directory:
  python plot/plot_single_token_trajectory.py --eval_root . \
    --title "9B GRPO generator, full-dataset 5-epoch run - single-token judges" \
    --subtitle "{n} pairs per checkpoint, every 32 GRPO steps, on the SAME generations the full-schema evaluation scores.  Judges answer with one A/B token, no rubric and no schema.
* The generator was trained against Qwen3.5-9B, full schema, thinking ON. That judge is NOT plotted here: the 9B line is the same base model under the single-token protocol."

Verify this directory:
  shasum -a 256 -c SHA256SUMS
