Trained 9B Brier-judge evaluation of the full-dataset 5-epoch generator run
================================================================================
Provenance and mechanical validation only. Finalized 2026-08-18.

RUN
---
Epoch-boundary cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-17-test-eval-9b-full5ep-brier-judge
Half-epoch cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-17-test-eval-9b-full5ep-brier-judge-halfsteps
Epoch-boundary immutable source:
  /home/lancewicki/projects/turing-rl-sources/24ebd758fea39e4c8a41e1ee84865b0e9f7189bc
Half-epoch immutable source:
  /home/lancewicki/projects/turing-rl-sources/11d946ed35f4f358071085542dcd9bc2550cfbe9
Git commits: 24ebd758fea39e4c8a41e1ee84865b0e9f7189bc and
  11d946ed35f4f358071085542dcd9bc2550cfbe9
Launcher: scripts/launch_brier_judge_trajectory.sh
Evaluation interval: 2026-08-17T19:23:00 through 2026-08-17T22:22:49 UTC.

Generator run: 9b_full5ep_kl1e4_lr1e4_temp1, training job 14217.
Pair checkpoints: steps 0, 32, 64, 96, 128, 160, 192, 224, 256, 288, and 320.
Step 0 is the shared SFT initialization; the other steps are spaced at one-half
GRPO epoch.

TRAINED JUDGE
-------------
Model: Qwen/Qwen3.5-9B, graded/Brier RL arm, global step 52.
Judge training job: 17893.
Actor checkpoint:
  /home/lancewicki/projects/turing-rl/results/2026-08-14-judge-r1-9b-graded/checkpoints/global_step_52/actor
Base snapshot:
  /home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
Validated dense model:
  /home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/judge-9b-graded-step52/hf_dense
Merge/validation job: 18446. The merge report records scaling=0.5 and 128
adapter targets; its captured report is provenance/trained_judge_merge_report.json.

Step 0 reused completed scoring job 18447 from:
  /home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/raw/sweep/judge-9b-graded-step52/on
The reused reward tree and step-0 pair hashes are recorded in
provenance/baseline_reuse.json. New scoring jobs were:
  step 32   18503
  step 64   18454
  step 96   18504
  step 128  18455
  step 160  18505
  step 192  18456
  step 224  18506
  step 256  18457
  step 288  18507
  step 320  18458
All eleven jobs, including reused step 0, are listed in judge_jobs.csv.

DATA AND SCORING CONFIGURATION
------------------------------
Held-out source parquet:
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
Rows/users: 880 / 128
Parquet sha256:
  c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32
split_guard.json: PASS; zero row and user overlap with SFT train, GRPO train,
and GRPO validation.

The eleven pair parquets were copied byte-for-byte from:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs
Their hashes are in provenance/pair_SHA256SUMS.txt.

Every cell used the full ordered response schema, thinking enabled, 8192 maximum
completion tokens, and the judge model's generation_config.json sampling defaults.
Serving used TP=1 with eight replicas, client concurrency 32 per endpoint, and
one eight-A100 job at a time in a strict afterok chain.

RUNTIME
-------
Runtime manifests record Python 3.12.13, torch 2.10.0+cu130,
transformers 4.57.6, and vLLM 0.18.0+cu130 in turing-rl-train.
Jobs used eight NVIDIA A100-SXM4-40GB GPUs, driver 580.126.09, and CUDA 13.0.
The complete environment inventory and package-list fingerprints are retained in
provenance/expected_runtime.json, provenance/jobs/<job_id>/runtime.json, and the
corresponding provenance/halfsteps files.

MECHANICAL VALIDATION
---------------------
All eleven cells contain exactly 880 expected unique pair keys, with zero missing,
extra, or duplicate keys. Valid Likert counts in step order were 877, 879, 875,
880, 876, 879, 878, 879, 877, 878, and 877. See validation.txt and
summary_judge-9b-brier.csv.

LOCAL ARTIFACTS
---------------
  test_eval_judges_full5ep_with_brier.png
  summary_judge-9b-brier.csv and summary_judge-9b-brier.md
  judge_jobs.csv
  validation.txt
  split_guard.json
  base_summaries/summary_<judge>.csv (frozen inputs copied from the finalized
    2026-08-10 five-judge result package)
  plot/plot_brier_judge_trajectory.py
  provenance/ (both launches, runtimes, source summaries and validations,
    model-merge, baseline-reuse, and pair manifests)
  SHA256SUMS

REPRODUCTION
------------
Submit either five-cell group from a clean retained commit. For the epoch-boundary
group, omit STEPS (the default is "64 128 192 256 320"):

  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root /home/lancewicki/projects/turing-rl/results/<date>-test-eval-9b-full5ep-brier-judge \
    --env EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/<date>-test-eval-9b-full5ep-brier-judge \
    --env SOURCE_EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema \
    --env BASELINE_CELL_ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/raw/sweep/judge-9b-graded-step52/on \
    --env MODEL=/home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/judge-9b-graded-step52/hf_dense \
    scripts/launch_brier_judge_trajectory.sh

For the half-epoch group, use a distinct run root and add:

  --env 'STEPS=32 96 160 224 288'

Regenerate each source summary and strict validation on its cluster root, using
its source SHA recorded in provenance/launch.json or
provenance/halfsteps/launch.json:

  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  SRC=/home/lancewicki/projects/turing-rl-sources/24ebd758fea39e4c8a41e1ee84865b0e9f7189bc
  ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-17-test-eval-9b-full5ep-brier-judge
  $PY $SRC/scripts/summarize_test_eval.py --eval_root "$ROOT" \
    --cell judge-9b-brier --mode on --expect_pairs 880 --max_missing_frac 0 \
    --out_csv "$ROOT/summary_judge-9b-brier.csv" \
    --out_md "$ROOT/summary_judge-9b-brier.md"
  $PY $SRC/scripts/verify_judge_completeness.py --eval_root "$ROOT" \
    --expect_pairs 880 --pairs_tag 880 --max_missing_frac 0

Merge the two source CSVs by numeric checkpoint step; their step-0 rows must be
byte-identical. The exact source tables are retained under
provenance/epoch_boundaries/ and provenance/halfsteps/. The committed combined
summary is that stable numeric merge.

Regenerate the local overlay from this directory:

  python plot/plot_brier_judge_trajectory.py \
    --base-eval-root base_summaries \
    --brier-summary summary_judge-9b-brier.csv \
    --out test_eval_judges_full5ep_with_brier.png

Verify the local artifact manifest from this directory with:
  shasum -a 256 -c SHA256SUMS
