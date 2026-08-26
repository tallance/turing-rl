Full evaluation of the Gemma-12B-reward 9B generator, train-10% for 20 epochs
================================================================================
Provenance and mechanical validation only. Finalized 2026-08-21.

RUN
---
Cluster evaluation root:
  /home/lancewicki/projects/turing-rl/results/
    2026-08-20-test-eval-9b-train10pct-through12ep-gemma12b-reward-test50pct-full-schema
Immutable evaluation source:
  /home/lancewicki/projects/turing-rl-sources/924345eec142e759b25c4ace23f4d3b44aed6157
Git commit: 924345eec142e759b25c4ace23f4d3b44aed6157
Dependency profile: eval

The evaluation was submitted in three append-only segments, whose launch and
runtime manifests are preserved under provenance/:
  steps 0-72:  launched 2026-08-20T18:35:58Z
  steps 84-108: launched 2026-08-21T15:55:55Z
  step 120: launched 2026-08-21T19:49:06Z

TRAINING CHECKPOINTS
--------------------
Run tag:
  9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1
Checkpoint root:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/
    9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1/checkpoints
Evaluated steps:
  0 12 24 36 48 60 72 84 96 108 120

Initial training job 18669 ran through step 72 and then failed. Resume job
18771 completed through step 120. Their source commits were respectively
dd2e628e68c17a88a57dad6611c82e921a0ff6c5 and
924345eec142e759b25c4ace23f4d3b44aed6157. Exact launch, submission, and
runtime manifests are under provenance/. Step 0 is the shared pre-RL SFT
initialization. Every nonzero checkpoint was merged and hard-gated before
generation; provenance/merge_provenance.csv records its actor path, shard
count, and adapter hash.

DATA AND CONFIGURATION
----------------------
Frozen held-out subset:
  /home/lancewicki/projects/turing-rl/data/prism/
    full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
Rows/users: 440 / 123
Parquet sha256:
  0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2
split_guard.json: PASS.

Generation used vLLM TP=1, temperature=0.7, top_p=0.8, top_k=20,
max_tokens=1024, prompt truncation=12500, and max model length=13524.
provenance/pair_sha256.csv records every generated pair parquet.

Judges, in execution order:
  gemma4-12b  google/gemma-4-12B-it TP=1, replicas=8, concurrency=4
  qwen35-9b   Qwen/Qwen3.5-9B       TP=1, replicas=8, concurrency=32
Both used thinking=on, temperature=0.6, repetition_penalty=1.1,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
corrected full ordered response schema.

Step 0 was checksum-verified and reused from:
  /home/lancewicki/projects/turing-rl/results/
    2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
Generation job: 15805. Judge jobs: Gemma 12B=15974, Qwen 9B=15880.
Pair sha256:
  8dc1f53a9fd9c14df88945b5808c4a1001b9174622d99124bb010840909de040

RUNTIME
-------
The complete package inventories and dependency fingerprints are preserved in
provenance/*expected_runtime.json. Relevant environments were:
  train: Python 3.12.13, torch 2.10.0+cu130, transformers 4.57.6,
    vLLM 0.18.0+cu130
  RL-Qwen merge: Python 3.12.13, torch 2.10.0+cu130,
    transformers 5.4.0, veRL 0.9.0.dev0
  Gemma: Python 3.12.13, torch 2.13.0, transformers 5.14.1,
    vLLM 0.26.1rc1.dev335+gc687c1abb
Judge cells used eight NVIDIA A100-SXM4-40GB GPUs.

JOBS, TIMINGS, AND VALIDATION
-----------------------------
All 80 submitted pipeline jobs completed with exit code 0:0: 10 merges,
10 generation jobs, 10 pair builds, 20 judge jobs, and 30 continuations.
The two step-0 judge cells were reused rather than resubmitted.

Zero-tolerance pair-key validation: PASS. All 22 judge cells have exactly 440
unique pair keys, with zero missing, extra, or duplicated keys. Gemma 12B has
0/4,840 parse failures. Qwen 9B has 24/4,840 (0.50%); its worst checkpoint has
5/440 (1.14%), below the accepted 2% per-cell threshold. Summary metrics
exclude rows without a valid Likert score. judge_jobs.csv records the 20
completed judge jobs and two reused source jobs.

Recorded active-time medians:
  merge: 206 seconds
  generation: 578 seconds
  pair build: 7.5 seconds
  Gemma 12B model startup: 91.952 seconds; scoring: 1571.389 seconds
  Qwen 9B model startup: 103.031 seconds; scoring: 942.604 seconds
The topology-based active estimate is 34,064 seconds. Full per-job queue and
active timings are in pipeline_jobs.csv; aggregate values are in
timing_summary.json.

ARTIFACTS
---------
  test_eval_judges_train10pct_20ep_gemma12b_reward_test50pct_full_schema.png
  summary_gemma4-12b.csv and summary_gemma4-12b.md
  summary_qwen35-9b.csv and summary_qwen35-9b.md
  judge_jobs.csv and pipeline_jobs.csv
  timing_summary.json
  split_guard.json and validation.txt
  plot/build_derived_artifacts.py and plot/plotstyle.py
  provenance/ source validation, Slurm accounting, hashes, launch manifests,
    runtime inventories, and training provenance
  SHA256SUMS

REPRODUCTION
------------
Regenerate the local tables and plot from the preserved source summaries and
accounting:

  /usr/bin/python3 plot/build_derived_artifacts.py

Regenerate the raw summaries and exact-coverage validation on the cluster:

  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  SRC=/home/lancewicki/projects/turing-rl-sources/924345eec142e759b25c4ace23f4d3b44aed6157
  EVAL=/home/lancewicki/projects/turing-rl/results/2026-08-20-test-eval-9b-train10pct-through12ep-gemma12b-reward-test50pct-full-schema
  for CELL in gemma4-12b qwen35-9b; do
    $PY $SRC/scripts/summarize_test_eval.py --eval_root "$EVAL" \
      --cell "$CELL" --mode on --expect_pairs 440 --max_missing_frac 0 \
      --out_csv "summary_${CELL}.csv" --out_md "summary_${CELL}.md"
  done
  $PY $SRC/scripts/verify_judge_completeness.py --eval_root "$EVAL" \
    --expect_pairs 440 --pairs_tag 440 --max_missing_frac 0

Verify this package from its directory with:

  shasum -a 256 -c SHA256SUMS
