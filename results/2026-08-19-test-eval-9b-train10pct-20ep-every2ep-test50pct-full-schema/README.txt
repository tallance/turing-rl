Full-schema evaluation of the Qwen-judged 9B train-10% trajectory through 20 epochs
===================================================================================
Provenance and mechanical validation only. Finalized 2026-08-21.

RUN
---
This package joins two consecutive segments of the same generator trajectory:

  steps 0-60:
    /home/lancewicki/projects/turing-rl/results/
      2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
    source commit b30ab581a7afbc8e464845fe539f9f210b319c7a

  steps 72-120:
    /home/lancewicki/projects/turing-rl/results/
      2026-08-19-test-eval-9b-train10pct-20ep-every2ep-test50pct-full-schema
    initial source commit dd2e628e68c17a88a57dad6611c82e921a0ff6c5
    resumed source commit 01a517f5073114358624ea41296dab42f03bb725

The corresponding launch and runtime manifests are preserved under provenance/.
The evaluated steps are:
  0 6 12 18 24 30 36 42 48 54 60 72 84 96 108 120

TRAINING CHECKPOINTS
--------------------
Steps 0-60 use the original run:
  run tag 9b_frac10_10ep_kl1e4_lr1e4_temp1
  training job 15371, COMPLETED

Steps 72-120 use its continuation:
  run tag 9b_frac10_20ep_kl1e4_lr1e4_temp1
  successful training job 18571, COMPLETED, 2026-08-18T17:55:07 through
    2026-08-19T15:27:11 UTC
  source commit 35e1266ad0dec1cea8ca5ddd4a6cf0c99f3e73b1

Step 0 is the shared pre-RL SFT initialization. Nonzero steps are validated
dense merges of the corresponding GRPO actor checkpoints.

DATA AND CONFIGURATION
----------------------
Frozen held-out subset:
  /home/lancewicki/projects/turing-rl/data/prism/
    full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
Rows/users: 440 / 123
Parquet sha256:
  0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2
The base and extension split guards are byte-identical and PASS.

Generation used vLLM TP=1, temperature=0.7, top_p=0.8, top_k=20,
max_tokens=1024, prompt truncation=12500, and max model length=13524.

Judge order in the source evaluations:
  qwen35-9b, gemma4-12b, gemma4-31b, qwen35-4b, qwen35-27b
All cells used thinking=on, temperature=0.6, repetition_penalty=1.1,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
corrected full ordered response schema.

Judge model IDs and serving shapes:
  qwen35-9b    Qwen/Qwen3.5-9B       TP=1, replicas=8
  gemma4-12b   google/gemma-4-12B-it TP=1, replicas=8
  gemma4-31b   google/gemma-4-31B-it TP=8, replicas=1
  qwen35-4b    Qwen/Qwen3.5-4B       TP=1, replicas=8
  qwen35-27b   Qwen/Qwen3.5-27B      TP=8, replicas=1

RUNTIME
-------
The complete external inventories are preserved in provenance/. Relevant
evaluation environments were unchanged across the source segments:
  train: Python 3.12.13, torch 2.10.0+cu130, transformers 4.57.6,
    vLLM 0.18.0+cu130
  Gemma: Python 3.12.13, torch 2.13.0, transformers 5.14.1,
    vLLM 0.26.1rc1.dev335+gc687c1abb
Judge cells used NVIDIA A100-SXM4-40GB GPUs.

JOBS AND VALIDATION
-------------------
judge_jobs.csv records the 80 retained judge jobs. All 80 completed with exit
code 0:0. Zero-tolerance coverage verification passed for all five judges at
all 16 checkpoints: every cell has exactly 440 unique pair keys, with zero
missing, extra, or duplicated keys.

Historical extension job 18706 failed before scoring and was replaced by
completed job 18751. Its blocked continuation 18707 was cancelled. Neither is
included in the retained judge table; both remain in the source accounting file
provenance/qwen20ep_extension_sacct.psv.

ARTIFACTS
---------
  test_eval_judges_train10pct_20ep_test50pct_full_schema.png
  summary_<judge>.csv and summary_<judge>.md
  judge_jobs.csv
  split_guard.json
  validation.txt
  plot/build_derived_artifacts.py
  plot/plot_test_eval_judges.py and plot/plotstyle.py
  provenance/ source summaries, validations, launch manifests, runtime
    inventories, training provenance, and extension accounting
  SHA256SUMS

REPRODUCTION
------------
Regenerate every derived table, the combined job table, validation record, and
plot from the preserved source artifacts:

  /usr/bin/python3 plot/build_derived_artifacts.py

The builder rejects unexpected checkpoints, incomplete 440-pair rows, duplicate
or missing retained judge jobs, differing split guards, or failed source
validation records.

Verify the artifact manifest from this directory with:

  shasum -a 256 -c SHA256SUMS
