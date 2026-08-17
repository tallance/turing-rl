Full-schema evaluation of the 9B train-10%-for-10-epochs run on 50% of test
============================================================================
Provenance and mechanical validation only. Finalized 2026-08-17.

RUN
---
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
Source snapshot:
  /home/lancewicki/projects/turing-rl-sources/b30ab581a7afbc8e464845fe539f9f210b319c7a
Git commit: b30ab581a7afbc8e464845fe539f9f210b319c7a
Dependency profile: eval
Launch time: 2026-08-12T20:11:59Z
Pipeline interval: 2026-08-12T20:12:08 through 2026-08-15T15:35:42.
Launch record and expected runtime inventory are copied locally under provenance/;
per-job runtime/submission manifests remain under <cluster result root>/provenance/jobs/.

Launcher sha256 values:
  9ba21e4baf38cbba27669752ca41323eed44047128658b60ab5ab5e6023a18bb  scripts/launch_frac10_test50_eval.sh
  03c067385033425fea836768431189d41166af6029793e0c907d3cce2fae5ed4  scripts/launch_full_schema_eval.sh
  856e060250a5a35775201f75dc93f93f9121b8863217f550966c9d4cbc58e9fa  scripts/sample_eval_parquet.py

TRAINING CHECKPOINTS
--------------------
Run tag: 9b_frac10_10ep_kl1e4_lr1e4_temp1
Training job: 15371, COMPLETED with exit code 0, 2026-08-11 to 2026-08-12.
Checkpoint root:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1/checkpoints
Evaluated steps: 0 6 12 18 24 30 36 42 48 54 60.
Step 0 is the shared pre-RL SFT initialization. Each nonzero actor checkpoint
has eight FSDP shards and was merged and hard-gated before generation.

DATA
----
Source held-out parquet (880 rows / 128 users):
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
Source sha256:
  c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32
Frozen subset (440 rows / 123 users):
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/eval_subsets/test_seed42_n440.parquet
Subset sha256:
  0c727d910484aeed0bc59c7f88f3dedbef51bd6b6babe1e3cd053a854e45e1d2
Selected-key-set sha256:
  b6484cf442501a5a0b8f2c45fa5ac48489c9db4c79a74a05f6a3ba4ac8660e45
Selection: lowest seeded SHA-256 priority over each unique
(user_id, post_id, target_idx) key, seed 42; selected rows retain source order.
split_guard.json: PASS; zero row and user overlap with SFT train, GRPO train,
and GRPO validation.

EVALUATION CONFIGURATION
------------------------
Judge order: qwen35-9b, gemma4-12b, gemma4-31b, qwen35-4b, qwen35-27b.
All 11 checkpoints use the same frozen subset and pair-key order.
Generation: vLLM TP=1, temperature=0.7, top_p=0.8, top_k=20,
max_tokens=1024, prompt truncation=12500, max model length=13524.
Judging: thinking=on, temperature=0.6, repetition_penalty=1.1,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
corrected full ordered response schema. Judge cells ran model-major, one at a
time.

Judge model IDs and serving shapes:
  qwen35-9b    Qwen/Qwen3.5-9B       TP=1, replicas=8
  gemma4-12b   google/gemma-4-12B-it TP=1, replicas=8
  gemma4-31b   google/gemma-4-31B-it TP=8, replicas=1
  qwen35-4b    Qwen/Qwen3.5-4B       TP=1, replicas=8
  qwen35-27b   Qwen/Qwen3.5-27B      TP=8, replicas=1
Gemma cache snapshots: 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 (12B) and
842da3794eaa0b77d5f08bae87a17459d91ff475 (31B). Qwen launch IDs were
unpinned aliases; their resolved cache revisions were not recorded.

RUNTIME VERSIONS
----------------
The complete frozen inventory and package-list hashes are in
provenance/expected_runtime.json. Relevant execution environments were:
  train: Python 3.12.13, torch 2.10.0+cu130, transformers 4.57.6,
    vLLM 0.18.0+cu130, package-list sha256
    cee2ea131ed83674a14b5b98be2eb3997c7ee584f4c60ba1f396b2ffc24b27c1
  gemma4: Python 3.12.13, torch 2.13.0, transformers 5.14.1,
    vLLM 0.26.1rc1.dev335+gc687c1abb, package-list sha256
    e095c3652c46cbe5094a2c3e60d7831725fb0a016c28b24435ef006b50ce3287
Per-job manifests record NVIDIA driver 580.126.09, CUDA 13.0, CUDA compiler
12.8.93, and eight NVIDIA A100-SXM4-40GB GPUs for judge cells.

JOBS AND VALIDATION
-------------------
pipeline_jobs.csv records all 157 pipeline jobs; all are COMPLETED with exit
code 0. judge_jobs.csv records the 55 retained judge cells:
  qwen35-9b:  15880,15898,15912,15918,15922,15928,15954,15964,15968,15970,15972
  gemma4-12b: 15974,15981,15983,15985,15989,15992,15994,15998,16000,16003,16005
  gemma4-31b: 16007,16104,16139,16146,16148,16150,16267,16276,16311,16340,17243
  qwen35-4b:  17679,17800,17984,18209,18213,18215,18217,18219,18221,18223,18225
  qwen35-27b: 18227,18259,18281,18304,18306,18308,18310,18314,18317,18319,18321

Zero-tolerance verification: PASS. All 55 cells have exactly 440 expected unique
pair keys, with zero missing, extra, or duplicate keys. The minimum valid Likert
counts across checkpoints were qwen35-9b=436, gemma4-12b=440,
gemma4-31b=440, qwen35-4b=438, and qwen35-27b=437. See validation.txt and
the n_likert column in each summary CSV.

LOCAL ARTIFACTS
---------------
  test_eval_judges_train10pct_test50pct_full_schema.png
  summary_<judge>.csv and summary_<judge>.md (five judges)
  judge_jobs.csv and pipeline_jobs.csv
  split_guard.json and validation.txt
  provenance/launch.json and provenance/expected_runtime.json
  plot/plot_test_eval_judges.py and plot/plotstyle.py
  SHA256SUMS (checksums for every artifact above)

REPRODUCTION
------------
For an exact-code reproduction, use a clean worktree at commit
b30ab581a7afbc8e464845fe539f9f210b319c7a. Because this historical commit no
longer contains current lancewicki/main, the current gateway requires a labelled
debug run root. Run preflight-job-check and then:

  RUN=/home/lancewicki/projects/turing-rl/results/debug/repro-train10pct-test50pct/run-1
  scripts/cluster_launch.sh --debug --label repro-train10pct-test50pct \
    --plan-only --dependency-profile eval \
    --run-root "$RUN" scripts/launch_frac10_test50_eval.sh
  scripts/cluster_launch.sh --debug --label repro-train10pct-test50pct \
    --dependency-profile eval \
    --run-root "$RUN" scripts/launch_frac10_test50_eval.sh

The subset path and result root must be absent for a fresh reproduction; the
sampler and launcher refuse stale output rather than mixing runs.

Regenerate the summaries and zero-tolerance validation from the frozen raw tree:

  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  SRC=/home/lancewicki/projects/turing-rl-sources/b30ab581a7afbc8e464845fe539f9f210b319c7a
  EVAL=/home/lancewicki/projects/turing-rl/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
  for CELL in qwen35-9b gemma4-12b gemma4-31b qwen35-4b qwen35-27b; do
    $PY $SRC/scripts/summarize_test_eval.py --eval_root "$EVAL" \
      --cell "$CELL" --mode on --expect_pairs 440 --max_missing_frac 0 \
      --out_csv "summary_${CELL}.csv" --out_md "summary_${CELL}.md"
  done
  $PY $SRC/scripts/verify_judge_completeness.py --eval_root "$EVAL" \
    --expect_pairs 440 --pairs_tag 440 --max_missing_frac 0

From the repository root, regenerate the local plot:

  D=results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
  python "$D/plot/plot_test_eval_judges.py" --eval_root "$D" \
    --stem test_eval_judges_train10pct_test50pct_full_schema \
    --title "9B GRPO generator, 10%-dataset 10-epoch run, on 50% of the held-out test set" \
    --subtitle "{n} pairs per checkpoint, 123 unseen users. All five judges score the same frozen subset with the corrected full ordered schema.  * = judge the run was trained against. Run tag 9b_frac10_10ep_kl1e4_lr1e4_temp1; step 0 is the shared SFT init."

Verify the local artifact manifest from this directory with:
  shasum -a 256 -c SHA256SUMS
