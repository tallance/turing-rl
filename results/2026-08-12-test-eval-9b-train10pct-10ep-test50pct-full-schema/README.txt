Full-schema evaluation of the 9B train-10%-for-10-epochs run on 50% of test
============================================================================
Provenance and operational state only. Last updated 2026-08-12T20:13:48Z.

RUN
---
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
Source snapshot:
  /home/lancewicki/projects/turing-rl-sources/b30ab581a7afbc8e464845fe539f9f210b319c7a
Git commit: b30ab581a7afbc8e464845fe539f9f210b319c7a
Dependency profile: eval
Runtime manifests: <cluster result root>/provenance/

Launcher sha256:
  9ba21e4baf38cbba27669752ca41323eed44047128658b60ab5ab5e6023a18bb  scripts/launch_frac10_test50_eval.sh
  03c067385033425fea836768431189d41166af6029793e0c907d3cce2fae5ed4  scripts/launch_full_schema_eval.sh
  856e060250a5a35775201f75dc93f93f9121b8863217f550966c9d4cbc58e9fa  scripts/sample_eval_parquet.py

TRAINING CHECKPOINTS
--------------------
Run tag: 9b_frac10_10ep_kl1e4_lr1e4_temp1
Completed training job: 15371 (2026-08-11 to 2026-08-12, exit 0).
Training checkpoint root:
  /home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1/checkpoints
Evaluated steps: 0 6 12 18 24 30 36 42 48 54 60
Step 0 is the shared pre-RL SFT initialization. Each nonzero actor checkpoint has
eight FSDP shards and is merged and hard-gated before generation.

EVALUATION DATA
---------------
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
(user_id, post_id, target_idx) key; seed 42; selected rows retain source order.
Split guard: PASS; zero shared users and rows with SFT train, GRPO train, and
GRPO validation. The subset metadata is next to the parquet; the guard report is
<cluster result root>/split_guard.json.

EVAL CONFIGURATION
------------------
Judge order: qwen35-9b, gemma4-12b, gemma4-31b, qwen35-4b, qwen35-27b.
All 11 checkpoints use the same frozen subset and pair-key order.
Generation: vLLM TP=1, temperature=0.7, top_p=0.8, top_k=20,
max_tokens=1024, prompt truncation=12500, max model length=13524.
Judging: thinking on, temperature=0.6, repetition_penalty=1.1,
max completion tokens=8192, timeout=1800s, score clip max=7, corrected full
ordered response schema. Judge cells run model-major and one at a time.

JOB STATE AT LAST UPDATE
------------------------
15782  te_t10t50_prepare     COMPLETED  exit 0
15783  te_t10t50_continue    COMPLETED  exit 0
15784  te_t10t50_merge_6     RUNNING
15785  te_t10t50_merge_12    RUNNING
15786  te_t10t50_merge_18    RUNNING
15787  te_t10t50_merge_24    RUNNING
15788  te_t10t50_continue    PENDING (after all four merges)

The bounded controller submits at most four merge jobs, one 1-GPU generation
plus its CPU pair build, or one 8-GPU judge cell at a time. All dependent stages
use afterok and stop after a failed prerequisite.

REPRODUCTION
------------
From a clean commit containing current lancewicki/main, run preflight-job-check,
then:

  RUN=/home/lancewicki/projects/turing-rl/results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema
  scripts/cluster_launch.sh --plan-only --dependency-profile eval \
    --run-root "$RUN" scripts/launch_frac10_test50_eval.sh
  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root "$RUN" scripts/launch_frac10_test50_eval.sh

The subset path and result root must be absent for a fresh reproduction; the
sampler and judge launcher refuse stale output rather than mixing runs.
