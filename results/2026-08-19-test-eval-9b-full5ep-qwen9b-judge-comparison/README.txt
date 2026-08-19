Qwen 9B judge comparison on the full-dataset 5-epoch generator trajectory
============================================================================
Provenance and mechanical validation only. Packaged 2026-08-19.

SCOPE
-----
This package contains two Qwen3.5-9B judge trajectories over the same frozen
generator outputs at GRPO steps 0, 32, 64, 96, 128, 160, 192, 224, 256, 288,
and 320:
  1. zero-shot Qwen3.5-9B, evaluated with thinking ON;
  2. graded/Brier-RL Qwen3.5-9B step 52, trained with thinking OFF and evaluated
     with thinking OFF.

The plotting script accepts an optional future train-ON/eval-ON Brier summary
through --brier-on-summary.

GENERATOR AND DATA
------------------
Generator training run: 9b_full5ep_kl1e4_lr1e4_temp1, job 14217.
Step 0 is the shared pre-RL SFT initialization.

Held-out parquet:
  /home/lancewicki/projects/turing-rl/data/prism/full_s42_history_sft40_grpo60_test10/test.parquet
Rows/users: 880 / 128
Parquet sha256:
  c7b13e2d538630109e100b7f66e78b8fce4cb1b88c793db357dd1b97c8bf8e32
split_guard.json: PASS; zero row and user overlap with SFT train, GRPO train,
and GRPO validation.

Both judge trajectories score the same 11 pair parquets from:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs
Every pair parquet has 880 unique keys. Their hashes are recorded in the source
package README at results/2026-08-10-test-eval-9b-full5ep-full-schema.

ZERO-SHOT QWEN 9B, THINKING ON
------------------------------
Local source package:
  results/2026-08-10-test-eval-9b-full5ep-full-schema
Cluster source root:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
Submission source:
  /home/lancewicki/projects/turing-rl-runs/full-schema-a6b3990
Git commit: a6b3990f702f43b28ba0b233197c2885b01c5fff
Model: Qwen/Qwen3.5-9B
Job IDs and timestamps: judge_jobs.csv, generated from
results/2026-08-10-test-eval-9b-full5ep-full-schema/judge_jobs.csv.

Serving: TP=1, replicas=8, eight A100-SXM4-40GB GPUs. Judging used
thinking=on, temperature=0.6, repetition_penalty=1.1,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
corrected full ordered response schema. This run predates per-job runtime
manifests; package versions were not captured and must not be inferred from the
current environment.

BRIER-RL QWEN 9B, TRAIN OFF / EVAL OFF
--------------------------------------
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-19-test-eval-9b-full5ep-brier-train-off-eval-thinking-off
Immutable source:
  /home/lancewicki/projects/turing-rl-sources/9c591dd5ea190cf73fcb59ac0bd0ece8fa672060
Git commit: 9c591dd5ea190cf73fcb59ac0bd0ece8fa672060
Launcher: scripts/launch_brier_judge_trajectory.sh

Model: Qwen/Qwen3.5-9B, graded/Brier RL arm, global step 52.
Judge training job: 17893.
Actor checkpoint:
  /home/lancewicki/projects/turing-rl/results/2026-08-14-judge-r1-9b-graded/checkpoints/global_step_52/actor
Validated dense model:
  /home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/judge-9b-graded-step52/hf_dense
Dense merge/validation job: 18446.

Step 0 reused the completed thinking-OFF scoring cell recorded in
provenance/baseline_reuse.json from:
  /home/lancewicki/projects/turing-rl/results/2026-08-19-judge-only-rlvr-thinking-off-eval/raw/sweep/judge-9b-graded-step52/off
Job IDs and timestamps: judge_jobs.csv, generated from
provenance/brier_sacct.psv. The exact baseline tree and pair hashes are in
provenance/baseline_reuse.json; the retained launch record is
provenance/brier_launch.json, and the external runtime inventory is
provenance/expected_runtime.json.

Serving: TP=1, replicas=8, concurrency=32, eight A100-SXM4-40GB GPUs.
Judging used thinking=off, the full ordered response schema,
max_completion_tokens=8192, timeout=1800 seconds, score_clip_max=7, and the
model generation_config sampling defaults. Runtime: Python 3.12.13,
torch 2.10.0+cu130, transformers 4.57.6, and vLLM 0.18.0+cu130 in
turing-rl-train; package-list sha256
cee2ea131ed83674a14b5b98be2eb3997c7ee584f4c60ba1f396b2ffc24b27c1.
Jobs used NVIDIA A100-SXM4-40GB GPUs, driver 580.126.09, and CUDA 13.0.

MECHANICAL VALIDATION
---------------------
The source verifier outputs are preserved under provenance/. The combined job
table and validation.txt are generated mechanically by
plot/build_derived_artifacts.py; metric values, job IDs, and validation counts
are not transcribed by hand.

ARTIFACTS
---------
  qwen9b_judge_comparison.png
  summary_qwen35-9b-zero-shot-thinking-on.csv
  summary_qwen35-9b-brier-train-off-eval-off.csv
  judge_jobs.csv
  validation.txt
  split_guard.json
  plot/build_derived_artifacts.py
  plot/plot_qwen9b_judge_comparison.py
  provenance/brier_launch.json
  provenance/baseline_reuse.json
  provenance/expected_runtime.json
  provenance/brier_sacct.psv
  provenance/brier_validation.txt
  provenance/regular_validation.txt
  SHA256SUMS

REPRODUCTION
------------
Regenerate the two source summaries from their retained cluster roots:

  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  REG_SRC=/home/lancewicki/projects/turing-rl-runs/full-schema-a6b3990
  REG_ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
  $PY $REG_SRC/scripts/summarize_test_eval.py --eval_root "$REG_ROOT" \
    --cell qwen35-9b --mode on --expect_pairs 880 --max_missing_frac 0 \
    --out_csv summary_qwen35-9b-zero-shot-thinking-on.csv \
    --out_md /tmp/summary_qwen35-9b-zero-shot-thinking-on.md

  BRIER_SRC=/home/lancewicki/projects/turing-rl-sources/9c591dd5ea190cf73fcb59ac0bd0ece8fa672060
  BRIER_ROOT=/home/lancewicki/projects/turing-rl/results/2026-08-19-test-eval-9b-full5ep-brier-train-off-eval-thinking-off
  $PY $BRIER_SRC/scripts/summarize_test_eval.py --eval_root "$BRIER_ROOT" \
    --cell judge-9b-brier-train-off --mode off --expect_pairs 880 \
    --max_missing_frac 0 \
    --out_csv summary_qwen35-9b-brier-train-off-eval-off.csv \
    --out_md /tmp/summary_qwen35-9b-brier-train-off-eval-off.md
  $PY $BRIER_SRC/scripts/verify_judge_completeness.py \
    --eval_root "$BRIER_ROOT" --expect_pairs 880 --pairs_tag 880 \
    --max_missing_frac 0

From this package directory, regenerate the plot:

  /usr/bin/python3 plot/build_derived_artifacts.py

  /usr/bin/python3 plot/plot_qwen9b_judge_comparison.py \
    --regular-summary summary_qwen35-9b-zero-shot-thinking-on.csv \
    --brier-off-summary summary_qwen35-9b-brier-train-off-eval-off.csv \
    --out qwen9b_judge_comparison.png

After a train-ON/eval-ON summary is available, add:
  --brier-on-summary <summary.csv>

Verify this artifact manifest from the package directory with:
  shasum -a 256 -c SHA256SUMS
