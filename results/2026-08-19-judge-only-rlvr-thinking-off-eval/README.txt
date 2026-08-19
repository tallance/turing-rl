Judge-only RLVR round 1: held-out thinking-mode comparison
===========================================================
Provenance and mechanical validation only. Finalized 2026-08-19.

RUN
---
Cluster result root:
  /home/lancewicki/projects/turing-rl/results/2026-08-19-judge-only-rlvr-thinking-off-eval
Immutable source:
  /home/lancewicki/projects/turing-rl-sources/a0599c6aa7d3fee19d904a533a0c5c1870a3acea
Git source/main SHA: a0599c6aa7d3fee19d904a533a0c5c1870a3acea
Launcher: scripts/launch_judge_eval_matrix.sh
Run class/dependency profile: retained / eval
Evaluation interval: 2026-08-19T08:01:28 through 2026-08-19T11:05:17 UTC.

Thinking-ON zero-shot source root:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema
Source commit: a6b3990f702f43b28ba0b233197c2885b01c5fff

Jobs and cells are recorded in job_accounting.txt:
  18637 judge-9b-graded-step52
  18638 judge-4b-graded-step52
  18639 judge-4b-directional-step52
  18640 qwen35-27b
  18641 gemma4-31b
  18642 gemma4-12b
  18643 qwen35-9b
  18644 qwen35-4b
  18645 judge-9b-directional-step52

MODELS
------
Round-1 RL judges were trained for one epoch / 52 steps with hidden thinking
disabled. Dense evaluation models were reused from:
  4B graded/directional:
    /home/lancewicki/projects/turing-rl/results/2026-08-14-judge-4b-eval-v2/models/
  9B graded/directional:
    /home/lancewicki/projects/turing-rl/results/2026-08-17-judge-9b-eval/models/

Training jobs:
  16269 4B graded; 16244 4B directional
  17888 9B graded; 17893 9B directional

Zero-shot model revisions:
  Qwen/Qwen3.5-4B   851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
  Qwen/Qwen3.5-9B   c202236235762e1c871ad0ccb60c8ee5ba337b9a
  Qwen/Qwen3.5-27B  fc05daec18b0a78c049392ed2e771dde82bdf654
  google/gemma-4-12B-it 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
  google/gemma-4-31B-it 842da3794eaa0b77d5f08bae87a17459d91ff475

DATA AND SCORING
----------------
Pair parquet:
  /home/lancewicki/projects/turing-rl/results/2026-08-10-test-eval-9b-full5ep-full-schema/
    raw/pairs/gen_9b-full5ep-step0_880.parquet
Rows/users: 880 / 128
Parquet SHA-256:
  95f48a9c52d85a6f6c49fd3387e60efe0e1ee5e436bd961f1884750ecfcf7783

New cells used THINKING_MODE=off, the full ordered response schema, 8192
maximum completion tokens, and the model generation_config sampling defaults.
The four trained judges and Qwen 4B/9B/Gemma 12B used TP=1 x 8 replicas.
Qwen 27B and Gemma 31B used TP=8 x 1 replica.

Thinking-ON zero-shot rows use the finalized corrected full-schema step-0
summaries. Source jobs: Qwen 9B=15245, Gemma 12B=15365, Gemma 31B=15557,
Qwen 4B=15961, and Qwen 27B=17022. The plot includes only the 4B and 9B
graded/Brier RL judges; the directional/0-1 RL rows are omitted.

For each record:
  rating = rating_gt_first for randomized_order=gt_first, else rating_gen_first
  human_is_b = randomized_order != gt_first
  correct = 0.5 for rating 4, else float((rating > 4) == human_is_b)
Accuracy is mean(correct); se is sqrt(p * (1-p) / scored).

RUNTIME
-------
External-inventory SHA-256:
  00f3b7c6869033fbfbdea5075b9dc9b32f488f3426a049a56e08d3f7d189156d
Qwen/trained-judge client/server environment: Python 3.12.13,
  torch 2.10.0+cu130, transformers 4.57.6, vLLM 0.18.0+cu130.
Gemma server environment: Python 3.12.13, torch 2.13.0,
  transformers 5.14.1, vLLM 0.26.1rc1.dev335+gc687c1abb.
Local plot: Python 3.9.6, matplotlib 3.9.4.

MECHANICAL VALIDATION
---------------------
All nine thinking-OFF jobs completed with exit 0:0 and 880 valid ratings per
cell. The five thinking-ON zero-shot cells have 880 pair rows; Qwen 9B has 870
valid Likert ratings, Qwen 27B has 878, and the other three have 880. See
validation.txt.

ARTIFACTS
---------
  judge_eval_880.csv
    0859e0bae1da6c439186659d289fed67bfde10a6a54f8278e8a66c25691f7c09
  judge_eval_880_accuracy.png
    a4625f959570396e66af556e03386e9c436441d38582353057a5065b279273ca
  job_accounting.txt
    b0fa919afc4224f643eab8e4f791716e1819ee2521378eb736bafa176d76feef
  validation.txt
    a14a4cb2e023751534157173e8f47e7d30aa685696868be1d8924ec76ab2662d
  plot/plot_judge_eval.py
    317c42c6f2ec45aa1b1118ee621d74cbb196fc947e19dcd5c0f40ef5f4c573fe

The earlier 2026-08-12 plot is retained separately; it evaluated the
thinking-OFF-trained judges with THINKING_MODE=on.

REPRODUCTION
------------
Submit the scoring matrix from a clean retained commit:

  scripts/cluster_launch.sh --dependency-profile eval \
    --run-root /home/lancewicki/projects/turing-rl/results/<date>-judge-only-rlvr-thinking-off-eval \
    --env EVAL_ROOT=/home/lancewicki/projects/turing-rl/results/<date>-judge-only-rlvr-thinking-off-eval \
    --env THINKING_MODE=off --env CONFIRM_THINKING_OFF=1 \
    scripts/launch_judge_eval_matrix.sh

Regenerate the plot from this directory:

  python plot/plot_judge_eval.py --csv judge_eval_880.csv \
    --out judge_eval_880_accuracy.png

Verify artifacts:

  shasum -a 256 -c SHA256SUMS
