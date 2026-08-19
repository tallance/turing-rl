Judge-only RLVR round 1: thinking-OFF held-out evaluation
=========================================================
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

Every cell used THINKING_MODE=off, the full ordered response schema, 8192
maximum completion tokens, and the model generation_config sampling defaults.
The four trained judges and Qwen 4B/9B/Gemma 12B used TP=1 x 8 replicas.
Qwen 27B and Gemma 31B used TP=8 x 1 replica.

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
All nine jobs completed with exit 0:0. Every cell has exactly 880 JSONL rows,
880 unique (user_id, post_id, target_idx) keys, 880 ratings in [1, 7], zero
unparsed records, and zero length-truncated records. See validation.txt.

ARTIFACTS
---------
  judge_eval_880.csv
    9d6429e0ae14433e53ac7d448075fb9a421e3c9a4502ce6f607bc4aed8d080ee
  judge_eval_880_accuracy.png
    8762473c5851d100d50a2876312579e8ad9a5e67fe6e5e55a4b0c0d6b96c5e68
  job_accounting.txt
    3acab820e946aa22a89eb562db37982ef674efa5acead07fb06a4f8051930e76
  validation.txt
    b846e92f64509e728d0f607f18919a30345ef33081dca631ee3f55e6969da6c0
  plot/plot_judge_eval.py
    b361de9e803aaa8e04089e11a9f8b19376b28165c9429c20807465d628e5ed6d

The earlier 2026-08-12 plot is retained separately; it evaluated these
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
