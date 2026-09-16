rating_only judge on the frozen 880-pair held-out set
=====================================================
Four cells added to the judge-accuracy chart: the rating_only thinking-ON judge (J1', GRPO
step 52) and its zero-shot control, each decoded at T=0.7 and T=1.0.

Date: 2026-09-16
Code: branch worktree-thinking-judge, commit 65c1894 (snapshot
      /home/lancewicki/projects/turing-rl-sources/65c189424fa1a3378fc1a06d0123418f9a511c49)


RESULTS (n=880 each, failure_rate 0.000)
----------------------------------------
judge-9b-rating-trained-t07    0.609
judge-9b-rating-trained-t10    0.595
judge-9b-rating-zeroshot-t10   0.541
judge-9b-rating-zeroshot-t07   0.510

accuracy_half_tie = (correct + 0.5*ties)/880, parse failures in the denominator at 0.


FIRST ATTEMPT, DISCARDED
------------------------
Jobs 23558-23561 ran the same four cells and produced 0.691 / 0.700 / 0.716 / 0.728. They were
scored against the WRONG pair set and are not used here.

Cause: scripts/slurm/judge_sweep_cell.sh defaulted PAIRS to
results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet, and that default was taken.
That set's AI turns come from the Qwen3-8B SFT carrying the stop-token masking bug: 267/880
exceed 1000 chars (p50 7803 chars, 1545 words, max repeated 8-gram count p50 88 -- a decoding
loop) against a paired human turn of ~67 chars, and 36% exceed 5x the human length. The chart's
other 26 cells were never scored on it; they use gen_9b-full5ep-step0_880.parquet (gen mean
50.5 chars vs human 67.3, zero generations over 1000 chars, 1.9% over 5x).

The default was changed to the clean set in commit 4b82124, with tests
(tests/test_judge_sweep_default_pairs.py). The July parquet and its generations pickle, and the
Qwen3-8B SFT checkpoint (22G), were deleted from the cluster on 2026-09-16; their .meta.json
files are kept.

Measured difference between the two runs of the same cells, same code, same decode, pair set
the only variable: trained 0.728 -> 0.609, zero-shot 0.691 -> 0.510.


ARTIFACTS
---------
comparison.csv          30 rows: 26 copied verbatim from
                        results/2026-09-01-gemma-single-token-judge/comparison.csv
                        plus the 4 rows in rating_880_clean.csv. All 30 share one
                        pair_source; the merge refuses otherwise.
rating_880_clean.csv    the 4 cells used here (jobs 23585-23588)
rating_880.csv          the 4 DISCARDED cells (jobs 23558-23561), retained for the record
judge_accuracy_880.png  accuracy_half_tie
judge_a_rate_880.png    a_rate_half_tie
make_plot.py            renders both figures from comparison.csv


UPSTREAM CHAIN
--------------
J1' judge training      job 23152, 17h42m, COMPLETED, 1 epoch / 52 steps, graded arm,
                        JUDGE_TASK_WEIGHT=1.0 JUDGE_FORMAT_WEIGHT=0.0,
                        config qwen35_judge_rating_grpo
                        results/2026-09-15-rating-judge-j1/checkpoints/global_step_52
                        (round 1 only: from J2' the weights are the 0.9/0.1 defaults, since
                        the rating_only format score did not exist when this was launched)
training pairs          data/prism/judge/rating_iter1/rating_only/{train,val}.parquet
                        jobs 23068 / 23069; train slice [0.0,0.1), 416 contexts, k=4
                        = 2@T=0.7 + 2@T=1.0, 3328 rows; val 705 contexts k=1 @ T=0.7
                        generator: SFT merged_ep3 (Qwen3.5-9B)
merge to dense          job 23551, gate PASS (A targets=128 scaling=0.5; B 128/128 bit-exact;
                        C 647 non-target unchanged; D mismatched=0 rounding=24)
                        results/2026-09-16-rating-judge-j1-merge2/models/step52/hf_dense
                        (23549/23550 are earlier failed attempts, retained)


EVAL CELLS (used)
-----------------
job    cell                          model                              T
23585  judge-9b-rating-trained-t10   .../step52/hf_dense                1.0
23586  judge-9b-rating-trained-t07   .../step52/hf_dense                0.7
23587  judge-9b-rating-zeroshot-t10  Qwen/Qwen3.5-9B                    1.0
23588  judge-9b-rating-zeroshot-t07  Qwen/Qwen3.5-9B                    0.7

All COMPLETED 0:0, 9m19s-14m27s. JUDGE_PROMPT_STYLE=rating_only, THINKING_MODE=on, TP=1,
REPLICAS=8.

pairs   results/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet
dumps   results/2026-09-16-rating-judge-880-clean/raw/sweep/<cell>/on/rating_only/reward/

DECODE (differs from the other 26 bars; the pair set does not)
  these 4:        {"temperature":<T>,"top_p":0.95,"top_k":20,"min_p":0.0,
                   "presence_penalty":1.5,"repetition_penalty":1.0}
                  via JUDGE_SAMPLING -> PERSONA_JUDGE_SAMPLING (opt-in override added in
                  commit 0a4c34e; unset reproduces the frozen no-wire-override policy)
  other 26 bars:  T=0.6, repetition_penalty=1.1, no other wire override
The figures carry this as a footnote.


REPRODUCE
---------
# merge the trained checkpoint to a dense servable model
bash scripts/cluster_launch.sh --dependency-profile training --run-root <RUNROOT> \
  --env STEP=52 --env RUN_TAG=rating_j1 \
  --env ACTOR_DIR=<...>/2026-09-15-rating-judge-j1/checkpoints/global_step_52/actor \
  --env EVAL_ROOT=<RUNROOT> \
  --env MERGED_EP3=/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  scripts/submit_snapshot_job.sh --export=ALL -- scripts/slurm/merge_grpo_ckpt.sh

# one eval cell per (model, temperature); repeat 4x. PAIRS is explicit ON PURPOSE.
bash scripts/cluster_launch.sh --dependency-profile eval --run-root <EVALROOT> \
  --env MODEL=<dense path | Qwen/Qwen3.5-9B> --env TP=1 --env REPLICAS=8 \
  --env THINKING_MODE=on --env CELL_NAME=<cell> --env JUDGE_PROMPT_STYLE=rating_only \
  --env 'JUDGE_SAMPLING={"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}' \
  --env PAIRS=<...>/2026-08-10-test-eval-9b-full5ep-full-schema/raw/pairs/gen_9b-full5ep-step0_880.parquet \
  --env SWEEP_ROOT=<EVALROOT>/raw/sweep \
  scripts/submit_snapshot_job.sh --gres=gpu:8 --export=ALL -- scripts/slurm/judge_sweep_cell.sh

python <SNAP>/scripts/build_single_token_comparison.py \
  --sweep-root <EVALROOT>/raw/sweep --out <EVALROOT>/rating_880_clean.csv
python make_plot.py


VALIDATION
----------
- merge gate PASS, exit 0
- all 4 cells exit 0:0, n_pairs=880, failure_rate 0.000
- comparison.csv: all 30 rows carry the same pair_source (asserted at merge time)
- 26 historical rows copied unmodified; their plotted values are unchanged against
  results/2026-09-01-gemma-single-token-judge/judge_accuracy_880.png
- local test suite at commit 4b82124: 975 passed, 28 skipped
