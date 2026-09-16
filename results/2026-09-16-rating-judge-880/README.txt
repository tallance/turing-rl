rating_only judge on the frozen 880-pair held-out set
=====================================================
Four cells added to the existing judge-accuracy chart: the rating_only thinking-ON judge
(J1', GRPO step 52) and its zero-shot control, each decoded at T=0.7 and T=1.0.

Date: 2026-09-16
Code: branch worktree-thinking-judge, commit 65c1894 (snapshot
      /home/lancewicki/projects/turing-rl-sources/65c189424fa1a3378fc1a06d0123418f9a511c49)


ARTIFACTS
---------
comparison.csv          30 rows: 26 copied verbatim from
                        results/2026-09-01-gemma-single-token-judge/comparison.csv
                        plus the 4 rows in rating_880.csv
rating_880.csv          the 4 new cells, from build_single_token_comparison.py
judge_accuracy_880.png  accuracy_half_tie = (correct + 0.5*ties)/880, failures wrong
judge_a_rate_880.png    a_rate_half_tie
make_plot.py            renders both figures from comparison.csv


UPSTREAM CHAIN
--------------
J1' judge training      job 23152, 17h42m, COMPLETED
                        results/2026-09-15-rating-judge-j1/checkpoints/global_step_52
                        1 epoch / 52 steps, graded arm, JUDGE_TASK_WEIGHT=1.0
                        JUDGE_FORMAT_WEIGHT=0.0, config qwen35_judge_rating_grpo
                        (NOTE: round 1 only. From J2' the weights are the 0.9/0.1
                        defaults, because the rating_only format score did not exist yet
                        when this run was launched.)
training pairs          data/prism/judge/rating_iter1/rating_only/{train,val}.parquet
                        jobs 23068 (train) / 23069 (val)
                        train: slice [0.0,0.1), 416 contexts, k=4 = 2@T=0.7 + 2@T=1.0,
                               3328 rows, human_is_b_rate 0.5
                        val:   705 contexts, k=1 @ T=0.7, 1410 rows
                        generator: SFT merged_ep3 (Qwen3.5-9B)
merge to dense          job 23551, COMPLETED, gate PASS
                        [A] targets=128 scaling=0.5 unmatched=0
                        [B] verified=128/128 bit-exact worst_err=0 zero_delta=0
                        [C] non-target bit-identical=647 changed=0
                        [D] shared=760 mismatched=0 rounding=24 bad_missing=0 bad_extra=0
                        -> results/2026-09-16-rating-judge-j1-merge2/models/step52/hf_dense
                        (jobs 23549/23550 are earlier failed attempts of the same merge,
                        retained for provenance)


EVAL CELLS
----------
job    cell                          model                              T
23558  judge-9b-rating-trained-t10   .../step52/hf_dense                1.0
23559  judge-9b-rating-trained-t07   .../step52/hf_dense                0.7
23560  judge-9b-rating-zeroshot-t10  Qwen/Qwen3.5-9B                    1.0
23561  judge-9b-rating-zeroshot-t07  Qwen/Qwen3.5-9B                    0.7

All COMPLETED 0:0, 11m55s-13m36s. All: JUDGE_PROMPT_STYLE=rating_only, THINKING_MODE=on,
TP=1, REPLICAS=8, 880 pairs, failure_rate 0.000.

pairs   results/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet
        (sha256 of the same frozen set used by every other bar)
dumps   results/2026-09-16-rating-judge-880/raw/sweep/<cell>/on/rating_only/reward/

DECODE POLICY (differs from every other bar on the chart)
  these 4 cells:  {"temperature":<T>,"top_p":0.95,"top_k":20,"min_p":0.0,
                   "presence_penalty":1.5,"repetition_penalty":1.0}
                  passed via JUDGE_SAMPLING -> PERSONA_JUDGE_SAMPLING
  all other bars: T=0.6, repetition_penalty=1.1, no other wire override
The figures carry this as a footnote. The two groups are not decode-comparable.


REPRODUCE
---------
# 1. merge the trained checkpoint to a dense servable model
bash scripts/cluster_launch.sh --dependency-profile training \
  --run-root <RUNROOT> \
  --env STEP=52 --env RUN_TAG=rating_j1 \
  --env ACTOR_DIR=<...>/2026-09-15-rating-judge-j1/checkpoints/global_step_52/actor \
  --env EVAL_ROOT=<RUNROOT> \
  --env MERGED_EP3=/home/lancewicki/data/hf_cache/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  scripts/submit_snapshot_job.sh --export=ALL -- scripts/slurm/merge_grpo_ckpt.sh

# 2. one eval cell per (model, temperature); repeat 4x
bash scripts/cluster_launch.sh --dependency-profile eval \
  --run-root <EVALROOT> \
  --env MODEL=<dense path | Qwen/Qwen3.5-9B> --env TP=1 --env REPLICAS=8 \
  --env THINKING_MODE=on --env CELL_NAME=<cell> \
  --env JUDGE_PROMPT_STYLE=rating_only \
  --env 'JUDGE_SAMPLING={"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}' \
  --env PAIRS=<...>/2026-07-08-judge-sweep/raw/pairs/prism_heldout_880.parquet \
  --env SWEEP_ROOT=<EVALROOT>/raw/sweep \
  scripts/submit_snapshot_job.sh --gres=gpu:8 --export=ALL -- scripts/slurm/judge_sweep_cell.sh

# 3. summarise and plot
python <SNAP>/scripts/build_single_token_comparison.py \
  --sweep-root <EVALROOT>/raw/sweep --out <EVALROOT>/rating_880.csv
python make_plot.py     # merges nothing; reads comparison.csv in this directory


VALIDATION
----------
- merge gate: PASS (see above), exit 0
- all 4 cells exit 0:0, n_pairs=880 each, failure_rate 0.000
- 26 historical rows copied unmodified; their plotted values are unchanged against
  results/2026-09-01-gemma-single-token-judge/judge_accuracy_880.png
- local test suite at commit 65c1894: 971 passed, 28 skipped
