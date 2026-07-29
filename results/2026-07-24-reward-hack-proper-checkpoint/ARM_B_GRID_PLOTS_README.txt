Arm B completed-cell plots (Qwen3.5-9B, LR=1e-4)
==================================================

Generated: 2026-07-29
Completed jobs:
  11915  9b_proper_kl1e3_lr1e4  KL=1e-3, LR=1e-4
  11916  9b_proper_kl1e4_lr1e4  KL=1e-4, LR=1e-4

Files
-----
winrate_over_time_9b_completed_lr1e4.png
  Two-panel overall win-rate over the 50 actual optimizer updates. Each update
  has 28 reward rows (7 sampled prompts x G=4 rollouts). Rows are timestamp
  sorted and chunked by 28; all 49 update-boundary gaps are larger than every
  within-update gap. Final-update win rates are 0.81 and 0.86, respectively.

9b_proper_kl1e3_lr1e4_rating_scatter.png
9b_proper_kl1e4_lr1e4_rating_scatter.png
  Ten-panel per-example Likert trajectories from plot_overfit_ratings.py.
  Because each update samples 7 of 10 prompts, the x axis is a prompt-appearance
  block (four rollouts) rather than a globally aligned optimizer update.

Gate results
------------
9b_proper_kl1e3_lr1e4: 9/10 prompts won; final prompt-pooled win rate 0.8462
9b_proper_kl1e4_lr1e4: 9/10 prompts won; final prompt-pooled win rate 0.8250

Cluster source paths
--------------------
/home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_proper_kl1e3_lr1e4/reward_dump
/home/lancewicki/projects/turing-rl/results/grpo/rl-generator/9b_proper_kl1e4_lr1e4/reward_dump

Reproduce scatter plots on the cluster
--------------------------------------
  /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python \
    scripts/plot_overfit_ratings.py \
    --dump_dir results/grpo/rl-generator/<tag>/reward_dump \
    --out results/grpo/rl-generator/<tag>/rating_scatter.png \
    --group_size 4

The aggregate plot uses the same score rule as plot_winrate_over_time.py
(win=Likert>=5; Likert 4 and parse-fail 0/None excluded), but globally chunks
the timestamp-sorted rows by 28 to recover the true 50 optimizer updates. The
standard per-example chunker is not used for this figure because randomized
7-of-10 prompt sampling leaves uneven per-prompt row counts and creates a
misleading partial final chunk.
