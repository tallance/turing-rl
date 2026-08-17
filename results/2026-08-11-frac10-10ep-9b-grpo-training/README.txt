Qwen3.5-9B GRPO, 10% of the training split, 10 epochs -- TRAINING-SIDE artifacts
================================================================================
Provenance only. No interpretation of the numbers is recorded here.

Checkpoint evaluation for this run was performed separately and lives in
  results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema/
This directory holds only what that one does not: the training run's own record.


1. WHAT WAS RUN
---------------
Slurm job     15371  ("rl_gen")
State         COMPLETED, ExitCode 0:0
Elapsed       23:04:47
Start / End   2026-08-11T15:34:27Z / 2026-08-12T14:39:14Z
Nodes         a100-147-100, a100-208-0+  (2 nodes, gres/gpu=16, mem=1T, cpu=192)
Progress bar  60/60 [22:34:04<00:00, 1354.07s/it]

Source SHA deployed on the cluster: c76ce84e2e4969fa0d58e23cb52969bc72a17f32
  (read from /home/lancewicki/projects/turing-rl/DEPLOYED_SHA on 2026-08-17;
   the cluster tree is a deploy target, not a checkout, so this is the value
   currently stamped there rather than one recorded at launch time.)

Launcher      scripts/slurm/rl_generator_run_9b.sh  MODE=frac10ep10
              -> scripts/slurm/rl_generator_train_9b.sh
veRL config   --config-dir training/grpo/configs --config-name qwen3_9b_grpo_turing


2. CONFIGURATION AS RESOLVED
----------------------------
Pinned by the frac10ep10 MODE arm (present on the command line; see
resolved_trainer_command.txt for the verbatim invocation):

  data.train_max_samples=384          10% of the 4174-row train split
  data.val_max_samples=352            50% of the 705-row val split
  trainer.total_epochs=10
  trainer.save_freq=6                 epoch-aligned (384 // 64 = 6 steps/epoch)
  trainer.test_freq=6                 same grid, so every checkpoint has a val score
  trainer.val_before_train=True       adds the step-0 val point
  trainer.max_actor_ckpt_to_keep=null keep all
  PERSONA_ENABLE_EPOCH_END_CHECKPOINTING=0

Supplied by qwen3_9b_grpo_turing.yaml, NOT by the command line:

  data.train_batch_size            64
  actor.ppo_mini_batch_size        64
  actor.optim.lr                   1.0e-4
  actor.use_kl_loss                true
  actor.kl_loss_coef               1.0e-4
  rollout.temperature              1.0
  rollout.top_p                    1.0
  rollout.top_k                    -1
  rollout.val_kwargs               temperature 0.7, top_p 0.8, top_k 20,
                                   do_sample true, n 1

  These twelve values were passed via EXTRA_OVERRIDES at submit time on every
  earlier 9B run. They were moved into a config file in commit b382861 after job
  15143 was launched without them and trained at the 8B defaults. Their absence
  from resolved_trainer_command.txt is therefore expected, and is itself the
  evidence that the config path was used.

Other resolved values:

  model.path                       checkpoints/sft/qwen35_9b_prism_full_s42_bf16_fsdp_nopack_epochsave/merged_ep3
  lora_rank / lora_alpha           64 / 32   (lora.merge=True, lora_adapter_path=null)
  rollout.max_model_len            13524
  rollout.gpu_memory_utilization   0.55
  ppo_micro_batch_size_per_gpu     1
  data.train_files                 data/prism/full_s42_history_sft40_grpo60_test10/grpo/train.parquet
  data.val_files                   data/prism/full_s42_history_sft40_grpo60_test10/grpo/val.parquet
  judge concurrency                64, timeout 1800 s   (echoed by the launcher)


3. MECHANICAL VALIDATION STATUS
-------------------------------
Subset integrity -- PASS. Full transcript in subset_verification.txt.

  scripts/build_eval_subsets.py replays veRL's seeded draw
  (np.random.default_rng(42).choice, rl_dataset.py:186) and compares the result
  against the (user_id, post_id, target_idx) keys the run actually judged:
      [train] 384 rows / 277 users -- key set matches the reward dump exactly
      [val]   352 rows / 349 users -- key set matches the reward dump exactly

  scripts/verify_run_repetition.py counts occurrences per key:
      [train] 384 unique keys, 15360 judged calls -- every key seen exactly 40x
      [val]   352 unique keys,  3872 judged calls -- every key seen exactly 11x

  40x = 10 epochs x 4 rollouts per prompt; 11x = 11 validation passes
  (val_before_train plus test_freq=6 at steps 6..60). Even counts at an epoch
  boundary are what rules out a rotating drop_last remainder.

Val subset identity across runs -- the val_used352.parquet this run produced is
byte-identical to the one the half-data run wrote on 2026-08-05 in the shared
grpo data dir; both hash to
  41611b0818935728e3408f266dccfa95b748beea00016b28e1748448d560401b
i.e. seed 42 with val_max_samples=352 drew the same 352 val rows in both runs.

Launcher echoes from logs/rl_gen-15371.out (launcher_echoes.txt):
      selected 384 random samples out of 4174
      selected 352 random samples out of 705
      Size of train dataloader: 6, Size of val dataloader: 1
      === judge concurrency pinned: 64 (timeout 1800s) ===
      === trainer exit: 0 ===


4. VALIDATION TRAJECTORY
------------------------
val_trajectory.csv / .json -- 11 points, metric
val-core/prism_alignment_user_sim/reward/mean@1, one per epoch boundary.

Sourcing note: 10 of the 11 points (steps 0-54) came from the wandb API, run
u5qaz45l in project lancewicki/2026-07-15-rl-generator-vs-fixed-judge on
meta.wandb.io. The step-60 point is NOT on the wandb server -- wandb marked the
run "crashed" (sacct still reports COMPLETED 0:0), so the final point was
recovered from the "Final validation metrics" block in the training log. Each
row's `source` column records which of the two it came from.

The run's local wandb directory does not survive: the process wrote to
/tmp/wandb-15371 on the compute node, and /home/lancewicki/projects/turing-rl/
wandb/joblocal-15371 is an empty directory. wandb + the training log are the
only two sources for these numbers.


5. FILES HERE
-------------
README.txt                          this file
val_trajectory.csv/.json            11 val points, with per-point source
subset_verification.txt             verbatim output of both verification scripts
train_used_keys.tsv                 384 (user_id, post_id, target_idx) keys, sorted
val_used_keys.tsv                   352 keys, sorted
checkpoints.txt                     inventory: 10 checkpoints, 19G / 32 files each, 183G total
job_accounting.txt                  full sacct output
resolved_trainer_command.txt        verbatim Hydra invocation from the log
resolved_trainer_command_tokens.txt same, one token per line (diff-friendly)
launcher_echoes.txt                 pinned-value / sample-count lines from the log
cluster_SHA256SUMS.txt              checksums of the large cluster-resident files
SHA256SUMS                          checksums of the files in this directory

Deliberately NOT copied here: the 1.5 GB reward dump and the two subset
parquets (7.4 MB each). The largest file otherwise tracked under results/ is
97 KB. They stay on the cluster; cluster_SHA256SUMS.txt pins them, and
train/val_used_keys.tsv carries the same sample identity in text form.


6. CLUSTER SOURCE PATHS
-----------------------
Repo        /home/lancewicki/projects/turing-rl
Run dir     $REPO/results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1
Checkpoints $RUN/checkpoints/global_step_{6,12,...,60}
Reward dump $RUN/reward_dump/reward-15371-1556966.jsonl   (1542352660 bytes)
Subsets     $RUN/train_used384.parquet, $RUN/val_used352.parquet
Log         $REPO/logs/rl_gen-15371.out                   (10139638 bytes)
Python      /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
wandb run   https://meta.wandb.io/lancewicki/2026-07-15-rl-generator-vs-fixed-judge/runs/u5qaz45l


7. REPRODUCTION
---------------
Submit the training run (from the cluster repo, code at the SHA above):

  MODE=frac10ep10 RUN_TAG=9b_frac10_10ep_kl1e4_lr1e4_temp1 \
    sbatch scripts/slurm/rl_generator_run_9b.sh

  EXTRA_OVERRIDES must be empty: the frac10ep10 arm hard-fails (exit 5) if it
  sets any of the keys the arm pins.

Re-run the subset verification against the dump:

  cd /home/lancewicki/projects/turing-rl
  RUN=results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1
  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

  PYTHONPATH=. $PY scripts/build_eval_subsets.py \
    --grpo_dir data/prism/full_s42_history_sft40_grpo60_test10/grpo \
    --reward_dump $RUN/reward_dump \
    --train_max_samples 384 --val_max_samples 352 --seed 42 \
    --train_out $RUN/train_used384.parquet --val_out $RUN/val_used352.parquet

  PYTHONPATH=. $PY scripts/verify_run_repetition.py \
    --reward_dump $RUN/reward_dump --expect-train 384 --expect-val 352

  Note the --train_out/--val_out paths: the script's defaults write into the
  shared grpo data dir and would overwrite the half-data run's val_used352.parquet.

Re-fetch the val trajectory (needs the meta.wandb.io credentials in ~/.netrc):

  WANDB_BASE_URL=https://meta.wandb.io $PY -c "
  import wandb; r = wandb.Api().run('lancewicki/2026-07-15-rl-generator-vs-fixed-judge/runs/u5qaz45l')
  k = 'val-core/prism_alignment_user_sim/reward/mean@1'
  print([(h['_step'], h[k]) for h in r.history(keys=[k], pandas=False, samples=10000) if h.get(k) is not None])"

  This returns 10 points. The step-60 point must be read from the log:
  tr '\r' '\n' < logs/rl_gen-15371.out | grep -A1 'Final validation metrics'


8. HOW THIS DIRECTORY WAS BUILT
-------------------------------
Assembled 2026-08-17 by pulling from the cluster over the SSH tunnel
(ssh -p 2223 lancewicki@localhost). Every file except README.txt and SHA256SUMS
is machine-generated cluster output copied verbatim; no file here was
hand-edited after copying.
