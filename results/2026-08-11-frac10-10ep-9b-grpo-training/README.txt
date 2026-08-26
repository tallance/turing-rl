Qwen3.5-9B GRPO, 10% of the training split, 20 epochs -- TRAINING-SIDE artifacts
================================================================================
Provenance only. No interpretation of the numbers is recorded here.

ONE training trajectory, TWO Slurm jobs. Job 15371 ran epochs 1-10 (steps 1-60).
Job 18571 resumed from global_step_60 and ran epochs 11-20 (steps 61-120). Both wrote
to the same run directory, so the checkpoints, the reward dump and the subset files are
a single continuous record; only wandb is split across two run ids.

Checkpoint evaluation is performed separately. Epochs 1-10 were evaluated in
  results/2026-08-12-test-eval-9b-train10pct-10ep-test50pct-full-schema/
Epochs 11-20 are not yet evaluated. This directory holds only the training-side record.


1. WHAT WAS RUN
---------------
                        job 15371                      job 18571
state                   COMPLETED 0:0                  COMPLETED 0:0
elapsed                 23:04:47                       21:32:04
window (UTC)            2026-08-11 15:34 -> 08-12 14:39   2026-08-18 17:55 -> 08-19 15:27
steps                   1-60 (epochs 1-10)             61-120 (epochs 11-20)
mode                    MODE=frac10ep10                MODE=frac10ep20
per-step time           1354 s/it                      1264 s/it
nodes                   2 x 8 A100-40GB                2 x 8 A100-40GB

RUN_TAG for both: 9b_frac10_10ep_kl1e4_lr1e4_temp1
  The tag still says "10ep" because reusing it is exactly what makes the resume work:
  trainer.resume_mode=auto locates the checkpoint through trainer.default_local_dir,
  which is derived from RUN_TAG. Renaming it would have started from the SFT init.

Judge for both: Qwen3.5-9B, TP=1 DP=8, concurrency 64, timeout 1800 s.
Full sacct output in job_accounting.txt.


2. CONFIGURATION AS RESOLVED
----------------------------
Identical across both jobs except total_epochs (verbatim invocations in
resolved_trainer_command.txt and resolved_trainer_command_18571.txt):

  data.train_max_samples=384          10% of the 4174-row train split
  data.val_max_samples=352            50% of the 705-row val split
  trainer.save_freq=6                 epoch-aligned (384 // 64 = 6 steps/epoch)
  trainer.test_freq=6                 same grid, so every checkpoint has a val score
  trainer.val_before_train=True
  trainer.max_actor_ckpt_to_keep=null keep all
  trainer.total_epochs                10 (job 15371)  ->  20 (job 18571)

Supplied by qwen3_9b_grpo_turing.yaml rather than the command line: train_batch_size 64,
ppo_mini_batch_size 64, actor.optim.lr 1.0e-4, kl_loss_coef 1.0e-4, rollout temperature
1.0 / top_p 1.0 / top_k -1, and val_kwargs temperature 0.7 / top_p 0.8 / top_k 20 /
do_sample true / n 1. Their absence from the resolved command line is expected and is
itself the evidence that the config path was used rather than submit-time overrides.

Extending total_epochs is safe for the optimizer: lr_scheduler_type defaults to
"constant" with lr_warmup_steps_ratio 0.0, so the LR is flat at 1.0e-4 and does not
depend on total_training_steps. Both jobs logged "num_warmup_steps: 0"; job 18571 logged
"Total steps: 120".


3. MECHANICAL VALIDATION STATUS
-------------------------------
3a. Subset integrity -- PASS. Full transcript in subset_verification.txt.

  Over the COMBINED dump (both reward-*.jsonl files):
      [train] 384 unique keys, 30720 judged calls -- every key seen exactly 80x
      [val]   352 unique keys,  7744 judged calls -- every key seen exactly 22x
  and both key sets reproduce veRL's seeded draw exactly.

  80x = 20 epochs x 4 rollouts. 22x, not 21x: a single continuous 20-epoch run validates
  once before training plus once per epoch boundary (21). Splitting across two jobs adds a
  SECOND val_before_train, at step 60, when the resume re-validates the loaded checkpoint.
  Passing 21 here would fail the check for a reason that is not a defect.

3b. The resume genuinely continued -- PASS. Transcript in resume_verification.txt.

  Only LoRA trains (r=64, alpha=32), so base weights must be identical across checkpoints
  and all drift must appear in the adapter:

      d_base(60,66)  = 0.000000e+00      frozen base is bit-identical

      ||lora||  step 6  18.4978   step 48 18.7575   step 54 18.7904
                step 60 18.8259   step 66 18.8574

      d_lora(48,54) = 0.6532  rel 0.0348   within job 15371
      d_lora(54,60) = 0.6428  rel 0.0342   within job 15371
      d_lora(60,66) = 0.6241  rel 0.0332   ACROSS the resume
      d_lora( 6,60) = 3.3575  rel 0.1815   54-step scale reference

  LoRA B is zero-initialised, so a run that had silently restarted from the SFT init would
  carry a fresh adapter whose norm sat far below this curve; step 66 is on it. The
  across-resume 6-step delta matches the two preceding intra-run deltas within 3%.

  Log-level corroboration (launcher_echoes_18571.txt): "Load from checkpoint folder:
  .../global_step_60", "Setting global step to 60", and the epoch-boundary notice
  "Skipping dataloader state restore: global_steps=60 ... Next epoch will iterate from
  scratch". veRL derives the step counter and the weights from the same folder string, so
  the progress bar resuming at 61/120 also fixes which checkpoint was loaded.

3c. Launcher echoes, both jobs: "selected 384 random samples out of 4174",
  "selected 352 random samples out of 705", "Size of train dataloader: 6",
  "judge concurrency pinned: 64", "trainer exit: 0".


4. VALIDATION TRAJECTORY
------------------------
val_trajectory.csv / .json -- 22 rows, metric
val-core/prism_alignment_user_sim/reward/mean@1, one per epoch boundary plus two
val_before_train points. Each row records which source it came from.

SOURCING: wandb dropped the FINAL point of BOTH jobs. Its series stop at step 54 and step
114 respectively, and both runs are marked "crashed" on the server even though sacct
reports COMPLETED 0:0 for each. The step-60 and step-120 values were recovered from the
"Final validation metrics" block in the training logs. Neither job's local wandb directory
survives -- both wrote to /tmp on their compute node. wandb plus the training log are the
only two sources, and neither alone is complete.

  wandb run ids: u5qaz45l (job 15371), clzo0hpw (job 18571),
  project lancewicki/2026-07-15-rl-generator-vs-fixed-judge on meta.wandb.io

STEP 60 APPEARS TWICE, and the pair is worth keeping:
      0.5994   job 15371, after its last optimizer step
      0.5747   job 18571, val_before_train on the SAME reloaded weights
The 0.0247 gap is therefore not a change in the model -- section 3b establishes the weights
are the step-60 checkpoint. It is a repeat measurement of one fixed policy, so it bounds
how much of any epoch-to-epoch movement in this metric is measurement rather than learning.
Both the generator (temperature 0.7) and the judge (temperature 0.6) resample, and job
18571 ran against a freshly started judge server.


5. FILES HERE
-------------
README.txt                              this file
val_trajectory.csv/.json                22 val points, with per-point source and segment
subset_verification.txt                 verbatim output of both verification scripts
resume_verification.txt                 tensor-level check that 18571 continued 15371
train_used_keys.tsv                     384 (user_id, post_id, target_idx) keys, sorted
val_used_keys.tsv                       352 keys, sorted
checkpoints.txt                         inventory: 20 checkpoints, 19G each, 365G total
job_accounting.txt                      sacct for both jobs
resolved_trainer_command.txt            verbatim Hydra invocation, job 15371
resolved_trainer_command_18571.txt      verbatim Hydra invocation, job 18571
*_tokens.txt                            same, one token per line (diff-friendly)
launcher_echoes.txt                     pinned-value lines, job 15371
launcher_echoes_18571.txt               pinned-value lines + resume markers, job 18571
cluster_SHA256SUMS.txt                  checksums of the large cluster-resident files
source_sha.txt                          deployed source SHA
SHA256SUMS                              checksums of the files in this directory

The key sets are unchanged between the two jobs -- the same 384/352 rows -- so
train_used_keys.tsv and val_used_keys.tsv cover both.

Deliberately NOT copied here: the two reward dumps (1.5 GB each) and the subset parquets
(7.4 MB each). They stay on the cluster; cluster_SHA256SUMS.txt pins them and the key
TSVs carry the same sample identity in text form.


6. CLUSTER SOURCE PATHS
-----------------------
Repo         /home/lancewicki/projects/turing-rl
Run dir      $REPO/results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1
Checkpoints  $RUN/checkpoints/global_step_{6,12,...,120}
Reward dumps $RUN/reward_dump/reward-15371-1556966.jsonl
             $RUN/reward_dump/reward-18571-3181616.jsonl
Subsets      $RUN/train_used384.parquet, $RUN/val_used352.parquet
Logs         $REPO/logs/rl_gen-15371.out
             $REPO/results/runs/frac10ep20-resume-4/logs/slurm-rl_gen-18571.out
Python       /home/lancewicki/miniconda3/envs/turing-rl-train/bin/python

Job 18571 ran under the immutable-snapshot workflow, so its log lives beneath its run
root rather than in $REPO/logs. Job 15371 predates that migration.


7. REPRODUCTION
---------------
Epochs 1-10, from the SFT init:

  scripts/cluster_launch.sh --dependency-profile training \
    --run-root /home/lancewicki/projects/turing-rl/results/runs/<name> \
    --env MODE=frac10ep10 --env JUDGE=9b \
    --env RUN_TAG=9b_frac10_10ep_kl1e4_lr1e4_temp1 \
    scripts/submit_snapshot_job.sh --export=ALL -- scripts/slurm/rl_generator_run_9b.sh

Epochs 11-20: the SAME command with MODE=frac10ep20 and the SAME RUN_TAG. Reusing the tag
is what triggers the resume; a fresh tag trains from the SFT init instead. EXTRA_OVERRIDES
must be empty -- the arm hard-fails (exit 5) if it sets any key the arm pins.

Re-run the subset verification:

  cd /home/lancewicki/projects/turing-rl
  RUN=results/grpo/rl-generator/9b_frac10_10ep_kl1e4_lr1e4_temp1
  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  PYTHONPATH=. $PY scripts/build_eval_subsets.py \
    --grpo_dir data/prism/full_s42_history_sft40_grpo60_test10/grpo \
    --reward_dump $RUN/reward_dump --train_max_samples 384 --val_max_samples 352 \
    --seed 42 --train_out $RUN/train_used384.parquet --val_out $RUN/val_used352.parquet
  PYTHONPATH=. $PY scripts/verify_run_repetition.py \
    --reward_dump $RUN/reward_dump --expect-train 384 --expect-val 352

  Note --train_out/--val_out: the defaults write into the shared grpo data dir and would
  overwrite the half-data run's val_used352.parquet.

Re-fetch the val trajectory (needs meta.wandb.io credentials in ~/.netrc). Remember each
run's final point is missing from the server and must be read from its log:

  tr '\r' '\n' < <log> | grep -A1 'Final validation metrics'


8. HOW THIS DIRECTORY WAS BUILT
-------------------------------
Epochs 1-10 assembled 2026-08-17; extended to cover epochs 11-20 on 2026-08-19. Pulled
from the cluster over the SSH tunnel (ssh -p 2223 lancewicki@localhost). Every file except
README.txt and SHA256SUMS is machine-generated cluster output copied verbatim; no file here
was hand-edited after copying.
