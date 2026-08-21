Qwen3.5-9B GRPO, 10% train split, 20 epochs, GEMMA-4-12B JUDGE -- training-side artifacts
=========================================================================================
Provenance only. No interpretation of the numbers is recorded here.

The judge-swap counterpart to results/2026-08-11-frac10-10ep-9b-grpo-training/, which is
the same recipe judged by Qwen3.5-9B. Identical in every respect except the training
judge: same SFT init, same 384-row train slice, same 352-row val subset at seed 42, same
20 epochs / 120 steps / 6-step checkpoint grid, same lr 1e-4 and kl_loss_coef 1e-4.

Do NOT compare the two arms' reward values directly. A different judge is a different
reward scale; only within-arm trends are comparable across them.

ONE trajectory, TWO Slurm jobs, because a cluster-wide event killed the first at step 72.


1. WHAT WAS RUN
---------------
                  job 18669                        job 18771
state             FAILED 1:0 at step 72            COMPLETED 0:0
elapsed           21:06:17                         14:02:41
window (UTC)      2026-08-19 16:09 -> 08-20 13:15  2026-08-21 04:36 -> 18:39
steps             1-72 (epochs 1-12)               73-120 (epochs 13-20)
per-step time     ~1107 s/it                       1023 s/it

RUN_TAG for both: 9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1
MODE=frac10ep20, JUDGE=gemma4-12b for both. Reusing the tag is what makes the resume work:
resume_mode=auto finds the checkpoint via default_local_dir, which derives from RUN_TAG.

Judge: google/gemma-4-12B-it, TP=1 DP=8, --reasoning-parser gemma4, concurrency 64,
served offline from pinned snapshot 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 out of the
turing-rl-gemma4-vllm-nightly environment. Launched with --dependency-profile all, the
only profile carrying both the veRL tree and the Gemma environment.

WHY 18669 FAILED -- not this job's fault. Four unrelated jobs died within 12 seconds of
each other (13:15:30 judge_grpo, 13:15:34 judge_grpo, 13:15:36 te_t20t50_gemma4-12b,
13:15:42 this run), on different nodes, across different workflows. No exception in
either stream; the log stops mid-judge-traffic on HTTP 200s. Memory was not the cause
(347 GB peak against 1 T requested) and no ENOSPC appears in the log, though a full
shared filesystem would produce exactly this simultaneous signature. Treated as a
cluster-level event.


2. MECHANICAL VALIDATION STATUS
-------------------------------
2a. Which samples -- PASS. Both subsets reproduce veRL's seeded draw exactly:
    384 train keys / 277 users, 352 val keys / 349 users.

2b. Repetition counts -- val PASS, train UNEVEN BY DESIGN OF THE CRASH.
    Full transcript in subset_verification.txt.

        [val]   352 unique keys, 7744 judged calls -- every key exactly 22x   OK
        [train] 384 unique keys, 30973 judged calls -- counts range 79..84    flagged

    The script reports UNEVEN and attributes it to a rotating per-epoch sample set. That
    diagnosis is wrong here, and the split by source file shows why:

        job 18771: 12288 train calls = 384 x 32 exactly    (8 epochs x 4 rollouts)
        job 18669: 18685 train calls, vs 18432 expected    (12 epochs x 4 rollouts)
                   excess = 253, concentrated in exactly 64 keys

    64 keys is exactly one training batch. Job 18669 had begun judging step 73 when the
    cluster event hit; those judge calls reached the dump, then the resume restarted from
    checkpoint 72 and discarded the rollouts. So the dump carries 253 extra LOG LINES that
    never contributed a gradient. One key sits at 79x -- a single dump line lost to the
    process dying mid-write.

    Training itself is even: 12 complete epochs from 18669 plus 8 from 18771 = 20, every
    sample seen exactly 20 times at 4 rollouts each. The unevenness is an artifact of
    crash-and-resume in the log, not of the sample set.

    22x per val key = val_before_train plus 20 epoch boundaries (21), plus a second
    val_before_train at step 72 when 18771 re-validated the reloaded checkpoint.

2c. The resume genuinely continued -- PASS. Transcript in resume_verification.txt.
    Only LoRA trains (r=64, alpha=32), so the frozen base must be identical and all drift
    must land in the adapter:

        d_base(72,78) = 0.000000e+00

        ||lora||  step 60 18.7891   step 66 18.8144   step 72 18.8358   step 78 18.8600

        d_lora(60,66) = 0.5679  rel 0.0302   within job 18669
        d_lora(66,72) = 0.5609  rel 0.0298   within job 18669
        d_lora(72,78) = 0.5580  rel 0.0296   ACROSS the resume

    LoRA B is zero-initialised, so a run that had restarted from the SFT init would carry
    a fresh adapter far below this curve. The across-resume delta matches the two
    preceding intra-run deltas within 2%. Log corroboration in launcher_echoes_18771.txt:
    "Load from checkpoint folder: .../global_step_72" and "Setting global step to 72".

2d. Judge health -- the reason this arm needed a smoke test first. Gemma's clean record
    was earned under the eval sweep's 37-field ordered schema, which training does not
    enable (reward.py falls back to {"type":"json_object"}), and
    docs/judge-response-schema.md records gemma-4-31B collapsing to 5/16 valid without it.
    Measured live during job 18669, at 7496 judged calls:

        parsed rate           1.0000
        hard-fail rate        0.0000
        finish_reason=length  0.0000   (all 7496 = "stop")

    So the ordered schema was not needed, and this arm differs from the Qwen arm in the
    judge alone rather than in the judge plus the response format.

2e. Judge-independent pipeline check at step 0. The format metrics are computed from the
    generated turn, not the judge verdict, so with the same SFT init on the same 352 val
    prompts they must match the Qwen arm regardless of who scores:

        format_score/mean@1        0.1000 vs 0.1000 reference   match
        num_turns/mean             2.0    vs 2.0                match
        length_generated_words     9.9744 vs 10.5653 reference   -0.59

    The -0.59 is below this metric's own within-run step-to-step variation in the Qwen arm
    (mean 0.778, max 1.875), so it is sampling noise rather than a pipeline difference.


3. VALIDATION TRAJECTORY
------------------------
val_trajectory.csv / .json -- 21 rows, metric
val-core/prism_alignment_user_sim/reward/mean@1. Each row records its source.

21 rows and not 22, unlike the Qwen arm: job 18669 DID complete its step-72 validation
(the dump contains all 22 passes) but crashed before that value reached wandb, and only
the "Initial"/"Final" validation blocks print to stdout -- the per-epoch values are
wandb-only. So 18669's step-72 number is lost from both sources and the step-72 row here
is 18771's re-validation alone. Consequence: this arm has no same-weights duplicate
measurement, so it lacks the empirical noise floor the Qwen arm's doubled step-60 point
provides.

wandb also dropped the FINAL point of the completed job, as it did for both Qwen jobs:
run tsuj3tr0's series stops at step 114 and the step-120 value came from the training log.

  wandb run ids: w7lc0rhg (job 18669), tsuj3tr0 (job 18771),
  project lancewicki/2026-07-15-rl-generator-vs-fixed-judge on meta.wandb.io


4. FILES HERE
-------------
README.txt                          this file
val_trajectory.csv/.json            21 val points, with per-point source and segment
subset_verification.txt             verbatim output of both verification scripts
resume_verification.txt             tensor-level check that 18771 continued 18669
checkpoints.txt                     inventory: 20 checkpoints, 19G each, 365G total
job_accounting.txt                  sacct for both jobs
resolved_trainer_command_*.txt      verbatim Hydra invocations, per job
launcher_echoes_*.txt               pinned-value lines and resume markers, per job
cluster_SHA256SUMS.txt              checksums of the large cluster-resident files
SHA256SUMS                          checksums of the files in this directory

Not copied here: the two reward dumps (~1.5 GB each) and the subset parquets. They stay on
the cluster and cluster_SHA256SUMS.txt pins them. The train/val key sets are identical to
the Qwen arm's, so train_used_keys.tsv / val_used_keys.tsv in
results/2026-08-11-frac10-10ep-9b-grpo-training/ cover this arm too.


5. CLUSTER SOURCE PATHS
-----------------------
Repo         /home/lancewicki/projects/turing-rl
Run dir      $REPO/results/grpo/rl-generator/9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1
Checkpoints  $RUN/checkpoints/global_step_{6,12,...,120}
Reward dumps $RUN/reward_dump/reward-18669-*.jsonl, reward-18771-*.jsonl
Logs         $REPO/results/runs/frac10ep20-gemma12b/logs/slurm-rl_gen-18669.{out,err}
             $REPO/results/runs/frac10ep20-gemma12b-resume/logs/slurm-rl_gen-18771.{out,err}
Judge model  /home/lancewicki/data/hf_cache/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7


6. REPRODUCTION
---------------
  scripts/cluster_launch.sh --dependency-profile all \
    --run-root /home/lancewicki/projects/turing-rl/results/runs/<name> \
    --env MODE=frac10ep20 --env JUDGE=gemma4-12b \
    --env RUN_TAG=9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1 \
    scripts/submit_snapshot_job.sh --export=ALL -- scripts/slurm/rl_generator_run_9b.sh

A fresh RUN_TAG trains from the SFT init; reusing a tag that already has checkpoints
resumes from the latest one. That single command therefore covers both jobs here.

Acceptance gates for the judge itself: scripts/slurm/gemma4_judge_training_smoke.sh.

Re-run the subset verification:

  cd /home/lancewicki/projects/turing-rl
  RUN=results/grpo/rl-generator/9b_frac10_20ep_gemma12b_kl1e4_lr1e4_temp1
  PY=/home/lancewicki/miniconda3/envs/turing-rl-train/bin/python
  PYTHONPATH=. $PY scripts/build_eval_subsets.py \
    --grpo_dir data/prism/full_s42_history_sft40_grpo60_test10/grpo \
    --reward_dump $RUN/reward_dump --train_max_samples 384 --val_max_samples 352 \
    --seed 42 --train_out $RUN/train_used384.parquet --val_out $RUN/val_used352.parquet
  PYTHONPATH=. $PY scripts/verify_run_repetition.py \
    --reward_dump $RUN/reward_dump --expect-train 384 --expect-val 352

  It will report train counts UNEVEN. See section 2b before treating that as a defect.


7. HOW THIS DIRECTORY WAS BUILT
-------------------------------
Assembled 2026-08-21, pulled from the cluster over the SSH tunnel
(ssh -p 2223 lancewicki@localhost). Every file except README.txt and SHA256SUMS is
machine-generated cluster output copied verbatim; no file here was hand-edited after
copying.
